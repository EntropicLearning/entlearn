"""Private classification-head parameters and tensor calculations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from entlearn.network.data import _ClassificationSupervision, _StagedData
from entlearn.network.state import _ClassificationParameters
from entlearn.network.transfer import copy_log, copy_tensor
from entlearn.primitives.normalise import _scratch_view, floored_log_, normalise_
from entlearn.primitives.output import (
    classification_output_accumulate_into_source_cost_,
    geometric_probabilities_,
    propagate_classification_output_,
    update_classification_output_,
)
from entlearn.primitives.reductions import _chunked_product_
from entlearn.recipe import ClassificationHead, Coupling

if TYPE_CHECKING:
    from entlearn.network.session import _FitSession, _Workspace


@dataclass
class _ClassificationBlock:
    """Fitted transition parameters of a classification head.

    ``weighted_target`` already carries the target, sample and class weights, so every
    call into the output primitives passes ``None`` for the effective output weights.
    ``cache`` holds one fit's co-occurrence counts: ``stage_cache_`` allocates it when a fit
    session opens and ``release_cache`` drops it at publication. ``initial_widths`` holds
    the class and source cluster counts the head was created with, which fix the prior's
    per-entry pseudocount through pruning.
    """

    description: ClassificationHead
    theta: torch.Tensor
    log_theta: torch.Tensor
    initial_widths: tuple[int, int]
    cache: torch.Tensor | None = field(default=None, repr=False)  # (M, K_source) for Πᵀ Γ
    target_sums_fresh: bool = field(default=False, repr=False)  # cache matches the affiliations

    @property
    def output_width(self) -> int:
        """Return the classification output dimension (number of classes)."""
        return self.theta.shape[0]

    @property
    def pseudocount(self) -> float:
        """Return the per-entry Dirichlet prior, scaled by the class and source-cluster counts."""
        M, K_source = self.initial_widths
        return (self.description.theta_alpha - 1.0) / (M * K_source)

    @classmethod
    def make_block(
        cls,
        description: ClassificationHead,
        data: _StagedData,
        K_source: int,
        parameters: _ClassificationParameters | None = None,
    ) -> _ClassificationBlock:
        """Construct the head from a resolved group or its uniform starting transition."""
        M = data.schema.M  # Number of classes
        if parameters is None:
            theta = torch.full(
                (M, K_source),
                1.0 / (M if description.coupling is Coupling.M else K_source),
                dtype=data.X_cont.dtype,
                device=data.X_cont.device,
            )
        else:
            theta = parameters.theta.detach().clone()
        log_theta = torch.empty_like(theta)
        floored_log_(log_theta, theta)
        return cls(
            description=description,
            theta=theta,
            log_theta=log_theta,
            initial_widths=(M, K_source),
        )

    def stage_cache_(self, _data: _StagedData) -> None:
        """Allocate this fit's co-occurrence counts for the active clusters."""
        self.cache = torch.empty_like(self.theta)
        self.target_sums_fresh = False

    def release_cache(self) -> None:
        """Drop the co-occurrence counts, keeping the fitted parameters."""
        self.cache = None
        self.target_sums_fresh = False

    def target_sums(self, session: _FitSession) -> torch.Tensor:
        """Return the co-occurrence counts ``Πᵀ Γ``, formed once per assignment."""
        assert self.cache is not None
        counts = _scratch_view(self.cache, self.theta.shape[1])
        if not self.target_sums_fresh:
            supervision = session.data.supervision
            assert isinstance(supervision, _ClassificationSupervision)
            gamma = session.graph.source_affiliations(self)
            _chunked_product_(counts, supervision.weighted_target, gamma, session.workspace.matrix)
            self.target_sums_fresh = True
        return counts

    def copy_active(
        self, dtype: torch.dtype, device: torch.device, _row_weights: torch.Tensor | None
    ) -> _ClassificationBlock:
        """Copy the class transition and its log.

        A head holds no row-bound state, so ``_row_weights`` is not used.
        """
        theta = copy_tensor(self.theta, dtype, device)
        log_theta = copy_log(self.log_theta, theta)
        return replace(self, theta=theta, log_theta=log_theta, cache=None, target_sums_fresh=False)

    def capture_parameters(self) -> _ClassificationParameters:
        """Copy the fitted class transition and its class count."""
        return _ClassificationParameters(
            replace(self.description, n_classes=self.output_width),
            self.theta.detach().clone(),
        )

    def prune_source(self, keep: torch.Tensor, scratch_M: torch.Tensor) -> None:
        """Compact the source axis after the source block for this head has pruned.

        Args:
            keep: ``(K_source,)`` boolean mask of the surviving source clusters.
            scratch_M: ``(M,)`` workspace, used only under S coupling.
        """
        index = keep.nonzero(as_tuple=True)[0]
        self.theta = torch.index_select(self.theta, 1, index)
        self.log_theta = torch.index_select(self.log_theta, 1, index)
        if self.description.coupling is Coupling.S:
            normalise_(self.theta, 1, scratch_M[: self.theta.shape[0]])
        floored_log_(self.log_theta, self.theta)

    def arithmetic_readout_(
        self,
        out: torch.Tensor,
        gamma_source: torch.Tensor,
        workspace: _Workspace,
    ) -> None:
        """Write the class distribution, calculated via arithmetic averaging."""
        propagate_classification_output_(
            out,
            gamma_source,
            self.theta,
            workspace.row_keepdim[: out.shape[0]],
        )

    def logits_(self, out: torch.Tensor, gamma_source: torch.Tensor) -> None:
        """Write unscaled geometric logits shared by prediction and calibration."""
        torch.mm(gamma_source, self.log_theta.transpose(0, 1), out=out)

    def geometric_readout_(
        self,
        out: torch.Tensor,
        gamma_source: torch.Tensor,
        delta: float,
        epsilon_P: float,
        workspace: _Workspace,
    ) -> None:
        """Write the class distribution, calculated via geometric averaging at the supplied output temperature."""
        self.logits_(out, gamma_source)
        geometric_probabilities_(
            out,
            out,
            delta,
            epsilon_P,
            workspace.row_keepdim,
            workspace.row_indices,
        )

    def update_parameters_(self, session: _FitSession) -> None:
        """Set the head transition matrix from the co-occurrence counts of the source."""
        M, K = self.theta.shape
        norm_dim = 0 if self.description.coupling is Coupling.M else 1
        scratch = session.workspace.scratch_K
        update_classification_output_(
            self.theta,
            self.log_theta,
            self.target_sums(session),
            scratch[:K] if norm_dim == 0 else scratch[:M],
            norm_dim=norm_dim,
            pseudocount=self.pseudocount,
        )

    def accumulate_into_source_cost_(
        self,
        session: _FitSession,
        *,
        cost: torch.Tensor,
    ) -> None:
        """Add this head's coupling term to the source assignment cost accumulator."""
        data = session.data
        supervision = data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        M = self.theta.shape[0]
        classification_output_accumulate_into_source_cost_(
            cost,
            self.log_theta,
            supervision.weighted_target,
            session.graph.head_delta,
            None,
            _scratch_view(session.workspace.log_matrix, M),
        )

    def partial_loss_(self, session: _FitSession) -> float:
        """Add this block's classification loss ``-δ·⟨log θ, Πᵀ Γ + a⟩`` to the total loss.

        where ``a`` is the pseudocount, absent if ``theta_alpha = 1``.
        Returns zero: the head holds no entropy penalty.
        """
        workspace = session.workspace
        counts = self.target_sums(session)
        torch.dot(self.log_theta.view(-1), counts.view(-1), out=workspace.scalar)
        workspace.loss.add_(workspace.scalar, alpha=-session.graph.head_delta)
        pseudocount = self.pseudocount
        if pseudocount != 0.0:
            torch.sum(self.log_theta, dim=(0, 1), out=workspace.scalar)
            workspace.loss.add_(workspace.scalar, alpha=-session.graph.head_delta * pseudocount)
        return 0.0
