"""Private classification-head parameters and tensor calculations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from entlearn.network.data import _ClassificationSupervision, _StagedData
from entlearn.network.state import _ClassificationParameters
from entlearn.network.transfer import copy_log, copy_tensor
from entlearn.primitives.normalise import floored_log_, normalise_
from entlearn.primitives.output import (
    classification_output_accumulate_into_source_cost_,
    classification_output_loss_,
    geometric_probabilities_,
    propagate_classification_output_,
    update_classification_output_,
)
from entlearn.recipe import ClassificationHead, Coupling

if TYPE_CHECKING:
    from entlearn.network.session import _FitSession, _Workspace


@dataclass
class _ClassificationBlock:
    """Fitted transition parameters of a classification head.

    ``weighted_target`` already carries the target, sample and class weights, so every
    call into the output primitives passes ``None`` for the effective output weights.
    """

    description: ClassificationHead
    theta: torch.Tensor
    log_theta: torch.Tensor

    @property
    def output_width(self) -> int:
        """Return the classification output dimension (number of classes)."""
        return self.theta.shape[0]

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
        return cls(description=description, theta=theta, log_theta=log_theta)

    def copy_active(
        self, dtype: torch.dtype, device: torch.device, _row_weights: torch.Tensor | None
    ) -> _ClassificationBlock:
        """Copy the class transition and its log.

        A head holds no row-bound state, so ``_row_weights`` is not used.
        """
        theta = copy_tensor(self.theta, dtype, device)
        return replace(self, theta=theta, log_theta=copy_log(self.log_theta, theta))

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
        """Set the head transition matrix from the weighted target and source affiliations."""
        data = session.data
        supervision = data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        workspace = session.workspace
        T = data.X_cont.shape[0]
        M, K = self.theta.shape
        norm_dim = 0 if self.description.coupling is Coupling.M else 1
        norm_scratch = workspace.scratch_K[:K] if norm_dim == 0 else workspace.scratch_K[:M]
        update_classification_output_(
            self.theta,
            self.log_theta,
            supervision.weighted_target,
            session.graph.source_affiliations(self),
            None,
            data.labelled,
            norm_scratch,
            workspace.matrix[:T, :K],
            norm_dim=norm_dim,
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
            session.graph.connection_into(self).delta,
            None,
            data.labelled,
            session.workspace.log_matrix[:, :M],
        )

    def partial_loss_(self, session: _FitSession) -> None:
        """Add this block's classification loss to the total loss."""
        data = session.data
        supervision = data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        workspace = session.workspace
        T = data.X_cont.shape[0]
        M, K = self.theta.shape
        classification_output_loss_(
            workspace.loss,
            self.log_theta,
            supervision.weighted_target,
            session.graph.source_affiliations(self),
            session.graph.connection_into(self).delta,
            None,
            data.labelled,
            workspace.matrix[:T, :M],
            workspace.log_matrix[:, :K],
            workspace.scalar,
        )
