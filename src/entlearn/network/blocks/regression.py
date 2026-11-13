"""Private regression-head parameters and tensor calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from entlearn.network.data import _RegressionSupervision, _StagedData
from entlearn.network.state import _RegressionParameters
from entlearn.network.transfer import copy_tensor
from entlearn.primitives.distance import weighted_sq_norm_
from entlearn.primitives.normalise import _is_soft, _scratch_view
from entlearn.primitives.output import (
    propagate_regression_output_,
    regression_output_accumulate_into_source_cost_,
    regression_output_loss_,
    update_regression_output_,
)
from entlearn.primitives.output import (
    update_output_weights_ as update_output_weights_primitive,
)
from entlearn.primitives.reductions import _chunked_product_
from entlearn.primitives.statistics import entropy_penalty_
from entlearn.recipe import RegressionHead

if TYPE_CHECKING:
    from entlearn.network.session import _FitSession, _Workspace


@dataclass
class _RegressionCache:
    """Data-dependent caches and local scratch for one regression-head fit."""

    weighted_target_mean: torch.Tensor  # (M,), weighted mean across labelled rows
    weighted_target: torch.Tensor  # (T, M + 1) row t is ω_t·Y_t, then ω_t; Γ's product: target_sums
    target_moment: torch.Tensor  # (M,) Σ_t ω_t·Y[t, m]², the W_M cost's and loss's row term
    target_sums: torch.Tensor  # (M + 1, K_source) weighted target sums, then cluster masses
    scratch_KM: torch.Tensor  # (K_source, M), disjoint from live affiliation cost
    output_weight_cost: torch.Tensor | None  # (M,), learnable W_M only
    output_keepdim: torch.Tensor | None  # (1,), learnable W_M only
    output_index: torch.Tensor | None  # (1,) int64, learnable W_M only

    @classmethod
    def allocate(
        cls,
        data: _StagedData,
        K_source: int,
        *,
        learn_output_weights: bool,
    ) -> _RegressionCache:
        """Allocate regression target statistics and scratch from the fitted dimensions."""
        M = data.target.shape[1]
        dtype = data.X_cont.dtype
        device = data.X_cont.device
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        # Row norms of Yᵀ weighted by the row weights, in row blocks, without forming Y²
        target_moment = torch.empty(M, dtype=dtype, device=device)
        weighted_sq_norm_(target_moment, data.target.transpose(0, 1), supervision.row_weights)
        row_weights = supervision.row_weights.unsqueeze(1)
        return cls(
            weighted_target_mean=supervision.target_mean,
            weighted_target=torch.cat((data.target * row_weights, row_weights), dim=1),
            target_moment=target_moment,
            target_sums=torch.empty(M + 1, K_source, dtype=dtype, device=device),
            scratch_KM=torch.empty(K_source, M, dtype=dtype, device=device),
            output_weight_cost=(
                torch.empty(M, dtype=dtype, device=device) if learn_output_weights else None
            ),
            output_keepdim=(
                torch.empty(1, dtype=dtype, device=device) if learn_output_weights else None
            ),
            output_index=(
                torch.empty(1, dtype=torch.int64, device=device) if learn_output_weights else None
            ),
        )


@dataclass
class _RegressionBlock:
    """Fitted centroids and optional output weights of a regression head.

    ``cache`` holds the target statistics and scratch of one fit: ``stage_cache_``
    allocates it when a fit session opens and ``release_cache`` drops it at publication.
    """

    description: RegressionHead
    C_y: torch.Tensor
    W_M: torch.Tensor | None
    cache: _RegressionCache | None = field(default=None, repr=False)
    target_sums_fresh: bool = field(default=False, repr=False)  # sums match the affiliations

    @property
    def output_width(self) -> int:
        """Return the regression output dimension."""
        return self.C_y.shape[0]

    @property
    def live_cache(self) -> _RegressionCache:
        """Return the fit-only cache."""
        if self.cache is None:
            raise RuntimeError(
                "the regression head holds no cache; a published Network must be re-staged "
                "before its tensor calculations can run again"
            )
        return self.cache

    @classmethod
    def make_block(
        cls,
        description: RegressionHead,
        data: _StagedData,
        K_source: int,
        parameters: _RegressionParameters | None = None,
    ) -> _RegressionBlock:
        """Construct supplied or weighted-mean-seeded centroids and configure output weights.

        The block holds no cache until ``stage_cache_``.
        """
        M = data.schema.M
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        if supervision.output_weights is not None:
            W_M = supervision.output_weights.detach().clone()
        elif parameters is not None and parameters.W_M is not None:
            W_M = parameters.W_M.detach().clone()
        elif math.isfinite(description.epsilon_M):
            W_M = torch.full(
                (M,),
                1.0 / M,
                dtype=data.X_cont.dtype,
                device=data.X_cont.device,
            )
        else:
            W_M = None
        if parameters is None:
            C_y = supervision.target_mean[:, None].expand(M, K_source).clone()
        else:
            C_y = parameters.C_y.detach().clone()
        return cls(description=description, C_y=C_y, W_M=W_M)

    def stage_cache_(self, data: _StagedData) -> None:
        """Allocate this fit's target statistics and scratch for the active clusters."""
        self.cache = _RegressionCache.allocate(
            data,
            self.C_y.shape[1],
            learn_output_weights=math.isfinite(self.description.epsilon_M),
        )
        self.target_sums_fresh = False

    def target_sums(self, session: _FitSession) -> torch.Tensor:
        """Return ``[ω ∘ Y, ω]ᵀ Γ``, formed once after each assignment.

        Its first ``M`` rows are the weighted target sums and its last row the cluster masses.
        """
        cache = self.live_cache
        sums = _scratch_view(cache.target_sums, self.C_y.shape[1])
        if not self.target_sums_fresh:
            gamma = session.graph.source_affiliations(self)
            _chunked_product_(sums, cache.weighted_target, gamma, session.workspace.matrix)
            self.target_sums_fresh = True
        return sums

    def release_cache(self) -> None:
        """Drop the fit cache, keeping the fitted parameters."""
        self.cache = None
        self.target_sums_fresh = False

    def copy_active(
        self, dtype: torch.dtype, device: torch.device, _row_weights: torch.Tensor | None
    ) -> _RegressionBlock:
        """Copy the centroids and output weights. The copy holds no cache.

        A head holds no row-bound state, so ``_row_weights`` is not used.
        """
        return replace(
            self,
            C_y=copy_tensor(self.C_y, dtype, device),
            W_M=None if self.W_M is None else copy_tensor(self.W_M, dtype, device),
            cache=None,
            target_sums_fresh=False,
        )

    def capture_parameters(self) -> _RegressionParameters:
        """Copy output centroids and weights without data-dependent caches."""
        return _RegressionParameters(
            self.description,
            self.C_y.detach().clone(),
            None if self.W_M is None else self.W_M.detach().clone(),
        )

    def effective_delta(self, delta: float) -> float:
        """Return the coupling strength matched to the output-weight representation."""
        return delta / self.output_width if self.W_M is None else delta

    def readout_(
        self,
        output: torch.Tensor,
        source_gamma: torch.Tensor,
        _delta: float,
        _workspace: _Workspace,
    ) -> None:
        """Write predictions from source affiliations."""
        propagate_regression_output_(output, source_gamma, self.C_y)

    def prune_source(self, keep: torch.Tensor, _scratch: torch.Tensor) -> None:
        """Compact the source axis after the source block has pruned."""
        index = keep.nonzero(as_tuple=True)[0]
        self.C_y = torch.index_select(self.C_y, 1, index)

    def update_parameters_(self, session: _FitSession) -> None:
        """Update ``C_y`` before any learnable ``W_M``."""
        update_centroids_(self, session)
        update_output_weights_(self, session)

    def accumulate_into_source_cost_(
        self,
        session: _FitSession,
        *,
        cost: torch.Tensor,
    ) -> None:
        """Add the weighted regression residual, less its row constant, to the source cost."""
        supervision = session.data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        cache = self.live_cache
        M, K = self.C_y.shape
        regression_output_accumulate_into_source_cost_(
            cost,
            self.C_y,
            cache.weighted_target[:, :M],
            self.effective_delta(session.graph.head_delta),
            supervision.row_weights,
            session.workspace.scratch_K[:K],
            cache.scratch_KM[:K],
            Wm=self.W_M,
        )

    @property
    def learns_output_weights(self) -> bool:
        """Whether the loss holds the output-weight entropy penalty."""
        epsilon_M = self.description.epsilon_M
        return self.W_M is not None and math.isfinite(epsilon_M) and _is_soft(epsilon_M)

    def partial_loss_(self, session: _FitSession) -> float:
        """Add the regression residual and learnable-output entropy to the total loss.

        Returns the entropy penalty's largest magnitude, ``epsilon_M log M``, or zero.
        """
        data = session.data
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        workspace = session.workspace
        K = self.C_y.shape[1]
        regression_output_loss_(
            workspace.loss,
            self.C_y,
            data.target,
            self.live_cache.target_moment,
            session.graph.source_affiliations(self),
            self.effective_delta(session.graph.head_delta),
            supervision.row_weights,
            _scratch_view(workspace.log_matrix, K),
            workspace.scratch_K[:K],
            self.live_cache.scratch_KM[:K],
            workspace.scalar,
            Wm=self.W_M,
        )
        if self.W_M is None or not self.learns_output_weights:
            return 0.0
        entropy_penalty_(
            workspace.loss,
            self.W_M,
            self.description.epsilon_M,
            self.live_cache.scratch_KM[0],  # an (M,) row, free once the residual is summed
            workspace.scalar,
        )
        return self.description.epsilon_M * math.log(self.output_width)


def update_centroids_(block: _RegressionBlock, session: _FitSession) -> None:
    """Set the regression centroids from the weighted target sums of the source."""
    M, K = block.C_y.shape
    sums = block.target_sums(session)
    update_regression_output_(
        block.C_y,
        sums[:M],
        sums[M],
        block.live_cache.weighted_target_mean,
        session.workspace.scratch_K[:K],
    )


def update_output_weights_(block: _RegressionBlock, session: _FitSession) -> None:
    """Set learnable output weights from residuals against the current centroids."""
    if block.W_M is None or not math.isfinite(block.description.epsilon_M):
        return
    cache = block.live_cache
    assert cache.output_weight_cost is not None
    assert cache.output_keepdim is not None
    assert cache.output_index is not None
    M, K = block.C_y.shape
    sums = block.target_sums(session)
    update_output_weights_primitive(
        block.W_M,
        block.C_y,
        sums[:M],
        sums[M],
        cache.target_moment,
        session.graph.head_delta,
        block.description.epsilon_M,
        cache.output_weight_cost,
        _scratch_view(cache.scratch_KM, K, M),
        cache.output_keepdim,
        cache.output_index,
    )
