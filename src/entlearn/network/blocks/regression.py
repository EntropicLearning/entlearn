"""Private regression-head parameters and tensor calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from entlearn.network.data import _RegressionSupervision, _StagedData
from entlearn.network.state import _RegressionParameters
from entlearn.network.transfer import copy_tensor
from entlearn.primitives.normalise import _is_soft
from entlearn.primitives.output import (
    propagate_regression_output_,
    regression_output_accumulate_into_source_cost_,
    regression_output_loss_,
    update_regression_output_,
)
from entlearn.primitives.output import (
    update_output_weights_ as update_output_weights_primitive,
)
from entlearn.primitives.statistics import entropy_penalty_
from entlearn.recipe import RegressionHead

if TYPE_CHECKING:
    from entlearn.network.session import _FitSession, _Workspace


@dataclass
class _RegressionCache:
    """Data-dependent caches and local scratch for one regression-head fit."""

    weighted_target_mean: torch.Tensor  # (M,), weighted mean across labelled rows
    scratch_KM: torch.Tensor  # (K_source, M), disjoint from live affiliation cost
    target_sq: torch.Tensor  # (T, M), elementwise Y²
    target_sq_norm: torch.Tensor  # (T,), unweighted sum across outputs
    effective_target_sq_norm: torch.Tensor | None  # (T,), matched to a concrete W_M
    output_weight_cost: torch.Tensor | None  # (M,), learnable W_M only
    output_scratch: torch.Tensor | None  # (M,), learnable W_M only
    output_keepdim: torch.Tensor | None  # (1,), learnable W_M only
    output_index: torch.Tensor | None  # (1,) int64, learnable W_M only

    @classmethod
    def allocate(
        cls,
        data: _StagedData,
        W_M: torch.Tensor | None,
        K_source: int,
        *,
        learn_output_weights: bool,
    ) -> _RegressionCache:
        """Allocate regression target statistics and scratch from the fitted dimensions."""
        M = data.target.shape[1]
        dtype = data.X_cont.dtype
        device = data.X_cont.device
        target_sq = data.target * data.target
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        return cls(
            weighted_target_mean=supervision.target_mean,
            scratch_KM=torch.empty(K_source, M, dtype=dtype, device=device),
            target_sq=target_sq,
            target_sq_norm=target_sq.sum(dim=1),
            effective_target_sq_norm=(torch.mv(target_sq, W_M) if W_M is not None else None),
            output_weight_cost=(
                torch.empty(M, dtype=dtype, device=device) if learn_output_weights else None
            ),
            output_scratch=(
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

    def stage_cache_(self, data: _StagedData, *, capacity: int | None = None) -> None:
        """Allocate this fit's target statistics and scratch, sized for ``capacity`` clusters.

        ``capacity=None`` sizes the scratch for the active clusters.
        """
        self.cache = _RegressionCache.allocate(
            data,
            self.W_M,
            self.C_y.shape[1] if capacity is None else capacity,
            learn_output_weights=math.isfinite(self.description.epsilon_M),
        )

    def release_cache(self) -> None:
        """Drop the fit cache, keeping the fitted parameters."""
        self.cache = None

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

    def refresh_effective_target_sq_(self) -> None:
        """Refresh the target norm cache after a learnable output-weight update."""
        if self.W_M is None:
            return
        cache = self.live_cache
        assert cache.effective_target_sq_norm is not None
        torch.mv(cache.target_sq, self.W_M, out=cache.effective_target_sq_norm)

    def _effective_target_sq_norm(self) -> torch.Tensor:
        """Return the target norm matched to this head's output-weight representation."""
        cache = self.live_cache
        if self.W_M is None:
            return cache.target_sq_norm
        assert cache.effective_target_sq_norm is not None
        return cache.effective_target_sq_norm

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
        """Add the weighted regression residual to the source assignment cost."""
        data = session.data
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        workspace = session.workspace
        K = self.C_y.shape[1]
        regression_output_accumulate_into_source_cost_(
            cost,
            self.C_y,
            data.target,
            self._effective_target_sq_norm(),
            self.effective_delta(session.graph.connection_into(self).delta),
            supervision.row_weights,
            data.labelled,
            workspace.log_matrix[:, :K],
            workspace.scratch_K[:K],
            self.live_cache.scratch_KM[:K],
            Wm=self.W_M,
        )

    def partial_loss_(self, session: _FitSession) -> None:
        """Add the regression residual and learnable-output entropy to the total loss."""
        data = session.data
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        workspace = session.workspace
        K = self.C_y.shape[1]
        regression_output_loss_(
            workspace.loss,
            self.C_y,
            data.target,
            self._effective_target_sq_norm(),
            session.graph.source_affiliations(self),
            self.effective_delta(session.graph.connection_into(self).delta),
            supervision.row_weights,
            data.labelled,
            workspace.log_matrix[:, :K],
            workspace.scratch_K[:K],
            self.live_cache.scratch_KM[:K],
            workspace.scalar,
            Wm=self.W_M,
        )
        if (
            self.W_M is not None
            and math.isfinite(self.description.epsilon_M)
            and _is_soft(self.description.epsilon_M, data.X_cont.dtype)
        ):
            entropy_penalty_(
                workspace.loss,
                self.W_M,
                self.description.epsilon_M,
                workspace.scratch_K[: self.output_width],
                workspace.scalar,
            )


def update_centroids_(block: _RegressionBlock, session: _FitSession) -> None:
    """Set the regression centroids from the labelled source affiliations."""
    data = session.data
    supervision = data.supervision
    assert isinstance(supervision, _RegressionSupervision)
    workspace = session.workspace
    M = data.target.shape[1]
    K = block.C_y.shape[1]
    cache = block.live_cache
    update_regression_output_(
        block.C_y,
        data.target,
        session.graph.source_affiliations(block),
        supervision.row_weights,
        data.labelled,
        cache.weighted_target_mean,
        workspace.scratch_K[:K],
        workspace.matrix[:M, :K],
        workspace.log_matrix[:, :K],
    )


def update_output_weights_(block: _RegressionBlock, session: _FitSession) -> None:
    """Set learnable output weights from residuals against the current centroids."""
    if block.W_M is None or not math.isfinite(block.description.epsilon_M):
        return
    data = session.data
    supervision = data.supervision
    assert isinstance(supervision, _RegressionSupervision)
    workspace = session.workspace
    cache = block.live_cache
    output_weight_cost = cache.output_weight_cost
    output_scratch = cache.output_scratch
    output_keepdim = cache.output_keepdim
    output_index = cache.output_index
    assert output_weight_cost is not None
    assert output_scratch is not None
    assert output_keepdim is not None
    assert output_index is not None
    M = data.target.shape[1]
    K = block.C_y.shape[1]
    update_output_weights_primitive(
        block.W_M,
        data.target,
        cache.target_sq,
        block.C_y,
        session.graph.source_affiliations(block),
        supervision.row_weights,
        data.labelled,
        session.graph.connection_into(block).delta,
        block.description.epsilon_M,
        output_weight_cost,
        workspace.log_matrix[:, :K],
        workspace.matrix[:M, :K],
        workspace.scratch_K[:K],
        workspace.scratch_T,
        output_keepdim,
        output_index,
        cache.scratch_KM[:K],
        output_scratch,
    )
    block.refresh_effective_target_sq_()
