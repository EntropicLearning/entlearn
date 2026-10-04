"""Query-local coordinate refinement with every fitted parameter frozen."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from entlearn._warnings import _warn
from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input_common import recover_instance_weights_
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.blocks.types import _ClusteringBlock
from entlearn.network.config import PredictConfig
from entlearn.network.fit import LossIncreaseWarning, _converged, loss_noise_threshold
from entlearn.network.session import _CompiledGraph
from entlearn.primitives.coupling import accumulate_coupling_, transition_partial_loss_
from entlearn.primitives.normalise import _is_soft
from entlearn.primitives.output import (
    classification_output_accumulate_into_source_cost_,
    classification_output_loss_,
    regression_output_accumulate_into_source_cost_,
    regression_output_loss_,
)
from entlearn.primitives.softmax import argmin_assign_, softmax_with_temp_
from entlearn.primitives.statistics import entropy_penalty_

if TYPE_CHECKING:
    from entlearn.network.predict import _PredictionInputCache


@dataclass
class _Refinement:
    """Coordinates and scratch belonging to this query."""

    graph: _CompiledGraph
    query: _PredictionInputCache
    gammas: dict[str, torch.Tensor]
    prediction: torch.Tensor  # (T, M) latent output coordinate, standing in for query targets
    weights: torch.Tensor | None  # (T,) recovered ratios; None when recovery is unavailable
    input_scale: torch.Tensor  # (T,) T_train * weights / T; without recovery, row_weights itself
    row_weights: torch.Tensor  # (T,) uniform 1/T for coupling and head terms, not recovered
    labelled: torch.Tensor  # (T,) all true: every query has a latent output coordinate
    target_sq: torch.Tensor | None  # (T, M) squared regression coordinates, refreshed each update
    target_norm: torch.Tensor | None  # (T,) squared target norm, weighted by fitted W_M if present
    scratch_KM: torch.Tensor | None  # (K_source, M) residual scratch, disjoint from costs
    epsilon_P: float  # output temperature; arithmetic policies use the head's delta

    @classmethod
    def allocate(
        cls,
        graph: _CompiledGraph,
        query: _PredictionInputCache,
        gammas: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        weights: torch.Tensor | None,
        config: PredictConfig,
    ) -> _Refinement:
        """Allocate once before any coordinate update."""
        T, M = prediction.shape
        dtype, device = prediction.dtype, prediction.device
        regression = isinstance(graph.head, _RegressionBlock)
        row_weights = torch.full((T,), 1 / T, dtype=dtype, device=device)
        return cls(
            graph,
            query,
            gammas,
            prediction,
            weights,
            row_weights if weights is None else weights * (graph.training_rows / T),
            row_weights,
            torch.ones(T, dtype=torch.bool, device=device),
            torch.empty_like(prediction) if regression else None,
            torch.empty(T, dtype=dtype, device=device) if regression else None,
            torch.empty(gammas[graph.terminal.source].shape[1], M, dtype=dtype, device=device)
            if regression
            else None,
            graph.terminal.delta if config.epsilon_P is None else config.epsilon_P,
        )

    def refresh_target_(self) -> None:
        """Refresh every regression cache after setting the prediction coordinate."""
        head = self.graph.head
        if not isinstance(head, _RegressionBlock):
            return
        assert self.target_sq is not None and self.target_norm is not None
        torch.mul(self.prediction, self.prediction, out=self.target_sq)
        if head.W_M is None:
            torch.sum(self.target_sq, dim=1, out=self.target_norm)
        else:
            torch.mv(self.target_sq, head.W_M, out=self.target_norm)

    def prediction_step_(self) -> None:
        """Minimise the head term over the latent prediction coordinate."""
        head = self.graph.head
        gamma = self.gammas[self.graph.terminal.source]
        if isinstance(head, _ClassificationBlock):
            head.geometric_readout_(
                self.prediction,
                gamma,
                self.graph.terminal.delta,
                self.epsilon_P,
                self.query.workspace,
            )
        else:
            head.readout_(self.prediction, gamma, self.graph.terminal.delta, self.query.workspace)
        self.refresh_target_()

    def outgoing_cost_(self, name: str, cost: torch.Tensor) -> None:
        """Accumulate downstream costs against live query coordinates."""
        ws = self.query.workspace
        T = self.prediction.shape[0]
        for connection in self.graph.outgoing[name]:
            target = self.graph.blocks[connection.target]
            if isinstance(target, _HiddenBlock):
                accumulate_coupling_(
                    cost,
                    target.incoming[connection.name].log_theta,
                    self.gammas[connection.target],
                    connection.delta / T,
                )
            elif isinstance(target, _ClassificationBlock):
                classification_output_accumulate_into_source_cost_(
                    cost,
                    target.log_theta,
                    self.prediction,
                    connection.delta,
                    self.row_weights,
                    self.labelled,
                    ws.log_matrix[:, : target.output_width],
                )
            else:
                assert isinstance(target, _RegressionBlock)
                assert self.target_norm is not None and self.scratch_KM is not None
                K = cost.shape[1]
                regression_output_accumulate_into_source_cost_(
                    cost,
                    target.C_y,
                    self.prediction,
                    self.target_norm,
                    target.effective_delta(connection.delta),
                    self.row_weights,
                    self.labelled,
                    ws.log_matrix[:, :K],
                    ws.scratch_K[:K],
                    self.scratch_KM,
                    Wm=target.W_M,
                )

    def affiliation_step_(self, name: str) -> None:
        """Minimise one affiliation coordinate in its retained soft/hard assignment regime."""
        graph, ws = self.graph, self.query.workspace
        gamma = self.gammas[name]
        T, K = gamma.shape
        cost = ws.matrix[:T, :K]
        block = graph.blocks[name]
        assert isinstance(block, _ClusteringBlock)
        if isinstance(block, _HiddenBlock):
            cost.zero_()
            for connection in graph.incoming[name]:
                accumulate_coupling_(
                    cost,
                    block.incoming[connection.name].log_theta.transpose(0, 1),
                    self.gammas[connection.source],
                    connection.delta / T,
                )
        else:
            # Recovered weights scaled by T_train/T_query, or uniform 1/T without recovery.
            torch.mul(self.query.total, self.input_scale.unsqueeze(1), out=cost)
        self.outgoing_cost_(name, cost)
        # A large query batch can move epsilon / T below machine precision.
        # The fitted regime is referenced to avoid inadvertant regime change.
        if block.soft_assignments:
            softmax_with_temp_(gamma, cost, block.description.epsilon / T, 1, ws.row_keepdim)
        else:
            argmin_assign_(gamma, cost, 1, ws.row_indices)

    def input_step_(self) -> None:
        """Update input affiliations, then recover weights from both modalities."""
        self.affiliation_step_(self.graph.input.description.name)
        self.query.refresh_instance_cost_()
        if self.weights is not None:
            # These weights are influenced by the next input-affiliation update, so recovery
            # belongs inside each iteration.
            recover_instance_weights_(self.graph.input, self.weights, self.query.instance_cost)
            torch.mul(
                self.weights,
                self.graph.training_rows / self.prediction.shape[0],
                out=self.input_scale,
            )

    def loss(self) -> float:
        """Calculate the query loss.

        Let T be the query count, N the training count, a = N/T, and
        bₜ = Σₖ Γ⁰ₜₖ dₜₖ the input discretisation cost. With recovery active::

            L = a Σₜ [wₜ bₜ + ε_T wₜ(log wₜ + log Z_train - 1)]
                + (1/T) Σₙ ε̃ₙ Σₜₖ Γⁿₜₖ log Γⁿₜₖ
                - (1/T) Σₑ δₑ Σₜ Γ_target,tᵀ log Θₑ Γ_source,t
                + L_head

        The connection sum excludes the head. ε̃ₙ is the raw affiliation εₙ
        for a fitted soft regime and zero for a hard regime. Without instance-weight recovery,
        the first term becomes (1/T) Σₜ bₜ and there is no weight entropy.

        For classification with latent class distribution pₜ::

            L_head = -(δ_head/T) Σₜ pₜᵀ log Θ_head Γ_source,t
                     + (ε_P/T) Σₜₘ pₜₘ log pₜₘ

        The p entropy is omitted in the hard output regime. Arithmetic policies
        use ε_P = δ_head internally, so their final read-out is not priced here.
        For regression with latent target ŷₜ and normalised output weights vₘ::

            L_head = (δ_head/T) Σₜₖₘ Γ_source,tk vₘ (ŷₜₘ - C_y,mk)²

        Uniform output weighting means vₘ = 1/M.

        Each recovered wₜ is constrained independently to [0, 1], not to
        a simplex. Its partial loss term has stationary point
        exp(-bₜ/ε_T - log Z_train), clipped at one.
        """
        graph, ws = self.graph, self.query.workspace
        T, M = self.prediction.shape
        ws.loss.zero_()
        torch.dot(self.input_scale, self.query.instance_cost, out=ws.scalar)
        ws.loss.add_(ws.scalar)
        for name in graph.order[:-1]:
            block = graph.blocks[name]
            assert isinstance(block, _ClusteringBlock)
            if block.soft_assignments:
                gamma = self.gammas[name]
                entropy_penalty_(
                    ws.loss,
                    gamma,
                    block.description.epsilon / T,
                    ws.log_matrix[:, : gamma.shape[1]],
                    ws.scalar,
                )
        if self.weights is not None:
            scale = graph.training_rows / T * graph.input.description.epsilon_T
            entropy_penalty_(ws.loss, self.weights, scale, ws.scratch_T, ws.scalar)
            torch.sum(self.weights, dim=0, out=ws.scalar)
            ws.loss.add_(ws.scalar, alpha=scale * (float(graph.input.log_partition) - 1))
        head = graph.head
        gamma = self.gammas[graph.terminal.source]
        K = gamma.shape[1]
        if isinstance(head, _ClassificationBlock):
            classification_output_loss_(
                ws.loss,
                head.log_theta,
                self.prediction,
                gamma,
                graph.terminal.delta,
                self.row_weights,
                self.labelled,
                ws.log_matrix[:, :M],
                ws.matrix[:T, :K],
                ws.scalar,
            )
            if _is_soft(self.epsilon_P, self.prediction.dtype):
                entropy_penalty_(
                    ws.loss,
                    self.prediction,
                    self.epsilon_P / T,
                    ws.log_matrix[:, :M],
                    ws.scalar,
                )
        else:
            assert self.target_norm is not None and self.scratch_KM is not None
            regression_output_loss_(
                ws.loss,
                head.C_y,
                self.prediction,
                self.target_norm,
                gamma,
                head.effective_delta(graph.terminal.delta),
                self.row_weights,
                self.labelled,
                ws.matrix[:T, :K],
                ws.scratch_K[:K],
                self.scratch_KM,
                ws.scalar,
                Wm=head.W_M,
            )
        for connection in graph.connections:
            target = graph.blocks[connection.target]
            if isinstance(target, _HiddenBlock):
                source = self.gammas[connection.source]
                transition_partial_loss_(
                    ws.loss,
                    target.incoming[connection.name].log_theta,
                    self.gammas[connection.target],
                    source,
                    connection.delta / T,
                    ws.matrix[:T, : source.shape[1]],
                    ws.scalar,
                )
        result = float(ws.loss)
        if not math.isfinite(result):
            raise RuntimeError("iterative prediction produced a non-finite loss")
        return result


def refine_prediction_(
    graph: _CompiledGraph,
    query: _PredictionInputCache,
    gammas: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    weights: torch.Tensor | None,
    config: PredictConfig,
    *,
    seeded: bool,
) -> tuple[tuple[float, ...], bool]:
    """Refine coordinates in head-to-input order, recording complete iterations."""
    state = _Refinement.allocate(graph, query, gammas, prediction, weights, config)
    # A supplied regression start skips the first prediction update, not cache refresh.
    state.refresh_target_()
    history: list[float] = []
    converged = False
    for iteration in range(config.max_iter):
        if iteration or not seeded:
            state.prediction_step_()
        for name in reversed(graph.order[1:-1]):
            state.affiliation_step_(name)
        state.input_step_()
        history.append(state.loss())
        if len(history) > 1:
            rise = history[-1] - history[-2]
            if rise > loss_noise_threshold(history[-2], prediction.dtype):
                _warn(
                    f"prediction loss increased at iteration {iteration + 1} by {rise:.6g}",
                    LossIncreaseWarning,
                )
        if _converged(history, config.tol, prediction.dtype):
            converged = True
            break
    return tuple(history), converged
