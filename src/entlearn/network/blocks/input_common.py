"""Shared affiliations, instance weights, pruning and loss terms for fitted inputs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from entlearn.network.blocks.affiliations import affiliation_entropy_, assign_affiliations_
from entlearn.network.transfer import copy_tensor
from entlearn.primitives.normalise import _is_soft
from entlearn.primitives.reductions import compute_wt_cost_, reduce_input_cost_
from entlearn.primitives.softmax import assign_simplex_, compute_log_partition_
from entlearn.primitives.statistics import entropy_penalty_

if TYPE_CHECKING:
    from entlearn.network.blocks.types import _InputBlock
    from entlearn.network.session import _FitSession


def has_instance_weight_recovery(block: _InputBlock) -> bool:
    """Whether this fitted input learned weights and retained their finite normaliser."""
    epsilon = block.description.epsilon_T
    return (
        math.isfinite(epsilon)
        and _is_soft(epsilon, block.log_partition.dtype)
        and bool(torch.isfinite(block.log_partition))
    )


def copy_input_rows(
    block: _InputBlock,
    dtype: torch.dtype,
    device: torch.device,
    row_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy the input's row-bound state, or start it afresh for new rows.

    Returns the affiliations, the instance weights and the training log-normaliser.
    ``row_weights=None`` keeps the fitted rows. Otherwise the new rows take
    ``row_weights`` as their instance weights, and the normaliser is infinite until a
    fit writes it.
    """
    if row_weights is None:
        return (
            copy_tensor(block.gamma, dtype, device),
            copy_tensor(block.instance_weights, dtype, device),
            copy_tensor(block.log_partition, dtype, device),
        )
    return (
        torch.empty(len(row_weights), block.K, dtype=dtype, device=device),
        copy_tensor(row_weights, dtype, device),
        torch.full((), torch.inf, dtype=dtype, device=device),
    )


def recover_instance_weights_(
    block: _InputBlock, weights: torch.Tensor, cost: torch.Tensor
) -> None:
    """Recover query weights relative to the fitted log-normaliser.

    For query row ``t``, the unweighted discretisation cost is
    ``b[t] = sum_k gamma[t, k] * (sqdist[t, k] + categorical_cost[t, k])``,
    omitting absent modalities. The fitted instance-weight softmax retains
    ``log_Z_train = log(sum_s exp(-b_train[s] / epsilon_T))``. Recovery uses
    that same training normaliser rather than one recomputed on query rows::

        weights[t] = min(1, exp(-b[t] / epsilon_T - log_Z_train))

    These ratios are bounded by one, not normalised across the query batch.
    The caller must first check that the fit has instance-weight recovery.
    """
    torch.div(cost, -block.description.epsilon_T, out=weights)
    weights.sub_(block.log_partition).exp_().clamp_(0.0, 1.0)


def update_affiliations_(block: _InputBlock, session: _FitSession) -> None:
    """Set input affiliations from the assignment costs of the remaining clusters."""
    assign_affiliations_(block.gamma, block.live_cache.disc_cost, block, session)


def _compute_instance_weight_cost_(block: _InputBlock, session: _FitSession) -> None:
    """Price the fitted affiliations against the current unweighted distances."""
    cache = block.live_cache
    compute_wt_cost_(
        cache.instance_weight_cost,
        block.gamma,
        cache.sqdist,
        cache.categorical_cost,
        session.workspace.matrix[: session.data.X_cont.shape[0], : block.K],
    )


def _compute_log_partition_(block: _InputBlock, session: _FitSession) -> None:
    """Write the training normaliser from the (already computed) instance-weight cost."""
    compute_log_partition_(
        block.log_partition,
        block.live_cache.instance_weight_cost,
        block.description.epsilon_T,
        session.workspace.scratch_T,
        session.workspace.scalar,
    )


def update_instance_weights_(block: _InputBlock, session: _FitSession) -> None:
    """Update instance weights from the unweighted per-instance discretisation error."""
    epsilon_T = block.description.epsilon_T
    if not math.isfinite(epsilon_T):
        return
    _compute_instance_weight_cost_(block, session)
    cache = block.live_cache
    assign_simplex_(
        block.instance_weights,
        cache.instance_weight_cost,
        epsilon_T,
        0,
        cache.scratch_one,
        cache.scratch_index,
    )
    if _is_soft(epsilon_T, block.log_partition.dtype):
        _compute_log_partition_(block, session)


def refresh_log_partition_(block: _InputBlock, session: _FitSession) -> None:
    """Recalculate the log partition from the final geometry."""
    epsilon_T = block.description.epsilon_T
    if not math.isfinite(epsilon_T) or not _is_soft(epsilon_T, block.log_partition.dtype):
        block.log_partition.fill_(torch.inf)
        return
    _compute_instance_weight_cost_(block, session)
    _compute_log_partition_(block, session)


def prune_(block: _InputBlock, session: _FitSession) -> None:
    """Ask the graph to compact every block affected by the input's emptiness."""
    session.graph.prune_(block, session.workspace)


def discretisation_loss_(block: _InputBlock, session: _FitSession) -> None:
    """Add the discretisation error and sample-weighted affiliation entropy."""
    workspace = session.workspace
    T = session.data.X_cont.shape[0]
    reduce_input_cost_(
        workspace.loss,
        block.live_cache.disc_cost,
        block.gamma,
        workspace.matrix[:T, : block.K],
        workspace.scalar,
    )
    affiliation_entropy_(block.gamma, block, session)


def instance_weight_loss_(block: _InputBlock, session: _FitSession) -> None:
    """Add the instance-weight entropy when the input learns instance weights."""
    epsilon_T = block.description.epsilon_T
    if math.isfinite(epsilon_T) and _is_soft(epsilon_T, block.gamma.dtype):
        entropy_penalty_(
            session.workspace.loss,
            block.instance_weights,
            epsilon_T,
            session.workspace.scratch_T,
            session.workspace.scalar,
        )
