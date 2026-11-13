"""Shared affiliations, instance weights, pruning and loss terms for fitted inputs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from entlearn.network.blocks.affiliations import affiliation_entropy_, assign_affiliations_
from entlearn.network.transfer import copy_tensor
from entlearn.primitives.normalise import _is_soft, _scratch_view
from entlearn.primitives.reductions import compute_wt_cost_, reduce_input_cost_
from entlearn.primitives.softmax import _scale_ratio_, assign_simplex_, compute_log_partition_
from entlearn.primitives.statistics import entropy_penalty_

if TYPE_CHECKING:
    from entlearn.network.blocks.types import _InputBlock
    from entlearn.network.session import _FitSession


def has_instance_weight_recovery(block: _InputBlock) -> bool:
    """Whether this fitted input learned weights and retained their finite normaliser."""
    return learns_instance_weights(block) and bool(torch.isfinite(block.log_partition))


def copy_input_rows(
    block: _InputBlock,
    dtype: torch.dtype,
    device: torch.device,
    row_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy the input's row-bound state, or start it afresh for new rows.

    Returns the affiliations, the instance weights and the two terms for later
    normalisation. Setting ``row_weights=None`` keeps the fitted rows. Otherwise the new rows
    take ``row_weights`` as their instance weights, and the normaliser is set to infinite, to
    be modified during training.
    """
    if row_weights is None:
        return (
            copy_tensor(block.gamma, dtype, device),
            copy_tensor(block.instance_weights, dtype, device),
            copy_tensor(block.log_partition, dtype, device),
            copy_tensor(block.partition_shift, dtype, device),
        )
    return (
        torch.empty(len(row_weights), block.K, dtype=dtype, device=device),
        copy_tensor(row_weights, dtype, device),
        torch.full((), torch.inf, dtype=dtype, device=device),
        torch.full((), torch.inf, dtype=dtype, device=device),
    )


def recover_instance_weights_(
    block: _InputBlock, weights: torch.Tensor, cost: torch.Tensor
) -> None:
    """Recover query weights relative to the fitted log-normaliser.

    For query row ``t`` the unweighted discretisation cost is
    ``b[t] = sum_k gamma[t, k] * (sqdist[t, k] + categorical_cost[t, k])``
    (omitting absent modalities). During training, two quantities are kept: the smallest
    shifted training cost ``m = min_s (b_train[s] - epsilon_T * log(T * w[s]))`` and
    ``log_partition = log_Z_train + m / epsilon_T``, where
    ``log_Z_train = log(sum_s T * w[s] * exp(-b_train[s] / epsilon_T))`` over the ``T``
    training rows and their normalised sample weights ``w``. Recovery uses the normaliser
    as::

        weights[t] = min(1, exp(-(b[t] - m) / epsilon_T - log_partition))

    The values are not normalised across the set of queries. The cost difference is taken
    before the temperature divides it, so the values hold at each positive ``epsilon_T``.
    The caller must first check that the fit has instance-weight recovery.
    """
    torch.sub(block.partition_shift, cost, out=weights)  # -(b - m)
    _scale_ratio_(weights, 1.0, block.description.epsilon_T)
    weights.sub_(block.log_partition).exp_().clamp_(max=1.0)


def update_affiliations_(block: _InputBlock, session: _FitSession, *, coupled: bool = True) -> None:
    """Set input affiliations from the ``Wt``-weighted discretisation error and the couplings.

    ``coupled=False`` assigns the starting affiliations from the discretisation error alone.
    Downstream blocks do not yet include affiliations for these rows, so their coupling term
    is not included.
    """
    cost = _scratch_view(session.workspace.matrix, block.K, session.data.X_cont.shape[0])
    torch.mul(block.live_cache.disc_cost, block.instance_weights.unsqueeze(1), out=cost)
    if coupled:
        session.graph.accumulate_outgoing_cost_(block, cost, session)
    assign_affiliations_(block.gamma, cost, block, session)


def _compute_instance_weight_cost_(block: _InputBlock, session: _FitSession) -> None:
    """Price each row's error ``b``, minus ``epsilon_T·log(T·w)`` for soft, non-uniform ``w``."""
    cache = block.live_cache
    compute_wt_cost_(
        cache.instance_weight_cost,
        block.gamma,
        cache.disc_cost,
        None,
        _scratch_view(session.workspace.matrix, block.K, session.data.X_cont.shape[0]),
    )
    epsilon_T = block.description.epsilon_T
    log_relative = session.log_relative_weights
    if log_relative is not None and _is_soft(epsilon_T):
        cache.instance_weight_cost.sub_(log_relative, alpha=epsilon_T)


def _compute_log_partition_(block: _InputBlock, session: _FitSession) -> None:
    """Write the training normaliser from the (already computed) instance-weight cost."""
    compute_log_partition_(
        block.log_partition,
        block.partition_shift,
        block.live_cache.instance_weight_cost,
        block.description.epsilon_T,
        session.workspace.scratch_T,
    )


def update_instance_weights_(block: _InputBlock, session: _FitSession) -> None:
    """Update instance weights as ``W_T ∝ w·exp(-b/epsilon_T)``."""
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


def refresh_log_partition_(block: _InputBlock, session: _FitSession) -> None:
    """Recalculate the instance-weight normaliser from the final geometry."""
    if not learns_instance_weights(block):
        block.log_partition.fill_(torch.inf)
        block.partition_shift.fill_(torch.inf)
        return
    _compute_instance_weight_cost_(block, session)
    _compute_log_partition_(block, session)


def prune_(block: _InputBlock, session: _FitSession) -> None:
    """Ask the graph to compact every block affected by the input's emptiness."""
    session.graph.prune_(block, session.workspace)


def discretisation_loss_(block: _InputBlock, session: _FitSession) -> float:
    """Add the ``Wt``-weighted discretisation error and sample-weighted affiliation entropy.

    Returns the entropy penalty's largest magnitude, ``epsilon log K``; zero for a hard block.
    """
    workspace = session.workspace
    weighted = _scratch_view(workspace.matrix, block.K, session.data.X_cont.shape[0])
    torch.mul(block.live_cache.disc_cost, block.instance_weights.unsqueeze(1), out=weighted)
    reduce_input_cost_(workspace.loss, weighted, block.gamma, weighted, workspace.scalar)
    return affiliation_entropy_(block.gamma, block, session)


def learns_instance_weights(block: _InputBlock) -> bool:
    """Return whether the loss holds the instance-weight penalty."""
    epsilon_T = block.description.epsilon_T
    return math.isfinite(epsilon_T) and _is_soft(epsilon_T)


def instance_weight_loss_(block: _InputBlock, session: _FitSession) -> float:
    """Add ``-epsilon_T·[H(W_T) + Σ_t W_T[t]·log(T·w[t])]`` when the input block learns ``W_T``.

    Returns the penalty's largest magnitude, ``epsilon_T log T``, or zero.
    """
    epsilon_T = block.description.epsilon_T
    if not learns_instance_weights(block):
        return 0.0
    entropy_penalty_(
        session.workspace.loss,
        block.instance_weights,
        epsilon_T,
        session.workspace.scratch_T,
        session.workspace.scalar,
    )
    if session.log_relative_weights is not None:
        scalar = session.workspace.scalar
        torch.dot(block.instance_weights, session.log_relative_weights, out=scalar)
        session.workspace.loss.sub_(scalar, alpha=epsilon_T)
    return epsilon_T * math.log(block.instance_weights.numel())
