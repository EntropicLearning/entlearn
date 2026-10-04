"""The affiliation assignment step and entropy term shared by clustering blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from entlearn.primitives.softmax import (
    argmin_assign_,
    softmax_with_temp_,
    weighted_assign_simplex_,
)
from entlearn.primitives.statistics import entropy_penalty_

if TYPE_CHECKING:
    from entlearn.network.blocks.types import _ClusteringBlock
    from entlearn.network.session import _FitSession


def assign_affiliations_(
    gamma: torch.Tensor,
    cost: torch.Tensor,
    block: _ClusteringBlock,
    session: _FitSession,
) -> None:
    """Assign each row to the clusters from its ``(T, K)`` assignment cost.

    A hard block takes the minimum-cost cluster. A soft block takes the softmax at
    ``epsilon / T`` under uniform sample weights; otherwise row ``t`` takes it at
    ``epsilon · sample_weights[t]``. The block's resolved regime picks the route, so a
    per-row temperature below machine precision still takes the softmax.
    """
    workspace = session.workspace
    if not block.soft_assignments:
        argmin_assign_(gamma, cost, 1, workspace.row_indices)
    elif session.uniform_sample_weights:
        tau = block.description.epsilon / session.data.X_cont.shape[0]
        softmax_with_temp_(gamma, cost, tau, 1, workspace.row_keepdim)
    else:
        weighted_assign_simplex_(
            gamma,
            cost,
            block.description.epsilon,
            session.data.sample_weights,
            workspace.row_keepdim,
            workspace.row_indices,
        )


def affiliation_entropy_(
    gamma: torch.Tensor, block: _ClusteringBlock, session: _FitSession
) -> None:
    """Add the sample-weighted affiliation entropy at the block's assignment temperature."""
    if not block.soft_assignments:
        return
    epsilon = block.description.epsilon
    workspace = session.workspace
    log_buffer = workspace.log_matrix[:, : gamma.shape[1]]
    if session.uniform_sample_weights:
        entropy_penalty_(
            workspace.loss,
            gamma,
            epsilon / session.data.X_cont.shape[0],
            log_buffer,
            workspace.scalar,
        )
    else:
        entropy_penalty_(
            workspace.loss,
            gamma,
            epsilon,
            log_buffer,
            workspace.scalar,
            row_weights=session.data.sample_weights,
        )
