"""The affiliation assignment step and entropy term shared by clustering blocks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from entlearn.primitives.normalise import _scratch_view
from entlearn.primitives.softmax import (
    argmin_assign_,
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

    A hard block takes the minimum-cost cluster. A soft block gives row ``t`` the softmax
    at ``epsilon · sample_weights[t]``, which is ``epsilon / T`` under uniform sample
    weights. A positive ``epsilon`` stays soft even where that product underflows.
    """
    # The head caches sums over its source's affiliations; mark them for recomputation
    session.graph.head.target_sums_fresh = False
    workspace = session.workspace
    if not block.soft_assignments:
        argmin_assign_(gamma, cost, 1, workspace.row_keepdim, workspace.row_indices)
        return
    weighted_assign_simplex_(
        gamma,
        cost,
        block.effective_epsilon,
        session.data.sample_weights if session.uniform_weight is None else session.uniform_weight,
        workspace.row_keepdim,
        workspace.row_indices,
    )


def affiliation_entropy_(
    gamma: torch.Tensor, block: _ClusteringBlock, session: _FitSession
) -> float:
    """Add the sample-weighted affiliation entropy at the block's assignment temperature.

    Returns the penalty's largest magnitude, ``epsilon log K``; zero for a hard block.
    """
    if not block.soft_assignments:
        return 0.0
    epsilon = block.effective_epsilon
    workspace = session.workspace
    log_buffer = _scratch_view(workspace.log_matrix, gamma.shape[1])
    if session.uniform_weight is not None:
        entropy_penalty_(
            workspace.loss,
            gamma,
            epsilon * session.uniform_weight,
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
    return epsilon * math.log(gamma.shape[1])
