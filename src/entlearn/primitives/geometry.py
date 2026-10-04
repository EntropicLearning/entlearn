"""Grouped tensor geometry and cluster-mass bookkeeping."""

from __future__ import annotations

import torch

from entlearn.primitives.normalise import _eps


def grouped_means(
    index: torch.Tensor,
    weights: torch.Tensor,
    dense: torch.Tensor,
    cat_rows: list[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return mass-weighted row means for each group.

    A group whose mass is at or below the machine precision of ``weights``'s dtype
    receives the global dense mean and uniform categorical rows.

    Args:
        index: ``(N,)``  Integer group index for each row.
        weights: ``(N,)``  Mass for each row.
        dense: ``(N, D)`` Continuous rows.
        cat_rows: ``(N, M_d)`` Categorical distribution rows for each feature.
        k: Number of groups.

    Returns:
        The ``(k, D)`` dense means, the per-feature ``(k, M_d)`` categorical means, and
        the ``(k,)`` group masses.
    """
    masses = torch.zeros(k, dtype=weights.dtype, device=weights.device)
    masses.index_add_(0, index, weights)
    dense_means = torch.zeros(k, dense.shape[1], dtype=dense.dtype, device=dense.device)
    dense_means.index_add_(0, index, weights.unsqueeze(1) * dense)
    categorical_means: list[torch.Tensor] = []
    for rows in cat_rows:
        means = torch.zeros(k, rows.shape[1], dtype=rows.dtype, device=rows.device)
        means.index_add_(0, index, weights.unsqueeze(1) * rows)
        categorical_means.append(means)
    _finalise_group_means_(dense_means, categorical_means, masses, weights, dense)
    return dense_means, categorical_means, masses


def _finalise_group_means_(
    dense_means: torch.Tensor,
    categorical_means: list[torch.Tensor],
    masses: torch.Tensor,
    weights: torch.Tensor,
    dense: torch.Tensor,
) -> None:
    """Divide group sums by their masses in place. Empty groups get the global means."""
    eps = _eps(masses.dtype)
    empty = masses <= eps
    any_empty = bool(empty.any())
    denominator = masses.clamp_min(eps).unsqueeze(1)
    dense_means /= denominator
    if dense.shape[1] > 0 and any_empty:
        total = weights.sum().clamp_min(eps)
        dense_means[empty] = (weights.unsqueeze(1) * dense).sum(0) / total
    for means in categorical_means:
        means /= denominator
        if means.shape[1] > 0 and any_empty:
            means[empty] = 1.0 / means.shape[1]


def mark_non_empty_clusters_(
    not_empty: torch.Tensor,
    cluster_mass: torch.Tensor,
    gamma: torch.Tensor,
) -> int:
    """Flag the clusters whose unweighted mass exceeds dtype machine precision, and count them.

    Writes the unweighted mass and the survivor mask::

        cluster_mass[k] = Σ_t gamma[t, k]
        not_empty[k]    = cluster_mass[k] > finfo(gamma.dtype).eps

    Args:
        not_empty: ``(K,)`` bool output, overwritten.
        cluster_mass: ``(K,)`` output, overwritten with the column sums of ``gamma``.
        gamma: ``(T, K)`` affiliations, read-only.

    Returns:
        int: How many clusters are above dtype machine precision.
    """
    torch.sum(gamma, dim=0, out=cluster_mass)  # cluster_mass[k] = Σ_t gamma[t,k]
    torch.gt(cluster_mass, _eps(gamma.dtype), out=not_empty)
    return int(not_empty.sum().item())  # count survivors
