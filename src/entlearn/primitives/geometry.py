"""Grouped tensor geometry and cluster-mass bookkeeping."""

from __future__ import annotations

import torch

from entlearn.primitives.normalise import _tiny


def grouped_means(
    index: torch.Tensor,
    weights: torch.Tensor,
    dense: torch.Tensor,
    cat_rows: list[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return mass-weighted row means for each group.

    A group with zero mass receives the global dense mean and uniform categorical rows.
    Mass at or below ``finfo(dtype).tiny`` counts as zero.

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
    masses = _grouped_sum(index, weights, k)
    dense_means = _grouped_sum(index, weights.unsqueeze(1) * dense, k)
    categorical_means = [_grouped_sum(index, weights.unsqueeze(1) * rows, k) for rows in cat_rows]
    _finalise_group_means_(dense_means, categorical_means, masses, weights, dense)
    return dense_means, categorical_means, masses


def _grouped_sum(index: torch.Tensor, values: torch.Tensor, k: int) -> torch.Tensor:
    """Return ``sums[g] = Σ_{t: index[t] = g} values[t]`` for the ``k`` groups, the same on every run."""
    if index.device.type == "cpu":  # where index_add_ repeats
        return values.new_zeros(k, *values.shape[1:]).index_add_(0, index, values)
    # On CUDA, index_add_'s atomic adds land in whatever order the threads finish, so its
    # rounding varies from run to run. A matrix product adds in a fixed order: row t of
    # the one-hot matrix is 1 in column index[t], so its transpose times values sums each
    # group's rows.
    one_hot = values.new_zeros(index.shape[0], k).scatter_(1, index.unsqueeze(1), 1.0)
    return one_hot.T @ values


def _finalise_group_means_(
    dense_means: torch.Tensor,
    categorical_means: list[torch.Tensor],
    masses: torch.Tensor,
    weights: torch.Tensor,
    dense: torch.Tensor,
) -> None:
    """Divide group sums by their masses in place. Empty groups get the global means."""
    tiny = _tiny(masses.dtype)
    empty = masses <= tiny
    any_empty = bool(empty.any())
    denominator = masses.clamp_min(tiny).unsqueeze(1)
    dense_means /= denominator
    if dense.shape[1] > 0 and any_empty:
        total = weights.sum().clamp_min(tiny)
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
    """Flag the clusters whose unweighted mass exceeds ``finfo(dtype).tiny``, and count them.

    Writes the unweighted mass and the survivor mask::

        cluster_mass[k] = Σ_t gamma[t, k]
        not_empty[k]    = cluster_mass[k] > finfo(gamma.dtype).tiny

    Args:
        not_empty: ``(K,)`` bool output, overwritten.
        cluster_mass: ``(K,)`` output, overwritten with the column sums of ``gamma``.
        gamma: ``(T, K)`` affiliations, read-only.

    Returns:
        int: How many clusters have mass above the threshold.
    """
    torch.sum(gamma, dim=0, out=cluster_mass)  # cluster_mass[k] = Σ_t gamma[t,k]
    torch.gt(cluster_mass, _tiny(gamma.dtype), out=not_empty)
    return int(not_empty.sum().item())  # count survivors
