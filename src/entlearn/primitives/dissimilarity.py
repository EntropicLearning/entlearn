"""Feature-weighted dissimilarity to continuous and categorical points."""

from __future__ import annotations

import math

import torch

_D2_BLOCK_BYTES = 4 * 1024 * 1024
_MIN_D2_BLOCKS = 4


def _d2_row_blocks(n_rows: int, width: int, itemsize: int) -> list[tuple[int, int]]:
    """Split rows into near-equal blocks when the pass warrants splitting into blocks."""
    if n_rows <= 0:
        return []
    n_blocks = math.ceil(n_rows * max(width, 1) * itemsize / _D2_BLOCK_BYTES)
    if n_blocks < _MIN_D2_BLOCKS:
        return [(0, n_rows)]
    step = math.ceil(n_rows / n_blocks)
    return [(start, min(start + step, n_rows)) for start in range(0, n_rows, step)]


def feature_d2_to_point(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    log_p_cat: list[torch.Tensor],
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    p_c: torch.Tensor,
) -> torch.Tensor:
    """Return each row's feature-weighted dissimilarity to one point.

    Sums a Wd-weighted squared-Euclidean term over the continuous block and a
    Wd-weighted cross-entropy term over each categorical block::

        D^2(t) = sum_d Wd_cont[d]*(X[t,d] - p_c[d])^2
               + sum_i Wd_cat[i]*(-sum_m X_cat_i[t,m]*log_p_cat_i[m])

    Args:
        X: Continuous data with instances in rows.
        X_cat: Categorical distributions with instances in rows.
        log_p_cat: Floored log distributions of the point.
        Wd_cont: Continuous feature weights.
        Wd_cat: Categorical feature weights.
        p_c: Continuous coordinates of the point.

    Returns:
        Dissimilarity of each instance to the point.
    """
    n_rows, d_cont = X.shape
    width = d_cont + sum(rows.shape[1] for rows in X_cat)
    d2 = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    blocks = _d2_row_blocks(n_rows, width, X.element_size())
    max_rows = max((stop - start for start, stop in blocks), default=0)
    difference = torch.empty(max_rows, d_cont, dtype=X.dtype, device=X.device)
    weighted = torch.empty(max_rows, d_cont, dtype=X.dtype, device=X.device)
    for start, stop in blocks:
        rows = stop - start
        difference_block = difference[:rows]
        weighted_block = weighted[:rows]
        block = d2[start:stop]
        torch.sub(X[start:stop], p_c, out=difference_block)
        torch.mul(Wd_cont, difference_block, out=weighted_block)
        weighted_block.mul_(difference_block)
        torch.sum(weighted_block, dim=1, out=block)
        for feature, (categorical, point_log) in enumerate(zip(X_cat, log_p_cat, strict=True)):
            cross_entropy = -(categorical[start:stop] * point_log).sum(dim=1)
            block += Wd_cat[feature] * cross_entropy
    return d2


def seed_dissimilarity(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    logX_cat: list[torch.Tensor],
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Return each row's dissimilarity to a selected data row."""
    return feature_d2_to_point(
        X,
        X_cat,
        [categorical_log[seed] for categorical_log in logX_cat],
        Wd_cont,
        Wd_cat,
        X[seed],
    )
