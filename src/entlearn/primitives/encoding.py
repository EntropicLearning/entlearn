"""One-hot encoding helpers and the Wd modality split."""

from __future__ import annotations

import torch


def densify_categorical(
    X_cat: list[torch.Tensor], m_cat: list[int], dtype: torch.dtype
) -> list[torch.Tensor]:
    """Return every categorical column as a ``(T, M_d)`` distribution.

    A column of int64 codes is one-hot-encoded into fresh rows; a column that is
    already a distribution is passed through unchanged (not copied). The inputs are
    not mutated.

    Args:
        X_cat: Per feature, either a ``(T,)`` int64 tensor of codes or an
            already-dense ``(T, M_d)`` distribution (read-only).
        m_cat: The per-feature categorical cardinalities ``M_d``.
        dtype: Floating dtype for a freshly one-hot-encoded column.

    Returns:
        list[torch.Tensor]: Per feature, a ``(T, M_d)`` distribution.
    """
    dense: list[torch.Tensor] = []
    for i, m_d in enumerate(m_cat):
        col = X_cat[i]
        if col.dim() == 1:  # int64 codes -> fresh one-hot rows
            col = torch.nn.functional.one_hot(col, m_d).to(dtype)
        dense.append(col)
    return dense


def split_wd(wd: torch.Tensor, d_cont: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the feature weights ``wd`` into their continuous and categorical parts.

    The first ``d_cont`` entries weight the continuous features; the rest weight the
    categorical ones. Both returned tensors are views of ``wd``.

    Args:
        wd: ``(D_tot,)`` feature-weight vector.
        d_cont: number of continuous features (the boundary index).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the continuous and categorical views.
    """
    return wd[:d_cont], wd[d_cont:]
