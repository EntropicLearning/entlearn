"""Weighted squared-distance primitives for the input-block discretisation error."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from entlearn.primitives.normalise import _tiny


def weighted_sq_distance_(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    A_sq_sum: torch.Tensor | None,
    scratch_K: torch.Tensor,
    scratch_KD: torch.Tensor,
    feature_weights: torch.Tensor | None = None,
) -> None:
    """Write the feature-weighted squared distances into ``out``.

    Computes ``out[t, k] = Σ_d w[d]·(A[t, d] - B[k, d])²``, expanded as
    ``‖a‖²_w - 2·⟨a, b⟩_w + ‖b‖²_w``. ``feature_weights=None`` means every weight is 1.

    Operands should be appropriately scaled; continuous features are expected in
    or near ``[0, 1]``. Large offsets and nearly coincident points can cause
    cancellation and loss of relative accuracy in this expansion.

    Args:
        out: ``(T, K)`` output, overwritten.
        A: ``(T, D)`` left operand, e.g. data ``X`` or targets ``Y``.
        B: ``(K, D)`` right operand, e.g. centroids ``C`` or ``Cyᵀ``.
        A_sq_sum: ``(T,)`` precomputed ``Σ_d w[d]·A[t, d]²``, weighted by the same
            ``feature_weights``, or ``None`` to leave out this per-row term.
        scratch_K: ``(K,)`` workspace for ``‖b_k‖²_w``.
        scratch_KD: ``(K, D)`` workspace.
        feature_weights: ``(D,)`` per-feature weights, or ``None``.
    """
    if feature_weights is not None:
        torch.mul(B, feature_weights, out=scratch_KD)  # B ∘ w
        torch.mm(A, scratch_KD.transpose(0, 1), out=out)  # ⟨A, B⟩_w
        scratch_KD.mul_(B)  # w · B²
    else:
        torch.mm(A, B.transpose(0, 1), out=out)  # ⟨A, B⟩
        torch.mul(B, B, out=scratch_KD)  # B²
    torch.sum(scratch_KD, dim=1, out=scratch_K)  # ‖B‖²_w  (Σ_d w·B²)
    out.mul_(-2.0)
    if A_sq_sum is not None:
        out.add_(A_sq_sum.unsqueeze(1))  # + ‖a‖²_w  (broadcast over K)
    out.add_(scratch_K.unsqueeze(0))  # + ‖b‖²_w  (broadcast over T)


def assemble_categorical_cost_(
    cost: torch.Tensor,
    xent_scratch: torch.Tensor,
    X_cat_i: torch.Tensor,
    logC_cat_i: torch.Tensor,
    weight_scale: float | torch.Tensor,
    scratch_MK: torch.Tensor | None,
) -> None:
    """Add one categorical feature's cross-entropy error to ``cost``.

    Computes ``xent[t, k] = -Σ_m X̃[t, m]·logC[k, m]``, or the gather
    ``-logC[k, code_t]`` when ``X_cat_i`` holds integer codes, then adds
    ``weight_scale·xent[t, k]`` to ``cost[t, k]``.

    Args:
        cost: ``(T, K)`` cost, accumulated into.
        xent_scratch: ``(T, K)`` workspace, left holding ``weight_scale·xent``.
        X_cat_i: ``(T,)`` integer codes or ``(T, M_d)`` per-instance distribution.
        logC_cat_i: ``(K, M_d)`` log of this feature's categorical centroids.
        weight_scale: The scalar ``δ_cat · s_i · Wd[D_cont + i]`` for this feature.
            A zero-dimensional tensor keeps a device-resident scale on the device.
        scratch_MK: ``(M_d, K)`` workspace with unit-stride rows, required only for integer
            codes, which gather their rows from ``-weight_scale·logCᵀ`` written here.
    """
    if X_cat_i.dim() == 1:  # integer codes: gather row code_t of -ws·logCᵀ
        assert scratch_MK is not None  # required for integer codes
        torch.mul(logC_cat_i.transpose(0, 1), weight_scale, out=scratch_MK).neg_()
        torch.index_select(scratch_MK, 0, X_cat_i, out=xent_scratch)
    else:  # (T, M_d) distribution
        torch.mm(X_cat_i, logC_cat_i.transpose(0, 1), out=xent_scratch)
        xent_scratch.neg_().mul_(weight_scale)  # ws · cross-entropy
    cost.add_(xent_scratch)


# Row-block budget for the block-wise squared-norm reduction (used in prediction): avoids materialising a full (T, D) temporary.
_NORM_BLOCK_BYTES = 4 * 1024 * 1024


def weighted_sq_norm_(out: torch.Tensor, X: torch.Tensor, Wd: torch.Tensor | None = None) -> None:
    """Write ``out[t] = Σ_d Wd[d]·X[t, d]²`` without materialising ``X²``.

    The same quantity as :func:`refresh_weighted_norm_`, reduced straight from ``X`` a
    row block at a time. Use this when the ``X²`` cache is not already available.

    Args:
        out: ``(T,)`` output, overwritten.
        X: ``(T, D)`` continuous data.
        Wd: ``(D,)`` continuous feature weights, or ``None`` for the unweighted ``‖x_t‖²``.
    """
    n_rows, width = X.shape
    if n_rows == 0 or width == 0:
        out.zero_()
        return
    step = max(1, _NORM_BLOCK_BYTES // (width * X.element_size()))
    for start in range(0, n_rows, step):
        stop = min(start + step, n_rows)
        block = X[start:stop]
        if Wd is None:
            torch.sum(block * block, dim=1, out=out[start:stop])
        else:
            torch.mv(block * block, Wd, out=out[start:stop])


def refresh_weighted_norm_(
    X_sq_wd_sum: torch.Tensor,
    X_sq: torch.Tensor,
    Wd: torch.Tensor,
) -> None:
    """Write the Wd-weighted squared norm ``X_sq_wd_sum ← X_sq @ Wd``.

    Computes ``X_sq_wd_sum[t] = Σ_d Wd[d]·X_sq[t, d]``, the ``‖x_t‖²_w`` term of the
    distance expansion. Rerun it after every ``Wd`` update.

    Args:
        X_sq_wd_sum: ``(T,)`` output, overwritten.
        X_sq: ``(T, D)`` element-wise ``X²``, independent of ``Wd``.
        Wd: ``(D,)`` continuous feature weights.
    """
    torch.mv(X_sq, Wd, out=X_sq_wd_sum)


def dissimilarity_to_rows(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function giving every row's feature-weighted dissimilarity to the rows ``seeds``.

    Column ``j`` holds ``D²(t, s_j) = Σ_d w_d·(x_td - x_sd)² + Σ_i w_i·(-Σ_m p_tim·log p_sim)``,
    with ``s = seeds[j]`` and each log floored at ``finfo(dtype).tiny``, as during fitting.
    The expansion
    ``‖x‖²_w - 2x·(w∘x_s) + ‖x_s‖²_w`` reads ``X`` once for all seeds, and integer codes are
    compared as codes, giving their one-hot rows' values.

    Args:
        X: ``(T, D_cont)`` continuous rows.
        X_cat: Per-feature ``(T, M_i)`` categorical distributions or ``(T,)`` integer codes.
        Wd_cont: ``(D_cont,)`` continuous feature weights.
        Wd_cat: ``(D_cat,)`` categorical feature weights.

    Returns:
        A function of ``(n,)`` int64 row indices returning the new ``(T, n)`` dissimilarities.
    """
    tiny = _tiny(X.dtype)
    logX_cat = [rows if rows.dim() == 1 else rows.clamp_min(tiny).log() for rows in X_cat]
    X_norm = X.new_empty(X.shape[0])
    weighted_sq_norm_(X_norm, X, Wd_cont)

    def dissimilarity(seeds: torch.Tensor) -> torch.Tensor:
        n = seeds.numel()
        d2 = X.new_empty(X.shape[0], n)
        scratch_n, scratch_nD = X.new_empty(n), X.new_empty(n, X.shape[1])
        weighted_sq_distance_(d2, X, X[seeds], X_norm, scratch_n, scratch_nD, Wd_cont)
        d2.clamp_min_(0.0)  # the expansion can round below zero
        for feature, (rows, logs) in enumerate(zip(X_cat, logX_cat, strict=True)):
            if rows.dim() == 1:  # cross-entropy to a one-hot seed floored at tiny
                cross_entropy = (rows.unsqueeze(1) != rows[seeds]).to(X.dtype)
                cross_entropy.mul_(-math.log(tiny))
            else:
                cross_entropy = (rows @ logs[seeds].T).neg_()
            d2 += Wd_cat[feature] * cross_entropy
        return d2

    return dissimilarity
