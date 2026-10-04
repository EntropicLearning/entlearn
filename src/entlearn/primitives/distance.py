"""Weighted squared-distance primitives for the input-block discretisation error."""

from __future__ import annotations

import torch


def weighted_sq_distance_(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    A_sq_sum: torch.Tensor,
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
            ``feature_weights``.
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
    out.add_(A_sq_sum.unsqueeze(1))  # + ‖a‖²_w  (broadcast over K)
    out.add_(scratch_K.unsqueeze(0))  # + ‖b‖²_w  (broadcast over T)


def assemble_euclidean_cost_(
    cost: torch.Tensor,
    sqdist_wd: torch.Tensor,
    X: torch.Tensor,
    C: torch.Tensor,
    Wd: torch.Tensor,
    Wt: torch.Tensor,
    X_sq_wd_sum: torch.Tensor,
    scratch_KD: torch.Tensor,
    scratch_K: torch.Tensor,
) -> None:
    """Refresh the distance cache ``sqdist_wd`` and add its weighted form to ``cost``.

    Writes ``sqdist_wd[t, k] = Σ_d Wd[d]·(X[t, d] - C[k, d])²``, then adds
    ``Wt[t]·sqdist_wd[t, k]`` to ``cost[t, k]``. The caller must zero ``cost`` first.

    Args:
        cost: ``(T, K)`` discretisation error, accumulated into.
        sqdist_wd: ``(T, K)`` distance cache, overwritten.
        X: ``(T, D)`` continuous data.
        C: ``(K, D)`` continuous centroids.
        Wd: ``(D,)`` continuous feature weights.
        Wt: ``(T,)`` per-instance weights.
        X_sq_wd_sum: ``(T,)`` precomputed ``Σ_d Wd[d]·X[t, d]²``.
        scratch_KD: ``(K, D)`` workspace.
        scratch_K: ``(K,)`` workspace.
    """
    weighted_sq_distance_(sqdist_wd, X, C, X_sq_wd_sum, scratch_K, scratch_KD, feature_weights=Wd)
    cost.addcmul_(sqdist_wd, Wt.unsqueeze(1))  # cost += Wt[:, None] · sqdist_wd


def assemble_categorical_cost_(
    disc_cost: torch.Tensor | None,
    cat_cost: torch.Tensor,
    xent_scratch: torch.Tensor,
    X_cat_i: torch.Tensor,
    logC_cat_i: torch.Tensor,
    weight_scale: float | torch.Tensor,
    Wt: torch.Tensor | None,
) -> None:
    """Add one categorical feature's cross-entropy error to the two cost caches.

    Computes ``xent[t, k] = -Σ_m X̃[t, m]·logC[k, m]``, or the gather
    ``-logC[k, code_t]`` when ``X_cat_i`` holds integer codes, then adds::

        cat_cost[t, k]  += weight_scale · xent[t, k]
        disc_cost[t, k] += weight_scale · Wt[t] · xent[t, k]

    The caller must zero both caches first.

    Args:
        disc_cost: ``(T, K)`` Wt-weighted discretisation error, accumulated into.
            ``None`` skips it and leaves only ``cat_cost``.
        cat_cost: ``(T, K)`` unweighted cross-entropy summed across features,
            accumulated into.
        xent_scratch: ``(T, K)`` workspace for this feature's cross-entropy.
        X_cat_i: ``(T,)`` integer codes or ``(T, M_d)`` per-instance distribution.
        logC_cat_i: ``(K, M_d)`` log of this feature's categorical centroids.
        weight_scale: The scalar ``δ_cat · s_i · Wd[D_cont + i]`` for this feature.
            A zero-dimensional tensor keeps a device-resident scale on the device.
        Wt: ``(T,)`` per-instance weights. Required only when ``disc_cost`` is given.
    """
    if X_cat_i.dim() == 1:  # integer codes: gather column code_t of logC
        torch.index_select(logC_cat_i.transpose(0, 1), 0, X_cat_i, out=xent_scratch)
    else:  # (T, M_d) distribution
        torch.mm(X_cat_i, logC_cat_i.transpose(0, 1), out=xent_scratch)
    xent_scratch.neg_().mul_(weight_scale)  # ws · cross-entropy
    cat_cost.add_(xent_scratch)  # cat_cost += ws · xent
    if disc_cost is not None:
        assert Wt is not None  # required alongside disc_cost
        disc_cost.addcmul_(xent_scratch, Wt.unsqueeze(1))  # += ws·Wt·xent


# Row-block budget for the block-wise squared-norm reduction (used in prediction): avoids materialising a full (T, D) temporary.
_NORM_BLOCK_BYTES = 4 * 1024 * 1024


def weighted_sq_norm_(out: torch.Tensor, X: torch.Tensor, Wd: torch.Tensor) -> None:
    """Write ``out[t] = Σ_d Wd[d]·X[t, d]²`` without materialising ``X²``.

    The same quantity as :func:`refresh_weighted_norm_`, reduced straight from ``X`` a
    row block at a time. Use this when the ``X²`` cache is not already available.

    Args:
        out: ``(T,)`` output, overwritten.
        X: ``(T, D)`` continuous data.
        Wd: ``(D,)`` continuous feature weights.
    """
    n_rows, width = X.shape
    if n_rows == 0 or width == 0:
        out.zero_()
        return
    step = max(1, _NORM_BLOCK_BYTES // (width * X.element_size()))
    for start in range(0, n_rows, step):
        stop = min(start + step, n_rows)
        block = X[start:stop]
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
