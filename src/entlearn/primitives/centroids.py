"""Input-block update step writers: feature-weight cost + centroid updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from entlearn.primitives.normalise import _scratch_view, _tiny

# Row batches of a one-hot product: enough independent products to occupy a GPU
_ONE_HOT_BATCHES = 256


def compute_wd_cost_euclidean_(
    Wd_cost_slice: torch.Tensor,
    X: torch.Tensor,
    C: torch.Tensor,
    gamma_wt: torch.Tensor,
    denom: torch.Tensor,
    X_sq: torch.Tensor,
    centroid_numerator: torch.Tensor,
    scratch_T: torch.Tensor,
    scratch_KD: torch.Tensor,
    scratch_D: torch.Tensor,
    *,
    need_cost: bool = True,
) -> None:
    """Write the per-feature Euclidean cost ``b_d`` into ``Wd_cost_slice``.

    Computes ``b_d[d] = Σ_t Σ_k gamma_wt[t,k]·(X[t,d] - C[k,d])²``, expanded as::

        b_d[d] = Σ_t mass[t]·X_sq[t,d]      (mass[t] = Σ_k gamma_wt[t,k])
               - 2·Σ_k C[k,d]·numer[k,d]    (numer = gamma_wtᵀ @ X)
               + Σ_k denom[k]·C[k,d]²

    Also leaves ``centroid_numerator = gamma_wtᵀ @ X`` populated, which
    :func:`update_euclidean_centroids_` then consumes. ``centroid_numerator`` and
    ``scratch_KD`` must be distinct buffers. ``C`` is not modified.

    Args:
        Wd_cost_slice: ``(D,)`` output, overwritten with ``b_d``.
        X: ``(T, D)`` continuous data.
        C: ``(K, D)`` current continuous centroids, read-only.
        gamma_wt: ``(T, K)`` the ``Γ ∘ Wt`` product.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t,k]``.
        X_sq: ``(T, D)`` element-wise ``X²``.
        centroid_numerator: ``(K, D)`` output, overwritten with ``gamma_wtᵀ @ X``.
        scratch_T: ``(T,)`` workspace.
        scratch_KD: ``(K, D)`` workspace.
        scratch_D: ``(D,)`` workspace.
        need_cost: ``False`` computes only ``centroid_numerator`` and leaves
            ``Wd_cost_slice`` untouched. Pass it when ``Wd`` is frozen.
    """
    torch.mm(gamma_wt.transpose(0, 1), X, out=centroid_numerator)  # numer = gamma_wtᵀ @ X
    if not need_cost:
        return  # Wd frozen: the numerator above is the only output the caller consumes
    # Term 1: Σ_t mass[t]·X_sq[t,d], with mass[t] = Σ_k gamma_wt[t,k]
    torch.sum(gamma_wt, dim=1, out=scratch_T)
    torch.mv(X_sq.transpose(0, 1), scratch_T, out=Wd_cost_slice)
    # Term 2: - 2·Σ_k C[k,d]·numer[k,d]
    torch.mul(C, centroid_numerator, out=scratch_KD)
    torch.sum(scratch_KD, dim=0, out=scratch_D)
    Wd_cost_slice.sub_(scratch_D, alpha=2.0)
    # Term 3: + Σ_k denom[k]·C[k,d]²
    torch.mul(C, C, out=scratch_KD)
    scratch_KD.mul_(denom.unsqueeze(1))
    torch.sum(scratch_KD, dim=0, out=scratch_D)
    Wd_cost_slice.add_(scratch_D)


def compute_wd_cost_categorical_(
    Wd_cost_slice_cat: torch.Tensor,
    X_cat: Sequence[torch.Tensor],
    gamma_wt: torch.Tensor,
    logC_cat_list: Sequence[torch.Tensor],
    combined_scales: torch.Tensor,
    cat_numerator_list: Sequence[torch.Tensor],
    scratch_KM: torch.Tensor,
    *,
    need_cost: bool = True,
    code_scratch: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Write each categorical feature's discretisation error into ``Wd_cost_slice_cat``.

    Per feature ``i``, forms the ``(M_d, K)`` numerator and takes its Frobenius inner
    product with the log centroids::

        cat_numerator_i[m, k] = Σ_t X̃_i[m, t]·gamma_wt[t, k]
        Wd_cost_slice_cat[i]  = -combined_scales[i]·Σ_{k,m} logC_cat_i[k,m]·cat_numerator_i[m,k]

    A feature given as ``(T,)`` integer codes builds its numerator by scatter-add, or by
    ``_sum_by_code_`` when ``code_scratch`` is given; a ``(T, M_d)`` feature uses a matmul.
    Elements of ``Wd_cost_slice_cat`` are overwritten.

    Also leaves every ``cat_numerator_i`` populated for the categorical centroid update.

    Args:
        Wd_cost_slice_cat: ``(D_cat,)`` output, overwritten.
        X_cat: Per feature, a ``(T, M_d)`` distribution or a ``(T,)`` long tensor of
            integer codes.
        gamma_wt: ``(T, K)`` the ``Γ ∘ Wt`` product.
        logC_cat_list: Per feature, ``(K, M_d)`` log of the categorical centroids.
        combined_scales: ``(D_cat,)`` per-feature scaling coefficients, independent of
            ``Wd``.
        cat_numerator_list: Per feature, a contiguous ``(M_d, K_max)`` output whose first
            ``M_d·K`` elements are overwritten with ``X̃_iᵀ @ gamma_wt`` as an ``(M_d, K)``
            matrix.
        scratch_KM: contiguous ``(K_max, max_d M_d)`` workspace.
        need_cost: ``False`` computes only ``cat_numerator_list`` and leaves
            ``Wd_cost_slice_cat`` untouched. Pass it when ``Wd`` is frozen.
        code_scratch: The ``(T, max_d M_d)`` and ``(_ONE_HOT_BATCHES·max_d M_d·K_max,)``
            workspaces of ``_sum_by_code_``, which then replaces the scatter-add, whose
            atomic adds on CUDA vary from run to run.
    """
    K = gamma_wt.shape[1]
    numerators = [_scratch_view(numerator, K) for numerator in cat_numerator_list]
    coded = [num_i for X_i, num_i in zip(X_cat, numerators, strict=True) if X_i.dim() == 1]
    if coded and code_scratch is None:
        torch._foreach_zero_(coded)  # the scatter-add accumulates
    features = zip(X_cat, numerators, logC_cat_list, Wd_cost_slice_cat, strict=True)
    for X_i, num_i, logC_i, slice_i in features:
        if X_i.dim() == 2:  # matrix path
            torch.mm(X_i.transpose(0, 1), gamma_wt, out=num_i)  # X̃ᵀ @ gamma_wt
        elif code_scratch is None:  # one-hot scatter-add
            num_i.index_add_(0, X_i, gamma_wt)  # num_i[code_t] += gamma_wt[t]
        else:
            _sum_by_code_(num_i, X_i, gamma_wt, *code_scratch)
        if not need_cost:
            continue  # Wd frozen: the numerator above is the only output the caller consumes
        view = _scratch_view(scratch_KM, logC_i.shape[1], K)  # (K, M_d)
        torch.mul(logC_i, num_i.transpose(0, 1), out=view)  # logC ∘ numᵀ
        torch.sum(view, dim=(0, 1), out=slice_i)  # ⟨logC, numᵀ⟩_F -> output element
    if need_cost:
        Wd_cost_slice_cat.mul_(combined_scales).neg_()  # times -combined_scales


def _sum_by_code_(
    out: torch.Tensor,
    codes: torch.Tensor,
    values: torch.Tensor,
    scratch_TM: torch.Tensor,
    scratch_BMK: torch.Tensor,
) -> None:
    """Write ``out[m] = Σ_{t: codes[t] = m} values[t]``, adding in the same order on every run.

    One-hot rows times ``values``, one product per row batch, then the batch products summed.
    """
    M, K = out.shape
    rows = codes.shape[0] // _ONE_HOT_BATCHES
    head = rows * _ONE_HOT_BATCHES
    one_hot = _scratch_view(scratch_TM, M).zero_().scatter_(1, codes.unsqueeze(1), 1.0)
    products = scratch_BMK[: _ONE_HOT_BATCHES * M * K].view(_ONE_HOT_BATCHES, M, K)
    torch.bmm(
        one_hot[:head].view(_ONE_HOT_BATCHES, rows, M).transpose(1, 2),
        values[:head].view(_ONE_HOT_BATCHES, rows, K),
        out=products,
    )
    torch.sum(products, dim=0, out=out)
    out.addmm_(one_hot[head:].T, values[head:])  # the rows after the last whole batch


def update_euclidean_centroids_(
    C_out: torch.Tensor,
    centroid_numerator: torch.Tensor,
    denom: torch.Tensor,
    empty: torch.Tensor,
) -> None:
    """Set the continuous centroids to ``C ← centroid_numerator / denom``.

    ``C_out[k, :]`` becomes the weighted mean of cluster ``k``'s instances. A cluster
    with mass at or below ``finfo(dtype).tiny`` counts as empty and is zeroed instead.
    ``C_out`` may alias ``centroid_numerator``.

    Args:
        C_out: ``(K, D)`` output, overwritten.
        centroid_numerator: ``(K, D)`` weighted sum ``gamma_wtᵀ @ X``, as left by
            :func:`compute_wd_cost_euclidean_`.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t, k]``, read-only.
        empty: ``(K,)`` bool workspace, overwritten with the empty-cluster mask.
    """
    torch.le(denom, _tiny(C_out.dtype), out=empty)  # empty clusters
    torch.div(centroid_numerator, denom.unsqueeze(1), out=C_out)  # replaced below where empty
    C_out.masked_fill_(empty.unsqueeze(1), 0.0)  # zero empty clusters


def compute_gamma_wt_(
    gamma_wt: torch.Tensor,
    denom: torch.Tensor,
    gamma: torch.Tensor,
    Wt: torch.Tensor | None = None,
) -> None:
    """Write ``gamma_wt = Γ ∘ Wt`` and its column sums ``denom[k] = Σ_t gamma_wt[t, k]``.

    ``Wt = None`` means uniform weights, so ``gamma_wt = Γ`` and ``denom`` is the plain
    column sum of ``Γ``.

    Args:
        gamma_wt: ``(T, K)`` output, overwritten.
        denom: ``(K,)`` output, overwritten with the column sums.
        gamma: ``(T, K)`` row-stochastic cluster affiliations ``Γ``.
        Wt: ``(T,)`` per-instance weights, or ``None``.
    """
    if Wt is None:
        gamma_wt.copy_(gamma)
    else:
        torch.mul(gamma, Wt.unsqueeze(1), out=gamma_wt)
    torch.sum(gamma_wt, dim=0, out=denom)  # per-cluster mass


def update_categorical_centroids_(
    C_cat: tuple[torch.Tensor, ...],
    logC_cat: tuple[torch.Tensor, ...],
    cat_numerators: Sequence[torch.Tensor],
    denom: torch.Tensor,
    empty: torch.Tensor,
) -> None:
    """Set every categorical feature's centroid distributions and refresh their log caches.

    For each feature ``i``, writes ``C_cat_i[k, m] = cat_numerator_i[m, k] / denom[k]``, so
    every row is a distribution over the feature's ``M_d`` categories, then refreshes
    ``logC_cat_i = log(max(C_cat_i, tiny))``. A cluster
    with mass at or below ``finfo(dtype).tiny`` counts as empty and takes the uniform ``1 / M_d``.

    There must be at least one feature.

    Args:
        C_cat: Per feature, a ``(K, M_d)`` output, overwritten.
        logC_cat: Per feature, a ``(K, M_d)`` output, overwritten with ``log(C_cat_i)``.
        cat_numerators: Per feature, the ``(M_d, K)`` weighted counts ``X̃_iᵀ @ (Γ ∘ Wt)``.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t, k]``, read-only.
        empty: ``(K,)`` bool workspace, overwritten with the empty-cluster mask.
    """
    tiny = _tiny(denom.dtype)
    torch.le(denom, tiny, out=empty)  # empty clusters
    for C_i, numerator_i in zip(C_cat, cat_numerators, strict=True):
        torch.div(numerator_i.transpose(0, 1), denom.unsqueeze(1), out=C_i)
        C_i.masked_fill_(empty.unsqueeze(1), 1.0 / C_i.shape[1])  # 1 / M_d where empty
    # log(max(C_cat_i, tiny)), as floored_log_ writes it, for every feature at once
    torch._foreach_copy_(logC_cat, C_cat)
    torch._foreach_clamp_min_(logC_cat, tiny)
    torch._foreach_log_(logC_cat)
