"""Input-block update step writers: feature-weight cost + centroid updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from entlearn.primitives.normalise import _eps, floored_log_


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
) -> None:
    """Write each categorical feature's discretisation error into ``Wd_cost_slice_cat``.

    Per feature ``i``, forms the ``(M_d, K)`` numerator and takes its Frobenius inner
    product with the log centroids::

        cat_numerator_i[m, k] = Σ_t X̃_i[m, t]·gamma_wt[t, k]
        Wd_cost_slice_cat[i]  = -combined_scales[i]·Σ_{k,m} logC_cat_i[k,m]·cat_numerator_i[m,k]

    A feature given as ``(T,)`` integer codes builds its numerator by scatter-add
    rather than by a matmul against a dense one-hot matrix; a ``(T, M_d)`` feature uses
    the matmul. Elements of ``Wd_cost_slice_cat`` are overwritten.

    Also leaves every ``cat_numerator_i`` populated for the categorical centroid update.

    Args:
        Wd_cost_slice_cat: ``(D_cat,)`` output, overwritten.
        X_cat: Per feature, a ``(T, M_d)`` distribution or a ``(T,)`` long tensor of
            integer codes.
        gamma_wt: ``(T, K)`` the ``Γ ∘ Wt`` product.
        logC_cat_list: Per feature, ``(K, M_d)`` log of the categorical centroids.
        combined_scales: ``(D_cat,)`` per-feature scaling coefficients, independent of
            ``Wd``.
        cat_numerator_list: Per feature, an ``(M_d, K_max)`` output whose leading
            active columns are overwritten with ``X̃_iᵀ @ gamma_wt``.
        scratch_KM: ``(K, max_d M_d)`` workspace.
        need_cost: ``False`` computes only ``cat_numerator_list`` and leaves
            ``Wd_cost_slice_cat`` untouched. Pass it when ``Wd`` is frozen.
    """
    K = gamma_wt.shape[1]
    for X_i, numerator, logC_i, scale_i, slice_i in zip(
        X_cat, cat_numerator_list, logC_cat_list, combined_scales, Wd_cost_slice_cat, strict=True
    ):
        num_i = numerator[:, :K]
        if X_i.dim() == 1:  # fast path: one-hot scatter-add
            num_i.zero_()
            num_i.index_add_(0, X_i, gamma_wt)  # num_i[code_t] += gamma_wt[t]
        else:  # matrix path
            torch.mm(X_i.transpose(0, 1), gamma_wt, out=num_i)  # X̃ᵀ @ gamma_wt
        if not need_cost:
            continue  # Wd frozen: the numerator above is the only output the caller consumes
        view = scratch_KM[:, : logC_i.shape[1]]  # (K, M_d)
        torch.mul(logC_i, num_i.transpose(0, 1), out=view)  # logC ∘ numᵀ
        torch.sum(view, dim=(0, 1), out=slice_i)  # ⟨logC, numᵀ⟩_F -> output element
        slice_i.mul_(scale_i).neg_()  # times -combined_scales[i]


def update_euclidean_centroids_(
    C_out: torch.Tensor,
    centroid_numerator: torch.Tensor,
    denom: torch.Tensor,
    empty: torch.Tensor,
) -> None:
    """Set the continuous centroids to ``C ← centroid_numerator / denom``.

    ``C_out[k, :]`` becomes the weighted mean of cluster ``k``'s instances. A cluster
    whose mass ``denom[k]`` is at or below ``finfo(dtype).eps`` counts as empty and is
    zeroed instead.

    ``C_out`` must not overlap the numerator or denominator: it briefly holds the
    protected divisors. The shared denominator remains unchanged.

    Args:
        C_out: ``(K, D)`` output, overwritten.
        centroid_numerator: ``(K, D)`` weighted sum ``gamma_wtᵀ @ X``, as left by
            :func:`compute_wd_cost_euclidean_`.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t, k]``.
        empty: ``(K,)`` bool workspace, overwritten with the empty-cluster mask.
    """
    eps = _eps(C_out.dtype)
    torch.le(denom, eps, out=empty)  # empty clusters
    # Reuse the destination for safe divisors; the shared masses remain read-only.
    C_out.copy_(denom.unsqueeze(1))
    C_out.masked_fill_(empty.unsqueeze(1), 1.0)
    torch.div(centroid_numerator, C_out, out=C_out)
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
    C_cat_i: torch.Tensor,
    logC_cat_i: torch.Tensor,
    cat_numerator_i: torch.Tensor,
    denom: torch.Tensor,
    empty: torch.Tensor,
) -> None:
    """Set one categorical feature's centroid distribution and refresh its log cache.

    Writes ``C_cat_i[k, m] = cat_numerator_i[m, k] / denom[k]``, so every row is a
    distribution over the feature's ``M_d`` categories, then refreshes
    ``logC_cat_i = log(max(C_cat_i, eps))``. A cluster whose mass ``denom[k]`` is at or
    below ``finfo(dtype).eps`` counts as empty and takes the uniform ``1 / M_d``.

    ``C_cat_i`` must not overlap the numerator or denominator, as it briefly holds the
    protected divisors. The shared denominator remains unchanged.

    Args:
        C_cat_i: ``(K, M_d)`` output, overwritten.
        logC_cat_i: ``(K, M_d)`` output, overwritten with ``log(C_cat_i)``.
        cat_numerator_i: ``(M_d, K)`` weighted counts ``X̃_iᵀ @ (Γ ∘ Wt)``.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t, k]``.
        empty: ``(K,)`` bool workspace, overwritten with the empty-cluster mask.
    """
    eps = _eps(C_cat_i.dtype)
    uniform = 1.0 / C_cat_i.shape[1]  # 1 / M_d
    torch.le(denom, eps, out=empty)  # empty cluster
    C_cat_i.copy_(denom.unsqueeze(1))
    C_cat_i.masked_fill_(empty.unsqueeze(1), 1.0)
    torch.div(cat_numerator_i.transpose(0, 1), C_cat_i, out=C_cat_i)
    C_cat_i.masked_fill_(empty.unsqueeze(1), uniform)
    floored_log_(logC_cat_i, C_cat_i)  # cache log(C_cat_i)
