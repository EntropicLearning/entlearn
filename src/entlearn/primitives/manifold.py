"""EOMC manifold primitives: metrised residual, weighted subspace, projection.

The distance is g_k(t) = (1+alpha)||x-mu_k||^2 - ||T_k^T (x-mu_k)||^2.
"""

from __future__ import annotations

import torch


def assemble_metrised_cost_(
    sqdist: torch.Tensor,
    X: torch.Tensor,
    mu: torch.Tensor,
    T_proj: torch.Tensor,
    alpha: float,
    k: int,
    *,
    RHS: torch.Tensor,
    combo: torch.Tensor,
    proj_sq: torch.Tensor,
    b_all: torch.Tensor,
    mu_sq: torch.Tensor,
    X_sq_sum: torch.Tensor,
) -> None:
    """Write the metrised distance ``g[t,k]`` into ``sqdist``.

    Computes ``g[t,k] = (1+alpha)·‖x_t-mu_k‖² - ‖T_kᵀ(x_t-mu_k)‖²`` for the first ``k``
    clusters. One fused matmul ``combo = X @ [muᵀ | stacked projectors]`` yields both
    the inner product ``⟨x_t, mu_k⟩`` and the projected coordinates ``T_kᵀ x_t``, and
    the Euclidean term comes out of ``combo`` by
    ``‖x-mu‖² = ‖x‖² - 2⟨x,mu⟩ + ‖mu‖²``.

    Args:
        sqdist: ``(T, k)`` output, overwritten with the metrised distances.
        X: ``(T, D)`` continuous data.
        mu: ``(K_max, D)`` centroids; only the first ``k`` rows are read.
        T_proj: ``(K_max, D, d)`` orthonormal projectors; only the first ``k`` read.
        alpha: scalar metric anisotropy ``≥ 0``.
        k: number of active clusters (``≤ K_max``); pins the k-compact slicing.
        RHS: ``(D, K_max + K_max·d)`` scratch; columns ``0:k+k·d`` receive the
            k-compact ``[muᵀ(:k) | stacked T_proj(:k)]`` right-hand side.
        combo: ``(T, K_max + K_max·d)`` scratch; columns ``0:k+k·d`` receive the
            fused GEMM output.
        proj_sq: ``(T, K_max)`` scratch; columns ``0:k`` receive ``‖T_kᵀ(x-mu)‖²``.
        b_all: ``(K_max, d)`` scratch; rows ``0:k`` receive ``T_kᵀ mu_k``.
        mu_sq: ``(K_max,)`` scratch; entries ``0:k`` receive ``‖mu_k‖²``.
        X_sq_sum: ``(T,)`` precomputed ``Σ_d X[t,d]²``.
    """
    D = X.shape[1]
    d = T_proj.shape[2]
    kd = k * d

    # k-compact RHS: [muᵀ(:k) | stacked projectors(:k)] in columns 0:k+k·d
    RHS[:, :k].copy_(mu[:k].t())  # muᵀ into the first k columns
    # 3-D destination copy of the permuted projectors.
    RHS[:, k : k + kd].view(D, k, d).copy_(T_proj[:k].permute(1, 0, 2))

    # Fused GEMM: combo[:, :k]=⟨x,mu_k⟩, combo[:, k:k+kd]=T_kᵀx (flattened)
    torch.mm(X, RHS[:, : k + kd], out=combo[:, : k + kd])
    euclid_ip = combo[:, :k]  # (T, k) ⟨x_t, mu_k⟩
    proj_coords = combo[:, k : k + kd].view(X.shape[0], k, d)  # (T, k, d) T_kᵀx_t

    # ‖mu_k‖² via bmm (no (K,D) staging buffer)
    torch.bmm(mu[:k].unsqueeze(1), mu[:k].unsqueeze(2), out=mu_sq[:k].view(k, 1, 1))

    # Euclidean term from combo: (1+alpha)·(‖x‖² - 2⟨x,mu⟩ + ‖mu‖²) into sqdist
    sqdist.copy_(euclid_ip)
    sqdist.mul_(-2.0)
    sqdist.add_(X_sq_sum.unsqueeze(1))  # + ‖x_t‖²  (broadcast over k)
    sqdist.add_(mu_sq[:k].unsqueeze(0))  # + ‖mu_k‖²  (broadcast over t)
    sqdist.mul_(1.0 + alpha)  # (1+alpha)·‖x-mu‖²

    # Projection term: proj_sq[t,k] = Σ_i (T_kᵀx - T_kᵀmu)_i²
    # b_all = T_kᵀ mu_k via one bmm: (k,1,D)·(k,D,d) -> (k,1,d).
    torch.bmm(mu[:k].unsqueeze(1), T_proj[:k], out=b_all[:k].view(k, 1, d))
    proj_coords.sub_(b_all[:k].unsqueeze(0))  # (proj_coords, T_kᵀmu_k), in place
    proj_coords.mul_(proj_coords)  # square in place
    torch.sum(proj_coords, dim=2, out=proj_sq[:, :k])  # Σ_i (...)²

    # Combine in place: sqdist ← (1+alpha)·euclid - proj_sq
    sqdist.sub_(proj_sq[:, :k])


def weighted_subspace_(
    X: torch.Tensor,
    mu: torch.Tensor,
    w: torch.Tensor,
    d: int,
    *,
    out: torch.Tensor,
    sqrtw_buf: torch.Tensor,
    centred_buf: torch.Tensor,
    wsum_buf: torch.Tensor | None = None,
    cov_buf: torch.Tensor | None = None,
    evals_buf: torch.Tensor | None = None,
    evecs_buf: torch.Tensor | None = None,
    revidx: torch.Tensor | None = None,
    svd_U: torch.Tensor | None = None,
    svd_S: torch.Tensor | None = None,
    svd_Vh: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-``d`` eigenvectors of the ``w``-weighted covariance of ``X`` about ``mu``.

    ``Cov = Σ_t w_t (x_t-mu)(x_t-mu)ᵀ / Σ_t w_t``; writes its ``d`` dominant
    eigenvectors ``(D, d)`` into ``out`` and returns it. When ``D <= N`` it takes the
    eigendecomposition of the ``D x D`` covariance. When ``D > N`` it takes the SVD of
    the ``N x D`` scaled-centred matrix instead. The covariance is rescaled without
    shifting its spectrum, and the selected eigenvectors are copied directly into ``out``.

    Always pass ``out``, ``sqrtw_buf`` and ``centred_buf``, plus the set for whichever
    branch runs: ``wsum_buf``, ``cov_buf``, ``evals_buf``, ``evecs_buf`` and ``revidx``
    for ``D <= N``, or ``svd_U``, ``svd_S`` and ``svd_Vh`` for ``D > N``.

    Args:
        X: ``(N, D)`` data rows.
        mu: ``(D,)`` centre the covariance is taken about.
        w: ``(N,)`` non-negative instance weights.
        d: number of dominant eigenvectors to return.
        out: ``(D, d)`` output, overwritten with the top-``d`` eigenvectors.
        sqrtw_buf: ``(N,)`` scratch; receives ``sqrt(w)``.
        centred_buf: ``(N, D)`` scratch; receives the scaled-centred ``sqrt(w)·(X-mu)``.
        wsum_buf: ``(1,)`` scratch (D≤N); receives weight mass and covariance scale.
        cov_buf: ``(D, D)`` scratch (D≤N); receives the rescaled covariance.
        evals_buf: ``(D,)`` write-only eigenvalue scratch (D≤N).
        evecs_buf: ``(D, D)`` scratch (D≤N); receives the eigenvectors.
        revidx: ``(d,)`` int64 ``[D-1, …, D-d]`` (D≤N); selects the top-``d`` descending.
        svd_U: ``(N, m)`` write-only left-singular scratch (D>N), ``m = min(N, D)``.
        svd_S: ``(m,)`` write-only singular-value scratch (D>N).
        svd_Vh: ``(m, D)`` scratch (D>N), or ``(D, D)`` when ``d > N`` needs
            an orthonormal completion; receives the right singular vectors.

    Returns:
        The ``out`` tensor holding the top-``d`` eigenvectors ``(D, d)``.

    Raises:
        ValueError: If a required buffer is not supplied.
    """
    N, D = X.shape
    # M = sqrt(w)·(X - mu): the scaled-centred matrix is orientation-independent, so
    # build it once (into centred_buf) before the eigh/SVD dispatch below.
    sw = torch.sqrt(w, out=sqrtw_buf)  # (N,)
    M = torch.sub(X, mu, out=centred_buf)  # (N, D) centred
    M.mul_(sw.unsqueeze(1))  # scale rows by sqrt(w)
    if D <= N:  # eigh the DxD covariance
        # A raise is used here instead of an assert, since `python -O` strips asserts, and `torch.sum`/`torch.mm` accept `out=None` by allocating.
        if wsum_buf is None or cov_buf is None or evals_buf is None or evecs_buf is None:
            raise ValueError("the D <= N branch needs wsum_buf, cov_buf, evals_buf and evecs_buf")
        if revidx is None:
            raise ValueError("the D <= N branch needs revidx")
        # wsum = clamp_min(Σ_t w_t)
        wsum = torch.sum(w, dim=0, keepdim=True, out=wsum_buf)
        wsum.clamp_min_(torch.finfo(X.dtype).tiny)
        # cov = MᵀM / wsum into cov_buf.
        cov = torch.mm(M.t(), M, out=cov_buf)
        cov.div_(wsum)
        # Scalar rescaling imposes no absolute noise floor. An absolute diagonal
        # shift can instead erase the signal of a small but nonzero cloud.
        torch.amax(cov.diagonal(), dim=0, keepdim=True, out=wsum_buf)
        wsum_buf.clamp_min_(torch.finfo(X.dtype).tiny)
        cov.div_(wsum_buf)
        # eigh (ascending); top-d descending via the reversed-index select.
        _evals, evecs = torch.linalg.eigh(cov, out=(evals_buf, evecs_buf))
        # revidx = [D-1, …, D-d] selects the top-d columns in descending order
        return torch.index_select(evecs, 1, revidx, out=out)
    # SVD the thin NxD matrix
    if svd_U is None or svd_S is None or svd_Vh is None:
        raise ValueError("the D > N branch needs svd_U, svd_S and svd_Vh")
    full = d > N
    if svd_Vh.shape != (D if full else N, D):
        raise ValueError("svd_Vh must hold the requested right singular basis")
    # gesvd on CUDA: the default Jacobi driver loses weak directions (see seed_projectors).
    _U, _S, Vh = torch.linalg.svd(
        M, full_matrices=full, driver="gesvd" if M.is_cuda else None, out=(svd_U, svd_S, svd_Vh)
    )
    out.copy_(Vh[:d].t())  # right singular vectors = eigenvectors
    return out


def project(
    Y: torch.Tensor, mu: torch.Tensor, T_proj: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """EOMC reconstruction ``Y^proj[n] = Σ_k gamma[n,k]·(mu_k + T_k T_kᵀ (Y_n - mu_k))``.

    ``Y`` ``(N, D)``, ``mu`` ``(K, D)``, ``T_proj`` ``(K, D, d)``, ``gamma`` ``(N, K)``;
    returns ``(N, D)``. Splits the mixture into a data term and a centroid term,
    ``Σ_k gamma[n,k]·T_k T_kᵀ Y_n + Σ_k gamma[n,k]·(mu_k - T_k T_kᵀ mu_k)``, so three
    matrix products provide the whole reconstruction and the ``(N, K, D)`` broadcast
    tensor is never built. The largest intermediate is the ``(N, K·d)`` block of
    projected coordinates.
    """
    N, D = Y.shape
    K, _, d = T_proj.shape
    # [T_1 | … | T_K] as (D, K·d): the stacked right-hand side assemble_metrised_cost_
    # builds, so one GEMM yields every cluster's projected coordinates at once.
    T_flat = T_proj.permute(1, 0, 2).reshape(D, K * d)
    coords = Y @ T_flat  # (N, K·d) = T_kᵀ Y_n, k-major
    coords.view(N, K, d).mul_(gamma.unsqueeze(2))  # weight cluster k's block by gamma[:,k]
    out = coords @ T_flat.t()  # (N, D) = Σ_k gamma[n,k]·T_k T_kᵀ Y_n
    # The centroid term, one cluster-sized bmm pair then a single gamma-weighted mixture.
    b = torch.bmm(T_proj.transpose(1, 2), mu.unsqueeze(2))  # (K, d, 1) = T_kᵀ mu_k
    p = torch.bmm(T_proj, b).squeeze(2)  # (K, D) = T_k T_kᵀ mu_k
    out.addmm_(gamma, mu - p)  # += Σ_k gamma[n,k]·(mu_k - T_k T_kᵀ mu_k)
    return out
