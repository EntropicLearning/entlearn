"""EOMC manifold primitives: metrised residual, weighted subspace, projection.

The distance is g_k(t) = ((1+alpha)||x-mu_k||^2 - ||T_k^T (x-mu_k)||^2) / D, a mean over the
D features, as the standard input's distance is with uniform feature weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from entlearn.primitives.normalise import _eps, _scratch_view, _tiny

# Byte budget of the matrices one batched eigendecomposition takes at once.
_STACK_BLOCK_BYTES = 64 * 1024 * 1024


def _stack_length(n: int, K: int, element_size: int) -> int:
    """Return how many of ``K`` ``(n, n)`` matrices one batched ``eigh`` takes within the budget."""
    return max(1, min(K, _STACK_BLOCK_BYTES // (n * n * element_size)))


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

    Computes ``g[t,k] = ((1+alpha)·‖x_t-mu_k‖² - ‖T_kᵀ(x_t-mu_k)‖²) / D`` for the first
    ``k`` clusters, the mean over the ``D`` features. One fused matmul ``combo = X @ [muᵀ | stacked projectors]`` yields both
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
        RHS: contiguous ``(D, K_max + K_max·d)`` scratch; its first ``D·(k+k·d)`` elements
            receive the k-compact ``[muᵀ(:k) | stacked T_proj(:k)]`` right-hand side.
        combo: contiguous ``(T, K_max + K_max·d)`` scratch; its first ``T·(k+k·d)``
            elements receive the fused GEMM output.
        proj_sq: contiguous ``(T, K_max)`` scratch; its first ``T·k`` elements receive
            ``‖T_kᵀ(x-mu)‖²``.
        b_all: ``(K_max, d)`` scratch; rows ``0:k`` receive ``T_kᵀ mu_k``.
        mu_sq: ``(K_max,)`` scratch; entries ``0:k`` receive ``‖mu_k‖²``.
        X_sq_sum: ``(T,)`` precomputed ``Σ_d X[t,d]²``.
    """
    D = X.shape[1]
    d = T_proj.shape[2]
    kd = k * d

    RHS = _scratch_view(RHS, k + kd)
    combo = _scratch_view(combo, k + kd)
    # k-compact RHS: [muᵀ(:k) | stacked projectors(:k)], a contiguous (D, k+k·d) matrix
    RHS[:, :k].copy_(mu[:k].t())  # muᵀ into the first k columns
    # 3-D destination copy of the permuted projectors.
    RHS[:, k : k + kd].view(D, k, d).copy_(T_proj[:k].permute(1, 0, 2))

    # Fused GEMM: combo[:, :k]=⟨x,mu_k⟩, combo[:, k:k+kd]=T_kᵀx (flattened)
    torch.mm(X, RHS, out=combo)
    euclid_ip = combo[:, :k]  # (T, k) ⟨x_t, mu_k⟩
    proj_coords = combo[:, k : k + kd].view(X.shape[0], k, d)  # (T, k, d) T_kᵀx_t

    # ‖mu_k‖² via bmm (no (K,D) staging buffer)
    torch.bmm(mu[:k].unsqueeze(1), mu[:k].unsqueeze(2), out=mu_sq[:k].view(k, 1, 1))

    # Euclidean term from combo: (1+alpha)/D·(‖x‖² - 2⟨x,mu⟩ + ‖mu‖²) into sqdist
    sqdist.copy_(euclid_ip)
    sqdist.mul_(-2.0)
    sqdist.add_(X_sq_sum.unsqueeze(1))  # + ‖x_t‖²  (broadcast over k)
    sqdist.add_(mu_sq[:k].unsqueeze(0))  # + ‖mu_k‖²  (broadcast over t)
    sqdist.mul_((1.0 + alpha) / D)  # (1+alpha)/D·‖x-mu‖²

    # Projection term: proj_sq[t,k] = Σ_i (T_kᵀx - T_kᵀmu)_i²
    # b_all = T_kᵀ mu_k via one bmm: (k,1,D)·(k,D,d) -> (k,1,d).
    torch.bmm(mu[:k].unsqueeze(1), T_proj[:k], out=b_all[:k].view(k, 1, d))
    proj_coords.sub_(b_all[:k].unsqueeze(0))  # (proj_coords, T_kᵀmu_k), in place
    proj_coords.mul_(proj_coords)  # square in place
    proj_sq = _scratch_view(proj_sq, k)
    torch.sum(proj_coords, dim=2, out=proj_sq)  # Σ_i (...)²

    # Combine in place: sqdist ← ((1+alpha)·euclid - proj_sq) / D
    sqdist.sub_(proj_sq, alpha=1.0 / D)


def _rescaled_gram_(
    M: torch.Tensor,
    w: torch.Tensor,
    *,
    out: torch.Tensor,
    wsum_buf: torch.Tensor,
    wide: bool,
) -> torch.Tensor:
    """Write ``MᵀM / Σ_t w_t``, or ``MMᵀ / Σ_t w_t`` when ``wide``, into ``out``, rescaled.

    The rescaling to a unit largest diagonal and the eps ridge leave the eigenvectors unchanged.
    """
    # wsum = clamp_min(Σ_t w_t)
    wsum = torch.sum(w, dim=0, keepdim=True, out=wsum_buf)
    wsum.clamp_min_(_tiny(M.dtype))
    gram = torch.mm(M, M.t(), out=out) if wide else torch.mm(M.t(), M, out=out)
    gram.div_(wsum)
    # Rescale to a unit largest diagonal, then raise the diagonal by eps. A feature constant
    # in the cluster, or a row of zero weight, leaves a zero row; many interleaved zero rows
    # drive the tridiagonal form into subnormal numbers, on which MKL's eigh fails or
    # returns NaN. After the rescaling the ridge is relative, so it erases no small cloud.
    torch.amax(gram.diagonal(), dim=0, keepdim=True, out=wsum_buf)
    wsum_buf.clamp_min_(_tiny(M.dtype))
    gram.div_(wsum_buf)
    gram.diagonal().add_(_eps(M.dtype))
    return gram


def leading_eigenvectors_(
    fill_: Callable[[int, torch.Tensor], object],
    K: int,
    *,
    out: torch.Tensor,
    stack_buf: torch.Tensor,
    evals_buf: torch.Tensor,
    revidx: torch.Tensor,
) -> torch.Tensor:
    """Write the leading eigenvectors of ``K`` symmetric matrices into ``out``, in chunks.

    The matrices are decomposed in
    chunks of ``len(stack_buf)``, each by one batched ``eigh`` whose eigenvectors overwrite
    the chunk, so the stack's length bounds the memory: a one-matrix stack decomposes the
    matrices one at a time.

    Args:
        fill_: writes matrix ``k`` ``(n, n)`` into its destination.
        K: number of matrices.
        out: ``(K, n, m)`` output (modified); receives the columns ``revidx`` of each
            matrix's ascending eigenvectors.
        stack_buf: ``(c, n, n)`` scratch, ``c >= 1``.
        evals_buf: ``(c, n)`` write-only eigenvalue scratch.
        revidx: ``(m,)`` int64 ``[n-1, …, n-m]``, the leading ``m`` in descending order.

    Returns:
        ``out``.
    """
    chunk = stack_buf.shape[0]
    for first in range(0, K, chunk):
        last = min(first + chunk, K)
        stack = stack_buf[: last - first]
        for k in range(first, last):
            fill_(k, stack[k - first])
        torch.linalg.eigh(stack, out=(evals_buf[: last - first], stack))
        torch.index_select(stack, 2, revidx, out=out[first:last])
    return out


@dataclass(frozen=True)
class SubspaceWorkspace:
    """Scratch for :func:`weighted_subspaces_` on one ``(T, D)`` data matrix and ``K`` clusters.

    ``n = min(T, D)`` is the side of the matrices decomposed: ``D x D`` covariances when
    ``D <= T``, ``T x T`` Gram matrices otherwise. ``m = min(d, T)`` directions come from each.
    """

    top: torch.Tensor  # (K, D, d) every cluster's orthonormal basis
    basis: torch.Tensor  # (K, D, d) leading eigenvectors, or Mᵀu and zero columns beyond m
    r: torch.Tensor  # (K, d, d) QR's triangular factors
    sqrtw: torch.Tensor  # (T,) sqrt(W[:, k]) for the cluster k being filled
    centred: torch.Tensor  # (T, D) one cluster's M = sqrt(w)·(X - C_k)
    wsum: torch.Tensor  # (1,) the weight mass, also used for the rescaling of the Gram matrix
    # (c, n, n) a chunk of covariances or Gram matrices, also used to store their eigenvectors
    stack: torch.Tensor
    evals: torch.Tensor  # (c, n) eigenvalues
    revidx: torch.Tensor  # (m,) int64 [n-1, …, n-m], the leading m in descending order
    u: torch.Tensor | None  # (K, T, m) Gram eigenvectors when D > T

    @classmethod
    def allocate(cls, X: torch.Tensor, K: int, d: int) -> SubspaceWorkspace:
        """Allocate the scratch for ``X``'s rows, ``K`` clusters and ``d`` directions each."""
        T, D = X.shape
        n, m = min(T, D), min(d, T)  # A Gram matrix supplies at most T directions
        chunk = _stack_length(n, K, X.element_size())
        return cls(
            top=X.new_empty(K, D, d),
            basis=X.new_empty(K, D, d),
            r=X.new_empty(K, d, d),
            sqrtw=X.new_empty(T),
            centred=X.new_empty(T, D),
            wsum=X.new_empty(1),
            stack=X.new_empty(chunk, n, n),
            evals=X.new_empty(chunk, n),
            revidx=torch.arange(n - 1, n - 1 - m, -1, device=X.device),
            u=X.new_empty(K, T, m) if D > T else None,
        )


def weighted_subspaces_(
    X: torch.Tensor,
    centres: torch.Tensor,
    W: torch.Tensor,
    skip: torch.Tensor,
    *,
    out: torch.Tensor,
    workspace: SubspaceWorkspace,
) -> None:
    """Write each cluster's top-``d`` weighted-covariance eigenvectors into ``out``.

    Cluster ``k`` takes the eigenvectors of ``Σ_t W[t,k](x_t-C_k)(x_t-C_k)ᵀ / Σ_t W[t,k]``;
    ``out[k]`` keeps its value where ``skip[k]`` is True. When ``D <= T`` the ``D x D``
    covariances are decomposed. When ``D > T`` the ``T x T`` Gram matrices of the scaled-centred rows
    ``M = sqrt(w)·(X-C_k)`` are decomposed instead, and a Gram eigenvector ``u`` maps to a
    covariance eigenvector through ``Mᵀu``. Each matrix is rescaled to a unit largest
    diagonal and its diagonal raised by machine precision, which leaves its eigenvectors
    unchanged. The matrices are decomposed in chunks of ``len(workspace.stack)``, the skip
    mask is applied on the device, and one QR factorisation per cluster orthonormalises its
    ``d`` directions.

    Args:
        X: ``(T, D)`` data rows.
        centres: ``(K, D)`` centres the covariances are taken about.
        W: ``(T, K)`` non-negative per-cluster row weights.
        skip: ``(K,)`` bool, the clusters whose ``out`` stays unchanged.
        out: ``(K, D, d)`` orthonormal bases (modified).
        workspace: scratch allocated for ``X`` and at least ``K`` clusters.
    """
    T, D = X.shape
    K = out.shape[0]
    wide = D > T
    ws = workspace
    basis, top = ws.basis[:K], ws.top[:K]

    def scaled_centred(k: int) -> torch.Tensor:
        sw = torch.sqrt(W[:, k], out=ws.sqrtw)
        M = torch.sub(X, centres[k], out=ws.centred)
        return M.mul_(sw.unsqueeze(1))

    def fill_(k: int, dest: torch.Tensor) -> None:
        _rescaled_gram_(scaled_centred(k), W[:, k], out=dest, wsum_buf=ws.wsum, wide=wide)

    decompose = {"stack_buf": ws.stack, "evals_buf": ws.evals, "revidx": ws.revidx}
    if not wide:
        leading_eigenvectors_(fill_, K, out=basis, **decompose)
    else:
        assert ws.u is not None  # allocated for every D > T workspace
        u = leading_eigenvectors_(fill_, K, out=ws.u[:K], **decompose)
        m = u.shape[2]
        for k in range(K):
            torch.mm(scaled_centred(k).t(), u[k], out=basis[k, :, :m])
        basis[:, :, m:].zero_()  # The QR completes these directions
    # QR restores the orthonormality that a batched eigh may lose.
    torch.linalg.qr(basis, out=(top, ws.r[:K]))
    torch.where(skip.view(K, 1, 1), out, top, out=out)


def project(
    Y: torch.Tensor, mu: torch.Tensor, T_proj: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """EOMC reconstruction ``Y^proj[n] = Σ_k gamma[n,k]·(mu_k + T_k T_kᵀ (Y_n - mu_k))``.

    ``Y`` ``(T, D)``, ``mu`` ``(K, D)``, ``T_proj`` ``(K, D, d)``, ``gamma`` ``(T, K)``;
    returns ``(T, D)``. Splits the mixture into a data term and a centroid term,
    ``Σ_k gamma[n,k]·T_k T_kᵀ Y_n + Σ_k gamma[n,k]·(mu_k - T_k T_kᵀ mu_k)``, so three
    matrix products provide the whole reconstruction and the ``(T, K, D)`` broadcast
    tensor is never built. The largest intermediate is the ``(T, K·d)`` block of
    projected coordinates.
    """
    T, D = Y.shape
    K, _, d = T_proj.shape
    # [T_1 | … | T_K] as (D, K·d): the stacked right-hand side assemble_metrised_cost_
    # builds, so one GEMM yields every cluster's projected coordinates at once.
    T_flat = T_proj.permute(1, 0, 2).reshape(D, K * d)
    coords = Y @ T_flat  # (T, K·d) = T_kᵀ Y_n, k-major
    coords.view(T, K, d).mul_(gamma.unsqueeze(2))  # weight cluster k's block by gamma[:,k]
    out = coords @ T_flat.t()  # (T, D) = Σ_k gamma[n,k]·T_k T_kᵀ Y_n
    # The centroid term, one cluster-sized bmm pair then a single gamma-weighted mixture.
    b = torch.bmm(T_proj.transpose(1, 2), mu.unsqueeze(2))  # (K, d, 1) = T_kᵀ mu_k
    p = torch.bmm(T_proj, b).squeeze(2)  # (K, D) = T_k T_kᵀ mu_k
    out.addmm_(gamma, mu - p)  # += Σ_k gamma[n,k]·(mu_k - T_k T_kᵀ mu_k)
    return out
