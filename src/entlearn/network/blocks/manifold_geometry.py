"""Local subspaces and operation-local manifold distance scratch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from entlearn._warnings import _warn
from entlearn.primitives.manifold import assemble_metrised_cost_, weighted_subspace_
from entlearn.primitives.normalise import _eps


@dataclass
class _SubspaceWorkspace:
    """Scratch for the fixed orientation of one weighted subspace calculation."""

    buffers: dict[str, torch.Tensor]

    @classmethod
    def allocate(cls, X: torch.Tensor, d: int) -> _SubspaceWorkspace:
        """Allocate only the buffers used by the covariance or SVD orientation."""
        T, D = X.shape
        dtype, device = X.dtype, X.device
        buffers = {
            "sqrtw_buf": torch.empty(T, dtype=dtype, device=device),  # (T,) √w_t
            "centred_buf": torch.empty(T, D, dtype=dtype, device=device),  # M_t = √w_t(x_t-C_k)
        }
        if D <= T:
            buffers.update(
                wsum_buf=torch.empty(1, dtype=dtype, device=device),  # (1,) Σ_t w_t
                cov_buf=torch.empty(D, D, dtype=dtype, device=device),  # MᵀM / Σ_t w_t
                evals_buf=torch.empty(D, dtype=dtype, device=device),  # Covariance eigenvalues
                evecs_buf=torch.empty(D, D, dtype=dtype, device=device),  # Eigenvectors as columns
                revidx=torch.arange(D - 1, D - 1 - d, -1, device=X.device),  # Leading d, descending
            )
        else:
            # M = U diag(S) Vᵀ. For d > T, complete Vᵀ to a full D-dimensional basis.
            buffers.update(
                svd_U=torch.empty(T, T, dtype=dtype, device=device),  # Left singular vectors
                svd_S=torch.empty(T, dtype=dtype, device=device),  # Singular values
                svd_Vh=torch.empty(D if d > T else T, D, dtype=dtype, device=device),  # Vᵀ
            )
        return cls(buffers)

    def write_(
        self, X: torch.Tensor, centre: torch.Tensor, weights: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Write the leading weighted subspace about the supplied centre."""
        weighted_subspace_(X, centre, weights, out.shape[1], out=out, **self.buffers)


def _complete_initial_basis(supported: torch.Tensor, d: int) -> torch.Tensor:
    """Keep supported directions and complete from coordinate axes.

    For each coordinate vector eᵢ, remove projections onto every retained qⱼ:
    v ← v - (qⱼᵀv)qⱼ. Repeat this orthogonalisation pass to reduce round-off.
    If ‖v‖ is above the dimension-scaled precision floor, append q = v / ‖v‖.
    The resulting columns satisfy QᵀQ ≈ I.
    """
    vectors = []
    for column in supported.unbind(dim=1):
        vector = column.clone()
        if vector[vector.abs().argmax()] < 0:
            vector.neg_()
        vectors.append(vector)
    if len(vectors) >= d:
        return torch.stack(vectors[:d], dim=1)
    features = supported.shape[0]
    # Unit coordinate vectors and a D-term projection give a D·ε precision scale.
    tolerance = features * _eps(supported.dtype)
    for axis in range(features):
        vector = supported.new_zeros(features)
        vector[axis] = 1
        # Two passes remove round-off along the retained directions. Each vector
        # is contiguous, so the requested number of basis vectors cannot alter the arithmetic.
        for _ in range(2):
            for previous in vectors:
                vector.sub_(torch.dot(previous, vector) * previous)
        norm = torch.linalg.vector_norm(vector)
        if norm <= tolerance:
            continue
        vectors.append(vector / norm)
        if len(vectors) == d:
            return torch.stack(vectors, dim=1)
    raise ValueError("could not complete the orthonormal manifold basis")


def seed_projectors(X: torch.Tensor, centroids: torch.Tensor, d: int) -> torch.Tensor:
    """Preserve supported local directions and deterministically complete missing ones.

    Rank and the ordered supported basis are independent of requested dimension.
    Missing directions come from orthogonalised coordinate axes, not random draws.
    A warning identifies clusters whose data cannot determine the requested basis.
    """
    projectors = torch.empty(centroids.shape[0], X.shape[1], d, dtype=X.dtype, device=X.device)
    nearest = torch.cdist(X, centroids).argmin(dim=1)
    deficient = []
    for index, centre in enumerate(centroids):
        members = nearest == index
        centred = X[members] - centre
        if centred.shape[0]:
            # Use one thin SVD for both rank and directions.
            _, singular_values, right = torch.linalg.svd(
                centred, full_matrices=False, driver="gesvd" if centred.is_cuda else None
            )
            threshold = max(centred.shape) * _eps(X.dtype) * singular_values[0]
            rank = int((singular_values > threshold).sum())
            supported = right[:rank].T
        else:
            rank = 0
            supported = X.new_empty((X.shape[1], 0))
        projectors[index].copy_(_complete_initial_basis(supported, d))
        if rank < d:
            deficient.append(f"{index} (rank={rank})")
    if deficient:
        _warn(
            f"manifold initialisation requested dimension {d}, but cluster(s) "
            f"{', '.join(deficient)} have insufficient rank; supported directions "
            "were preserved and missing directions completed deterministically. "
            "The additional directions are not identified by the data.",
            UserWarning,
        )
    return projectors


@dataclass
class _ManifoldDistance:
    """Fused distance buffers bound to one operation's rows."""

    X_sq_sum: torch.Tensor  # (T,) ‖x_t‖², precomputed for the fixed rows
    RHS: torch.Tensor  # (D, K_max(1+d)) packed [Cᵀ | P_1 | … | P_K] for a single GEMM
    combo: torch.Tensor  # (T, K_max(1+d)) XCᵀ and XP_k; latter centred and squared in place
    proj_sq: torch.Tensor  # (T, K_max) ‖P_kᵀ(x_t-C_k)‖², subtracted from the Euclidean term
    b_all: torch.Tensor  # (K_max, d) P_kᵀC_k, centres the local projected coordinates
    mu_sq: torch.Tensor  # (K_max,) ‖C_k‖², completes ‖x_t-C_k‖²

    @classmethod
    def allocate(cls, X: torch.Tensor, K: int, d: int) -> _ManifoldDistance:
        """Allocate the fused distance buffers for the operation's initial cluster count."""
        T, D = X.shape
        dtype, device = X.dtype, X.device
        return cls(
            X_sq_sum=X.square().sum(dim=1),
            RHS=torch.empty(D, K * (1 + d), dtype=dtype, device=device),
            combo=torch.empty(T, K * (1 + d), dtype=dtype, device=device),
            proj_sq=torch.empty(T, K, dtype=dtype, device=device),
            b_all=torch.empty(K, d, dtype=dtype, device=device),
            mu_sq=torch.empty(K, dtype=dtype, device=device),
        )

    def assemble_(
        self,
        out: torch.Tensor,
        X: torch.Tensor,
        centroids: torch.Tensor,
        projectors: torch.Tensor,
        alpha: float,
    ) -> None:
        """Write the unweighted metrised distance for the active clusters."""
        assemble_metrised_cost_(
            out,
            X,
            centroids,
            projectors,
            alpha,
            centroids.shape[0],
            RHS=self.RHS,
            combo=self.combo,
            proj_sq=self.proj_sq,
            b_all=self.b_all,
            mu_sq=self.mu_sq,
            X_sq_sum=self.X_sq_sum,
        )
