"""Primitive-level oracles for the EOMC metrised manifold ops.

Exercises the metrised projection-residual assembly, the ``w``-weighted subspace
step (covariance or Gram orientation, decomposed in chunks), and the blended local
reconstructions against dense references at small ``D``/``T``. Two behaviours are pinned beyond simple
correctness:

* ``project`` must never materialise the ``(N, K, D)`` broadcast tensor of the naive
  form (a probe below fails against that form).
* ``weighted_subspaces_`` drives every step through the workspace its caller allocated:
  there is no internal-allocation fallback.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from _alloc import _tensors_in, assert_zero_alloc, count_allocations
from conftest import DEVICE, DTYPE
from torch.utils._python_dispatch import TorchDispatchMode

from entlearn.primitives import manifold
from entlearn.primitives.manifold import (
    SubspaceWorkspace,
    assemble_metrised_cost_,
    project,
    weighted_subspaces_,
)
from entlearn.primitives.normalise import _tiny


def metrised_residual(
    X: torch.Tensor, mu: torch.Tensor, T_proj: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Metrised projection-residual distance, shape ``(N, K)``, the dense test oracle.

    ``g[n,k] = ((1+alpha)||X_n - mu_k||^2 - ||T_k^T (X_n - mu_k)||^2) / D``, the mean over the
    ``D`` features. ``X`` is ``(N, D)``,
    ``mu`` is ``(K, D)``, ``T_proj`` is ``(K, D, d)`` orthonormal.

    This is the direct ``(N, K, D)`` form of the discretisation error: readable and
    numerically clean, but it materialises the full broadcast difference. The hot path assembles the
    same distances allocation-free via :func:`assemble_metrised_cost_`'s fused GEMM; this
    function is retained as that assembler's dense reference oracle in the tests (it is
    not called on the fitting path). This dense oracle lives
    in the test module rather than the primitives module it exercises.
    """
    diff = X.unsqueeze(1) - mu.unsqueeze(0)  # (N, K, D)
    euclid = (diff * diff).sum(-1)  # (N, K)
    proj = torch.einsum("nkD,kDd->nkd", diff, T_proj)  # (N, K, d)
    proj_sq = (proj * proj).sum(-1)  # (N, K)
    return ((1.0 + alpha) * euclid - proj_sq) / X.shape[1]


# Absolute error budgets in units of the run dtype's eps. Each bounds the round-off of
# the operands' scale rather than the result's, so the comparisons pass rtol=0.
EPS = torch.finfo(DTYPE).eps
# The cost expansion cancels terms up to (1 + alpha)·4 ≈ 10 on unit rows, summed over
# D ≤ 8: about 80 eps before the mean over the features divides it by D.
ATOL_COST = 128 * EPS
# Three products of length ≤ K·d ≤ 9 on entries of size ≤ 3.
ATOL_RECON = 32 * EPS
# LAPACK eigenvectors and singular vectors are orthonormal to O(D·eps), D ≤ 128.
ATOL_ORTHO = 64 * EPS
# A computed top-d subspace is off by about eps·λ_max / gap (Davis-Kahan): at most
# 13 eps on these clouds, 18 eps on the 128-feature one.
ATOL_PROJ = 64 * EPS
# A Householder QR leaves three columns orthonormal to about 5 eps, while an eigh alone
# leaves them 13 to 21 eps from orthonormal at D = 24 over 128 clusters.
ATOL_QR = 8 * EPS


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _randn(*shape, seed):
    return torch.randn(*shape, dtype=DTYPE, device=DEVICE, generator=_gen(seed))


def _rand_projectors(K, D, d, *, seed):
    """Per-cluster orthonormal projectors ``(K, D, d)``."""
    return torch.linalg.qr(_randn(K, D, d, seed=seed)).Q


def _subspace_proj(T):  # projector P = T Tᵀ, basis-invariant
    return T @ T.transpose(-1, -2)


def _assemble_buffers(T, K_max, D, d):
    """Allocate the scratch buffers the assembler writes through."""
    return {
        "RHS": torch.empty(D, K_max + K_max * d, dtype=DTYPE, device=DEVICE),
        "combo": torch.empty(T, K_max + K_max * d, dtype=DTYPE, device=DEVICE),
        "proj_sq": torch.empty(T, K_max, dtype=DTYPE, device=DEVICE),
        "b_all": torch.empty(K_max, d, dtype=DTYPE, device=DEVICE),
        "mu_sq": torch.empty(K_max, dtype=DTYPE, device=DEVICE),
        "X_sq_sum": torch.empty(T, dtype=DTYPE, device=DEVICE),
    }


def _assemble_fixtures(T, K_max, D, d, seed):
    """Unit-magnitude, non-coincident fixtures (the expansion is not an identity)."""
    # Unit-magnitude rows so the cancellation in the expansion stays well-conditioned.
    X = torch.nn.functional.normalize(_randn(T, D, seed=seed), dim=1)
    mu = torch.nn.functional.normalize(_randn(K_max, D, seed=seed + 1), dim=1)
    T_proj = _rand_projectors(K_max, D, d, seed=seed + 2)  # orthonormal
    return X, mu, T_proj


class TestAssembleMetrisedCost:
    """The fused zero-alloc assembler must match the ``metrised_residual`` oracle."""

    def _run(self, T, K_max, D, d, k, alpha, seed):
        X, mu, T_proj = _assemble_fixtures(T, K_max, D, d, seed)
        bufs = _assemble_buffers(T, K_max, D, d)
        torch.sum(X * X, dim=1, out=bufs["X_sq_sum"])  # pre-filled Σ_d X² (constant in fit)
        sqdist = torch.empty(T, K_max, dtype=DTYPE, device=DEVICE)
        assemble_metrised_cost_(sqdist[:, :k], X, mu, T_proj, alpha, k, **bufs)
        ref = metrised_residual(X, mu[:k], T_proj[:k], alpha)
        return sqdist[:, :k], ref

    @pytest.mark.parametrize(
        ("T", "K_max", "D", "d", "k", "alpha", "seed"),
        [
            pytest.param(7, 4, 5, 2, 4, 0.3, 0, id="full-k"),
            pytest.param(6, 3, 5, 2, 3, 0.0, 3, id="no-anisotropy"),
            pytest.param(9, 5, 8, 3, 5, 0.7, 5, id="d3"),
            pytest.param(4, 2, 3, 2, 2, 1.5, 2, id="small"),
            # k < K_max pins the k-compact slicing: the dead tail columns must not leak.
            pytest.param(8, 6, 5, 2, 3, 0.4, 11, id="pruned"),
        ],
    )
    def test_matches_oracle(self, T, K_max, D, d, k, alpha, seed):
        out, ref = self._run(T, K_max, D, d, k=k, alpha=alpha, seed=seed)
        _close(out, ref, ATOL_COST)

    @pytest.mark.parametrize("name", ["RHS", "combo", "proj_sq"])
    def test_a_strided_scratch_buffer_is_refused(self, name):
        T, K_max, D, d, k = 7, 4, 5, 2, 3
        X, mu, T_proj = _assemble_fixtures(T, K_max, D, d, seed=0)
        bufs = _assemble_buffers(T, K_max, D, d)
        wide = bufs[name]
        bufs[name] = torch.empty(wide.shape[0], wide.shape[1] + 1, dtype=DTYPE, device=DEVICE)[
            :, : wide.shape[1]
        ]  # a column slice of a wider tensor
        sqdist = torch.empty(T, k, dtype=DTYPE, device=DEVICE)
        shape = rf"shape \({wide.shape[0]}, {wide.shape[1]}\)"
        with pytest.raises(ValueError, match=rf"{shape} .* not contiguous"):
            assemble_metrised_cost_(sqdist, X, mu, T_proj, 0.5, k, **bufs)

    def test_zero_alloc(self):
        T, K_max, D, d, k, alpha = 7, 5, 6, 2, 3, 0.5
        X, mu, T_proj = _assemble_fixtures(T, K_max, D, d, seed=21)
        bufs = _assemble_buffers(T, K_max, D, d)
        torch.sum(X * X, dim=1, out=bufs["X_sq_sum"])  # pre-filled Σ_d X² (constant in fit)
        sqdist = torch.empty(T, K_max, dtype=DTYPE, device=DEVICE)
        assert_zero_alloc(assemble_metrised_cost_, sqdist[:, :k], X, mu, T_proj, alpha, k, **bufs)


def _workspace(X, d, K=1, chunk=1):
    """Allocate the scratch for ``K`` clusters of ``X``, decomposed ``chunk`` at a time."""
    n = min(X.shape)
    with patch.object(manifold, "_STACK_BLOCK_BYTES", chunk * n * n * X.element_size()):
        workspace = SubspaceWorkspace.allocate(X, K, d)
    workspace.basis.fill_(float("nan"))  # The primitive writes it all
    return workspace


def _weighted_mean(X, w):
    return (w[:, None] * X).sum(0) / w.sum()


def _dense_subspace(X, mu, w, d):
    """Dense reference: top-``d`` eigenvectors of the ``w``-weighted covariance about ``mu``."""
    diff = X - mu
    cov = (w[:, None] * diff).t() @ diff / w.sum()
    _evals, evecs = torch.linalg.eigh(cov)
    return evecs[:, -d:]  # top-d


def _clusters(N, D, K, seed):
    """Rows near a planted plane, per-cluster weights and the clusters' weighted means.

    The plane keeps the leading two directions of every cluster well separated, so the
    computed subspaces stay within ``ATOL_PROJ``.
    """
    plane = torch.linalg.qr(_randn(D, 2, seed=seed + 2)).Q
    X = 3.0 * _randn(N, 2, seed=seed) @ plane.T + 0.1 * _randn(N, D, seed=seed + 3)
    W = torch.rand(N, K, dtype=DTYPE, device=DEVICE, generator=_gen(seed + 1))
    centres = (W.t() @ X) / W.sum(0)[:, None]
    return X, W, centres


class TestWeightedSubspaces:
    """Every call writes into a caller-allocated workspace, with no internal fallback."""

    def _call(self, X, mu, w, d):
        D = X.shape[1]
        out = torch.empty(1, D, d, dtype=DTYPE, device=DEVICE)
        skip = torch.zeros(1, dtype=torch.bool, device=DEVICE)
        weighted_subspaces_(X, mu[None], w[:, None], skip, out=out, workspace=_workspace(X, d))
        return out[0]

    def test_recovers_planted_plane(self):
        D, d, N = 6, 2, 400
        basis = torch.linalg.qr(_randn(D, d, seed=1)).Q  # true plane
        coords = _randn(N, d, seed=2) * 3.0
        mu_true = _randn(D, seed=3)
        X = mu_true + coords @ basis.T + 0.01 * _randn(N, D, seed=4)
        w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(5))
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)  # (D, d)
        # Noise of std 0.01 against an in-plane spread of 3 tilts the fitted plane by
        # about (0.01 / 3)·sqrt(D / N) ≈ 4e-4; 1e-2 is a statistical bound, not round-off.
        _close(_subspace_proj(T), _subspace_proj(basis), 1e-2)
        _close(T.t() @ T, torch.eye(d, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)

    @pytest.mark.parametrize("shape", [(50, 6), (12, 40)], ids=["covariance", "gram"])
    def test_matches_dense_reference(self, shape):
        N, D = shape
        d = 2
        X = _randn(N, D, seed=10 + N)
        w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(11 + N))
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)
        ref = _dense_subspace(X, mu, w, d)
        _close(_subspace_proj(T), _subspace_proj(ref), ATOL_PROJ)
        _close(T.t() @ T, torch.eye(d, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)

    @pytest.mark.parametrize("shape", [(40, 10), (10, 40), (2, 8)])
    @pytest.mark.parametrize("scale", [1.0, 1e-8, 1e-17, 0.0])
    def test_rank_deficient_subspace_is_orthonormal(self, shape, scale):
        # Only two points have nonzero weight, so the weighted cloud spans fewer than d
        # dimensions. A back-projected Gram
        # eigenvector alone would leave a non-orthonormal basis here.
        N, D = shape
        d = 3
        X = _randn(N, D, seed=30) * scale
        w = torch.zeros(N, dtype=DTYPE, device=DEVICE)
        w[:2] = torch.rand(2, dtype=DTYPE, device=DEVICE, generator=_gen(31))  # 2 < d=3
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)
        _close(T.t() @ T, torch.eye(d, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)

    @pytest.mark.parametrize("d", [1, 2, 3])
    @pytest.mark.parametrize("collapse", ["data", "weights"])
    def test_subnormal_covariance_preserves_direction_and_orthonormality(self, d, collapse):
        # Exercise the regime behind the historical diagonal ridge: a collapsed
        # weighted cloud with nonzero subnormal covariance. The planted direction
        # gives an oracle independent of eigh; d > 1 also checks null-space completion.
        X = torch.tensor(
            [[0, 0, 0], [1, 1, 0], [-1, -1, 0], [0, 0, 0]],
            dtype=DTYPE,
            device=DEVICE,
        )
        tiny = _tiny(DTYPE)
        w = torch.ones(4, dtype=DTYPE, device=DEVICE)
        if collapse == "data":
            X.mul_(tiny**0.5 / 8)
        else:
            w[1:3] = tiny / 64
            w[3] = 0
        mu = _weighted_mean(X, w)
        M = (X - mu) * w.sqrt().unsqueeze(1)
        cov = M.T @ M / w.sum()
        assert 0 < cov.abs().max().item() < tiny

        T = self._call(X, mu, w, d)
        _close(T.T @ T, torch.eye(d, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)
        direction = torch.tensor([1, 1, 0], dtype=DTYPE, device=DEVICE) / 2**0.5
        _close(_subspace_proj(T[:, :1]), direction[:, None] @ direction[None, :], ATOL_PROJ)

    @pytest.mark.parametrize("shape", [(60, 5), (9, 30)], ids=["covariance", "gram"])
    def test_chunks_decompose_each_cluster_and_skipped_clusters_keep_their_basis(self, shape):
        N, D = shape
        K, d = 5, 2
        X, W, centres = _clusters(N, D, K, seed=50)
        skip = torch.tensor([False, True, False, False, True], device=DEVICE)
        before = _rand_projectors(K, D, d, seed=52)
        results = []
        for chunk in (1, 2, K):
            out = before.clone()
            weighted_subspaces_(X, centres, W, skip, out=out, workspace=_workspace(X, d, K, chunk))
            results.append(out)
        for out in results:
            torch.testing.assert_close(out[skip], before[skip], rtol=0, atol=0)
            for k in (~skip).nonzero().flatten().tolist():
                ref = _dense_subspace(X, centres[k], W[:, k], d)
                _close(_subspace_proj(out[k]), _subspace_proj(ref), ATOL_PROJ)

    def test_a_skipped_cluster_with_a_degenerate_matrix_keeps_its_basis(self):
        # The skipped cluster's covariance has many interleaved zero rows. MKL's
        # float32 eigh fails to converge without the ridge.
        N, D, d = 400, 200, 2
        X = torch.rand(N, D, dtype=DTYPE, device=DEVICE, generator=_gen(70))
        zero = torch.randperm(D, generator=torch.Generator().manual_seed(71))[:176].to(DEVICE)
        X[:3, zero] = 0
        W = torch.zeros(N, 2, dtype=DTYPE, device=DEVICE)
        W[:, 0], W[:3, 1] = 1.0, 1.0
        centres = torch.stack((X.mean(dim=0), torch.zeros(D, dtype=DTYPE, device=DEVICE)))
        skip = torch.tensor([False, True], device=DEVICE)
        before = _rand_projectors(2, D, d, seed=72)
        out = before.clone()
        weighted_subspaces_(X, centres, W, skip, out=out, workspace=_workspace(X, d, 2, 2))
        torch.testing.assert_close(out[1], before[1], rtol=0, atol=0)
        _close(out[0].t() @ out[0], torch.eye(d, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)

    @pytest.mark.parametrize("shape", [(300, 24), (20, 40)], ids=["covariance", "gram"])
    def test_bases_are_orthonormal_to_working_precision(self, shape):
        N, D = shape
        K, d = 128, 3
        X = torch.rand(N, D, dtype=DTYPE, device=DEVICE, generator=_gen(80))
        W = torch.rand(N, K, dtype=DTYPE, device=DEVICE, generator=_gen(81))
        centres = (W.t() @ X) / W.sum(0)[:, None]
        out = torch.empty(K, D, d, dtype=DTYPE, device=DEVICE)
        skip = torch.zeros(K, dtype=torch.bool, device=DEVICE)
        weighted_subspaces_(X, centres, W, skip, out=out, workspace=_workspace(X, d, K, K))
        identity = torch.eye(d, dtype=DTYPE, device=DEVICE).expand(K, d, d)
        _close(out.transpose(1, 2) @ out, identity, ATOL_QR)

    def test_gram_with_many_zero_weight_rows_is_decomposed(self):
        # A hard assignment gives most rows zero weight, so the Gram matrix has many
        # interleaved zero rows; without the ridge, MKL's eigh fails to converge on such a
        # matrix (float32 from 128 rows).
        N, D, d = 200, 256, 2
        X = _randn(N, D, seed=60)
        w = torch.zeros(N, dtype=DTYPE, device=DEVICE)
        members = torch.randperm(N, generator=torch.Generator().manual_seed(61))[:24]
        w[members.to(DEVICE)] = 1.0
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)
        _close(_subspace_proj(T), _subspace_proj(_dense_subspace(X, mu, w, d)), ATOL_PROJ)
        _close(T.t() @ T, torch.eye(d, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)

    @pytest.mark.parametrize("seed", [0, 19])
    def test_many_constant_features_decompose(self, seed):
        # When most features are constant, the covariance has many interleaved zero rows.
        # Its tridiagonal form decays into subnormal numbers, on which float32 MKL
        # eigh fails to converge or returns NaN unless the diagonal is raised. Both seeds
        # are known failing instances.
        N = D = 128
        live = torch.randperm(D, device=DEVICE, generator=_gen(seed))[:16]
        # Distinct scales separate the leading variances, so the Davis-Kahan bound holds.
        scales = torch.linspace(1.0, 0.1, 16, dtype=DTYPE, device=DEVICE)
        X = torch.full((N, D), 0.5, dtype=DTYPE, device=DEVICE)
        X[:, live] = (
            torch.rand(N, 16, dtype=DTYPE, device=DEVICE, generator=_gen(seed + 1)) * scales
        )
        w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(seed + 2))
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, 2)
        ref = torch.zeros(D, 2, dtype=DTYPE, device=DEVICE)
        ref[live] = _dense_subspace(X[:, live], mu[live], w, 2)
        _close(_subspace_proj(T), _subspace_proj(ref), ATOL_PROJ)
        _close(T.t() @ T, torch.eye(2, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)

    def test_a_small_second_direction_survives_a_tiny_cloud(self):
        # Variances 100·eps² and 0.1·eps²: an absolute ridge of eps would round the second
        # away, whereas one applied after rescaling to a unit largest variance keeps it.
        N, D, d = 400, 6, 2
        basis = torch.linalg.qr(_randn(D, d, seed=50)).Q
        spread = torch.tensor([1.0, 1e-3], dtype=DTYPE, device=DEVICE).sqrt()
        X = 10 * EPS * (_randn(N, d, seed=51) * spread) @ basis.T
        w = torch.ones(N, dtype=DTYPE, device=DEVICE)
        T = self._call(X, _weighted_mean(X, w), w, d)
        # The exact plane is spanned by the rows; round-off of order eps relative to the
        # largest variance tilts it by about eps·sqrt(N)/1e-3 ≈ 2e4 eps.
        _close(_subspace_proj(T), _subspace_proj(basis), 2**16 * EPS)

    def test_the_selected_basis_is_orthonormalised(self, monkeypatch):
        # Some eigensolvers return a leading eigenvector off unit norm; the selected
        # columns are still orthonormal and span the same leading directions.
        eigh = torch.linalg.eigh

        def unnormalised(A, *, out):
            evals, evecs = eigh(A, out=out)
            evecs[..., -1].mul_(1.25)
            return evals, evecs

        N, D, d = 50, 6, 2
        X = _randn(N, D, seed=10)
        w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(11))
        mu = _weighted_mean(X, w)
        ref = _dense_subspace(X, mu, w, d)
        monkeypatch.setattr(torch.linalg, "eigh", unnormalised)
        T = self._call(X, mu, w, d)
        _close(T.t() @ T, torch.eye(d, dtype=DTYPE, device=DEVICE), ATOL_ORTHO)
        _close(_subspace_proj(T), _subspace_proj(ref), ATOL_PROJ)

    def test_zero_alloc_factory(self):
        # eigh/QR LAPACK/cuSOLVER workspace lives below the aten dispatcher (the
        # documented zero-alloc exception); the factory tracer sees only our buffers.
        for N, D, K, chunk in [(50, 6, 3, 2), (12, 40, 3, 2)]:  # covariance, Gram
            d = 2
            X, W, centres = _clusters(N, D, K, seed=40 + N)
            skip = torch.zeros(K, dtype=torch.bool, device=DEVICE)
            workspace = _workspace(X, d, K, chunk)
            out = torch.empty(K, D, d, dtype=DTYPE, device=DEVICE)
            weighted_subspaces_(X, centres, W, skip, out=out, workspace=workspace)  # warm once
            with count_allocations() as counts:
                for _ in range(5):
                    weighted_subspaces_(X, centres, W, skip, out=out, workspace=workspace)
            offending = {name: c for name, c in counts.items() if c}
            assert not offending, (N, D, d, offending)


def _project_naive(Y, mu, T_proj, gamma):
    """Naive reference: materialises the ``(N, K, D)`` broadcast tensor."""
    diff = Y.unsqueeze(1) - mu.unsqueeze(0)  # (N, K, D)
    coords = torch.einsum("nkD,kDd->nkd", diff, T_proj)  # (N, K, d)
    in_plane = torch.einsum("nkd,kDd->nkD", coords, T_proj)  # T_k T_kᵀ diff
    rec = mu.unsqueeze(0) + in_plane  # (N, K, D)
    return torch.einsum("nk,nkD->nD", gamma, rec)


class _LargestOutput(TorchDispatchMode):
    """Record the largest element count of any tensor an aten operation produces."""

    def __init__(self):
        super().__init__()
        self.numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in _tensors_in(result):
            self.numel = max(self.numel, tensor.numel())
        return result


class TestProject:
    """Blend local reconstructions without the ``(N, K, D)`` intermediate."""

    def _fixture(self, N, K, D, d, seed):
        X = _randn(N, D, seed=seed)
        mu = _randn(K, D, seed=seed + 1)
        T = _rand_projectors(K, D, d, seed=seed + 2)
        gamma = torch.softmax(_randn(N, K, seed=seed + 3), dim=1)
        return X, mu, T, gamma

    def test_reconstructs_on_subspace(self):
        N, K, D, d = 5, 2, 4, 2
        X, mu, T, gamma = self._fixture(N, K, D, d, seed=3)
        Yp = project(X, mu, T, gamma)  # (N, D)
        ref = np.zeros((N, D))
        for n in range(N):
            for k in range(K):
                diff = X[n] - mu[k]
                rec = mu[k] + T[k] @ (T[k].t() @ diff)
                ref[n] += gamma[n, k].item() * rec.cpu().numpy()
        np.testing.assert_allclose(Yp.cpu().numpy(), ref, rtol=0, atol=ATOL_RECON)
        # a point on cluster 0's affine subspace, hard-assigned, reconstructs exactly
        y = mu[0] + T[0] @ _randn(d, seed=99)
        hard = torch.tensor([[1.0, 0.0]], dtype=DTYPE, device=DEVICE)
        _close(project(y.unsqueeze(0), mu, T, hard), y.unsqueeze(0), ATOL_RECON)

    def test_matches_naive_oracle(self):
        X, mu, T, gamma = self._fixture(N=5, K=3, D=12, d=3, seed=60)
        _close(project(X, mu, T, gamma), _project_naive(X, mu, T, gamma), ATOL_RECON)

    def test_no_nkd_intermediate(self):
        # Every aten output counts, so a broadcast, matmul or reshape route is caught; the
        # naive form trips the probe, so it can fail.
        X, mu, T, gamma = self._fixture(N=8, K=3, D=5, d=2, seed=70)
        nkd = X.shape[0] * mu.shape[0] * X.shape[1]
        with _LargestOutput() as probe:
            project(X, mu, T, gamma)
        assert probe.numel < nkd
        with _LargestOutput() as naive_probe:
            _project_naive(X, mu, T, gamma)
        assert naive_probe.numel >= nkd
