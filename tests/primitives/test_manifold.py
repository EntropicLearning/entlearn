"""Primitive-level oracles for the EOMC metrised manifold ops.

Exercises the metrised projection-residual assembly, the ``w``-weighted subspace
step (eigh/SVD orientation dispatch), and the blended local reconstructions against
dense references at small ``D``/``T``. Two behaviours are pinned beyond simple
correctness:

* ``project`` accumulates per cluster and must never materialise the ``(N, K, D)``
  broadcast tensor of the naive form (a probe below fails against that form).
* ``weighted_subspace_`` drives every step through caller-allocated buffers: there
  is no internal-allocation fallback, so every call supplies the buffers its
  orientation needs.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch
from _alloc import assert_zero_alloc, count_allocations
from conftest import DEVICE, DTYPE

from entlearn.primitives.manifold import (
    assemble_metrised_cost_,
    project,
    weighted_subspace_,
)


def metrised_residual(
    X: torch.Tensor, mu: torch.Tensor, T_proj: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Metrised projection-residual distance, shape ``(N, K)``, the dense test oracle.

    ``g[n,k] = (1+alpha)||X_n - mu_k||^2 - ||T_k^T (X_n - mu_k)||^2``. ``X`` is ``(N, D)``,
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
    return (1.0 + alpha) * euclid - proj_sq


# Dtype-aware tolerances: tight on the float64 CPU lane, relaxed on the float32
# CUDA lane (which must still stay green).
_F64 = torch.float64 == DTYPE
ATOL = 1e-10 if _F64 else 1e-4
ATOL_TIGHT = 1e-12 if _F64 else 1e-5
ATOL_ORTHO = 1e-10 if _F64 else 1e-4
ATOL_PROJ = 1e-8 if _F64 else 1e-3


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _randn(*shape, seed):
    return torch.randn(*shape, dtype=DTYPE, device=DEVICE, generator=_gen(seed))


def _rand_projectors(K, D, d, *, seed):
    """Per-cluster orthonormal projectors ``(K, D, d)``."""
    return torch.linalg.qr(_randn(K, D, d, seed=seed)).Q


def _subspace_proj(T):  # projector P = T Tᵀ, basis-invariant
    return T @ T.transpose(-1, -2)


class TestMetrisedResidual:
    """The dense oracle ``g[n,k] = (1+alpha)||x-mu||^2 - ||T_k^T (x-mu)||^2``."""

    def test_metrised_residual_matches_reference(self):
        N, K, D, d = 6, 3, 5, 2
        X = _randn(N, D, seed=0)
        mu = _randn(K, D, seed=1)
        T = _rand_projectors(K, D, d, seed=2)
        alpha = 0.3
        out = metrised_residual(X, mu, T, alpha)  # (N, K)
        ref = np.zeros((N, K))
        for k in range(K):
            diff = (X - mu[k]).cpu().numpy()
            euclid = (diff**2).sum(1)
            proj = diff @ T[k].cpu().numpy()
            ref[:, k] = (1 + alpha) * euclid - (proj**2).sum(1)
        assert np.allclose(out.cpu().numpy(), ref, atol=ATOL_TIGHT)
        assert (out.cpu().numpy() >= -1e-6).all()  # Pd for alpha>0


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

    def test_matches_oracle_full_k(self):
        out, ref = self._run(T=7, K_max=4, D=5, d=2, k=4, alpha=0.3, seed=0)
        assert torch.allclose(out, ref, atol=ATOL, rtol=1e-7)

    def test_matches_oracle_d_ge_2_various_shapes(self):
        for T, K_max, D, d, alpha in [(6, 3, 5, 2, 0.0), (9, 5, 8, 3, 0.7), (4, 2, 3, 2, 1.5)]:
            out, ref = self._run(T, K_max, D, d, k=K_max, alpha=alpha, seed=K_max)
            assert torch.allclose(out, ref, atol=ATOL, rtol=1e-7), (T, K_max, D, d)

    def test_matches_oracle_pruned_k_lt_kmax(self):
        # k < K_max pins the k-compact slicing: the dead tail columns must not leak.
        out, ref = self._run(T=8, K_max=6, D=5, d=2, k=3, alpha=0.4, seed=11)
        assert torch.allclose(out, ref, atol=ATOL, rtol=1e-7)

    def test_zero_alloc(self):
        T, K_max, D, d, k, alpha = 7, 5, 6, 2, 3, 0.5
        X, mu, T_proj = _assemble_fixtures(T, K_max, D, d, seed=21)
        bufs = _assemble_buffers(T, K_max, D, d)
        torch.sum(X * X, dim=1, out=bufs["X_sq_sum"])  # pre-filled Σ_d X² (constant in fit)
        sqdist = torch.empty(T, K_max, dtype=DTYPE, device=DEVICE)
        assert_zero_alloc(assemble_metrised_cost_, sqdist[:, :k], X, mu, T_proj, alpha, k, **bufs)


def _subspace_buffers(N, D, d):
    """Allocate every buffer either orientation could write through.

    The hot-path caller allocates only the active orientation's buffers; the unit
    test hands the primitive a superset so both branches are exercisable.
    """
    m = min(N, D)
    return {
        "sqrtw_buf": torch.empty(N, dtype=DTYPE, device=DEVICE),
        "centred_buf": torch.empty(N, D, dtype=DTYPE, device=DEVICE),
        "wsum_buf": torch.empty(1, dtype=DTYPE, device=DEVICE),
        "cov_buf": torch.empty(D, D, dtype=DTYPE, device=DEVICE),
        "evals_buf": torch.empty(D, dtype=DTYPE, device=DEVICE),
        "evecs_buf": torch.empty(D, D, dtype=DTYPE, device=DEVICE),
        "revidx": torch.arange(D - 1, D - 1 - d, -1, dtype=torch.int64, device=DEVICE),
        "svd_U": torch.empty(N, m, dtype=DTYPE, device=DEVICE),
        "svd_S": torch.empty(m, dtype=DTYPE, device=DEVICE),
        "svd_Vh": torch.empty(m, D, dtype=DTYPE, device=DEVICE),
    }


def _weighted_mean(X, w):
    return (w[:, None] * X).sum(0) / w.sum()


def _dense_subspace(X, mu, w, d):
    """Dense reference: top-``d`` eigenvectors of the ``w``-weighted covariance about ``mu``."""
    diff = X - mu
    cov = (w[:, None] * diff).t() @ diff / w.sum()
    _evals, evecs = torch.linalg.eigh(cov)
    return evecs[:, -d:]  # top-d


class TestWeightedSubspace:
    """Buffers are caller-allocated (no internal fallback); every call supplies them."""

    def _call(self, X, mu, w, d):
        N, D = X.shape
        bufs = _subspace_buffers(N, D, d)
        out = torch.empty(D, d, dtype=DTYPE, device=DEVICE)
        got = weighted_subspace_(X, mu, w, d, out=out, **bufs)
        assert got is out  # writes through the supplied buffer
        return got

    @pytest.mark.parametrize(
        ("shape", "missing"),
        (
            # The eigh branch: torch.sum and torch.mm both accept out=None by allocating,
            # so a stripped assert here would break the no-allocation invariant silently.
            pytest.param((40, 6), "wsum_buf", id="eigh-wsum"),
            pytest.param((40, 6), "cov_buf", id="eigh-cov"),
            pytest.param((40, 6), "evals_buf", id="eigh-evals"),
            pytest.param((40, 6), "evecs_buf", id="eigh-evecs"),
            pytest.param((40, 6), "revidx", id="eigh-revidx"),
            pytest.param((4, 12), "svd_U", id="svd-U"),
            pytest.param((4, 12), "svd_S", id="svd-S"),
            pytest.param((4, 12), "svd_Vh", id="svd-Vh"),
        ),
    )
    def test_a_missing_branch_buffer_raises_rather_than_allocating(self, shape, missing):
        N, D = shape
        d = 2
        X = torch.randn(N, D, dtype=DTYPE, device=DEVICE)
        mu = X.mean(dim=0)
        w = torch.rand(N, dtype=DTYPE, device=DEVICE)
        bufs = _subspace_buffers(N, D, d)
        bufs[missing] = None
        out = torch.empty(D, d, dtype=DTYPE, device=DEVICE)

        with pytest.raises(ValueError, match="needs"):
            weighted_subspace_(X, mu, w, d, out=out, **bufs)

    def test_recovers_planted_plane(self):
        D, d, N = 6, 2, 400
        basis = torch.linalg.qr(_randn(D, d, seed=1)).Q  # true plane
        coords = _randn(N, d, seed=2) * 3.0
        mu_true = _randn(D, seed=3)
        X = mu_true + coords @ basis.T + 0.01 * _randn(N, D, seed=4)
        w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(5))
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)  # (D, d)
        # projector onto the recovered subspace matches the true plane
        assert torch.allclose(_subspace_proj(T), _subspace_proj(basis), atol=1e-2)
        eye = torch.eye(d, dtype=DTYPE, device=DEVICE)
        assert torch.allclose(T.t() @ T, eye, atol=ATOL_ORTHO)  # orthonormal

    def test_orientation_dispatch_agrees_with_dense(self):
        # D > N (SVD) vs D < N (eigh) must give the same subspace as the dense reference.
        for D, N in [(3, 50), (40, 12)]:
            d = 2
            X = _randn(N, D, seed=D)
            w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(D + 1))
            mu = _weighted_mean(X, w)
            T = self._call(X, mu, w, d)
            ref = _dense_subspace(X, mu, w, d)
            assert torch.allclose(_subspace_proj(T), _subspace_proj(ref), atol=ATOL_PROJ)

    def test_matches_dense_reference_d_le_n(self):
        N, D, d = 50, 6, 2  # D <= N -> eigh branch
        X = _randn(N, D, seed=10)
        w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(11))
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)
        ref = _dense_subspace(X, mu, w, d)
        assert torch.allclose(_subspace_proj(T), _subspace_proj(ref), atol=ATOL_PROJ)
        eye = torch.eye(d, dtype=DTYPE, device=DEVICE)
        assert torch.allclose(T.t() @ T, eye, atol=ATOL_ORTHO)
        # the reversed index_select yields the same top-d as the descending .flip(1).
        cov = (w[:, None] * (X - mu)).t() @ (X - mu) / w.sum()
        _evals, evecs = torch.linalg.eigh(cov)
        flipped = evecs[:, -d:].flip(1)
        revidx = torch.arange(D - 1, D - 1 - d, -1, dtype=torch.int64, device=DEVICE)
        assert torch.equal(flipped, torch.index_select(evecs, 1, revidx))

    def test_matches_dense_reference_d_gt_n(self):
        N, D, d = 12, 40, 2  # D > N -> svd branch
        X = _randn(N, D, seed=20)
        w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(21))
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)
        ref = _dense_subspace(X, mu, w, d)
        assert torch.allclose(_subspace_proj(T), _subspace_proj(ref), atol=ATOL_PROJ)
        eye = torch.eye(d, dtype=DTYPE, device=DEVICE)
        assert torch.allclose(T.t() @ T, eye, atol=ATOL_ORTHO)

    @pytest.mark.parametrize("shape", [(40, 10), (10, 40)])
    @pytest.mark.parametrize("scale", [1.0, 1e-8, 1e-17, 0.0])
    def test_rank_deficient_subspace_is_orthonormal(self, shape, scale):
        # D > N with effective rank < d: only <= d points carry non-zero weight, so the
        # weighted cloud spans < d dimensions. A snapshots-eigh would return a
        # non-orthonormal basis here; neither orientation may do so.
        N, D = shape
        d = 3
        X = _randn(N, D, seed=30) * scale
        w = torch.zeros(N, dtype=DTYPE, device=DEVICE)
        w[:2] = torch.rand(2, dtype=DTYPE, device=DEVICE, generator=_gen(31))  # 2 < d=3
        mu = _weighted_mean(X, w)
        T = self._call(X, mu, w, d)
        eye = torch.eye(d, dtype=DTYPE, device=DEVICE)
        assert torch.allclose(T.t() @ T, eye, atol=ATOL_ORTHO)  # the B1 invariant

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
        tiny = torch.finfo(DTYPE).tiny
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
        eye = torch.eye(d, dtype=DTYPE, device=DEVICE)
        assert torch.allclose(T.T @ T, eye, atol=ATOL_ORTHO, rtol=0)
        direction = torch.tensor([1, 1, 0], dtype=DTYPE, device=DEVICE) / 2**0.5
        expected = direction[:, None] @ direction[None, :]
        assert torch.allclose(_subspace_proj(T[:, :1]), expected, atol=ATOL_PROJ, rtol=0)

    def test_zero_alloc_factory(self):
        # eigh/SVD LAPACK/cuSOLVER workspace lives below the aten dispatcher (the
        # documented zero-alloc exception); the factory tracer sees only our buffers.
        for N, D, d in [(50, 6, 2), (12, 40, 2)]:  # eigh branch, svd branch
            X = _randn(N, D, seed=40 + N)
            w = torch.rand(N, dtype=DTYPE, device=DEVICE, generator=_gen(41 + N))
            mu = _weighted_mean(X, w)
            bufs = _subspace_buffers(N, D, d)
            out = torch.empty(D, d, dtype=DTYPE, device=DEVICE)
            weighted_subspace_(X, mu, w, d, out=out, **bufs)  # warm once
            with count_allocations() as counts:
                for _ in range(5):
                    weighted_subspace_(X, mu, w, d, out=out, **bufs)
            offending = {name: c for name, c in counts.items() if c}
            assert not offending, (N, D, d, offending)


def _project_naive(Y, mu, T_proj, gamma):
    """Naive reference: materialises the ``(N, K, D)`` broadcast tensor."""
    diff = Y.unsqueeze(1) - mu.unsqueeze(0)  # (N, K, D)
    coords = torch.einsum("nkD,kDd->nkd", diff, T_proj)  # (N, K, d)
    in_plane = torch.einsum("nkd,kDd->nkD", coords, T_proj)  # T_k T_kᵀ diff
    rec = mu.unsqueeze(0) + in_plane  # (N, K, D)
    return torch.einsum("nk,nkD->nD", gamma, rec)


@contextlib.contextmanager
def _trace_einsum_shapes():
    """Record the shapes of every tensor flowing through ``torch.einsum``.

    The naive ``project`` feeds the ``(N, K, D)`` broadcast tensor into einsum; the
    optimised loop form calls einsum not at all. Yields the recorded shape list.
    """
    seen: list[tuple[int, ...]] = []
    orig = torch.einsum

    def traced(equation, *operands):
        for o in operands:
            if isinstance(o, torch.Tensor):
                seen.append(tuple(o.shape))
        result = orig(equation, *operands)
        if isinstance(result, torch.Tensor):
            seen.append(tuple(result.shape))
        return result

    torch.einsum = traced  # type: ignore[assignment]
    try:
        yield seen
    finally:
        torch.einsum = orig  # type: ignore[assignment]


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
        assert np.allclose(Yp.cpu().numpy(), ref, atol=ATOL_TIGHT)
        # a point on cluster 0's affine subspace, hard-assigned, reconstructs exactly
        y = mu[0] + T[0] @ _randn(d, seed=99)
        hard = torch.tensor([[1.0, 0.0]], dtype=DTYPE, device=DEVICE)
        assert torch.allclose(project(y.unsqueeze(0), mu, T, hard), y.unsqueeze(0), atol=ATOL)

    def test_matches_naive_oracle_d_le_t(self):
        # D <= T regime.
        X, mu, T, gamma = self._fixture(N=20, K=4, D=6, d=2, seed=50)
        assert torch.allclose(project(X, mu, T, gamma), _project_naive(X, mu, T, gamma), atol=ATOL)

    def test_matches_naive_oracle_d_gt_t(self):
        # D > T regime.
        X, mu, T, gamma = self._fixture(N=5, K=3, D=12, d=3, seed=60)
        assert torch.allclose(project(X, mu, T, gamma), _project_naive(X, mu, T, gamma), atol=ATOL)

    def test_no_nkd_intermediate(self):
        # The optimised form never routes an (N, K, D) tensor through einsum; the naive
        # form does, so this probe is capable of failing (guarded below).
        X, mu, T, gamma = self._fixture(N=8, K=3, D=5, d=2, seed=70)
        N, D = X.shape
        K = mu.shape[0]
        nkd = (N, K, D)
        with _trace_einsum_shapes() as seen:
            project(X, mu, T, gamma)
        assert nkd not in seen, seen
        # capability check: the naive form does surface (N, K, D) through this probe
        with _trace_einsum_shapes() as seen_naive:
            _project_naive(X, mu, T, gamma)
        assert nkd in seen_naive
