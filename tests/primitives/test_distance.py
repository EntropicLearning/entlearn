"""Tests for the weighted squared-distance kernel."""

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.distance import (
    assemble_categorical_cost_,
    assemble_euclidean_cost_,
    refresh_weighted_norm_,
    weighted_sq_distance_,
    weighted_sq_norm_,
)
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn, stochastic

SHAPES = [(7, 4, 3), (10, 1, 4), (6, 3, 1), (5, 5, 5)]
CAT_SHAPES = [(7, 3, 4), (10, 2, 5), (6, 5, 3), (5, 4, 5)]  # (T, M_d, K); K <= T, M_d >= 2


def _reference(A, B, feature_weights=None):
    """Naive triple-loop oracle: out[t, k] = Σ_d w[d]·(A[t, d] - B[k, d])²."""
    T, D = A.shape
    K = B.shape[0]
    out = torch.zeros(T, K, dtype=A.dtype)
    for t in range(T):
        for k in range(K):
            s = 0.0
            for d in range(D):
                w = 1.0 if feature_weights is None else float(feature_weights[d])
                diff = float(A[t, d]) - float(B[k, d])
                s += w * diff * diff
            out[t, k] = s
    return out


def _make(T, D, K, *, seed, weighted):
    g = generator(seed)
    A = randn(T, D, g=g)
    B = randn(K, D, g=g)
    w = rand(D, g=g) if weighted else None
    A_sq_sum = ((w if w is not None else 1.0) * A * A).sum(1)
    out = empty(T, K)
    scratch_K = empty(K)
    scratch_KD = empty(K, D)
    return A, B, w, A_sq_sum, out, scratch_K, scratch_KD


class TestWeightedSqDistance:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference_weighted(self, shape, seed):
        A, B, w, A_sq_sum, out, scratch_K, scratch_KD = _make(*shape, seed=seed, weighted=True)
        assert_zero_alloc(
            weighted_sq_distance_, out, A, B, A_sq_sum, scratch_K, scratch_KD, feature_weights=w
        )
        assert_close(out, _reference(A, B, w))

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference_unweighted(self, shape, seed):
        A, B, _w, A_sq_sum, out, scratch_K, scratch_KD = _make(*shape, seed=seed, weighted=False)
        assert_zero_alloc(weighted_sq_distance_, out, A, B, A_sq_sum, scratch_K, scratch_KD)
        assert_close(out, _reference(A, B))


class TestAssembleEuclideanCost:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, _D, K = shape
        X, C, Wd, X_sq_wd_sum, _, scratch_K, scratch_KD = _make(*shape, seed=seed, weighted=True)
        g = generator(seed + 1)
        Wt = rand(T, g=g)
        cost = randn(T, K, g=g)  # pre-filled: checks +=
        cost_before = cost.clone()
        sqdist_wd = empty(T, K)
        assert_zero_alloc(
            assemble_euclidean_cost_,
            cost,
            sqdist_wd,
            X,
            C,
            Wd,
            Wt,
            X_sq_wd_sum,
            scratch_KD,
            scratch_K,
        )
        sq_ref = _reference(X, C, Wd)
        assert_close(sqdist_wd, sq_ref)
        assert_close(cost, cost_before.cpu() + Wt.cpu().unsqueeze(1) * sq_ref)


def _make_cat(T, M, K, *, seed, integer_codes, sparse=False):
    g = generator(seed)
    C = stochastic(K, M, dim=1, g=g)  # distribution over categories
    logC = C.clamp_min(torch.finfo(DTYPE).eps).log()
    if integer_codes:
        X_cat = torch.randint(0, M, (T,), generator=g, device=DEVICE)
    else:
        X_cat = rand(T, M, g=g)
        if sparse:
            keep = rand(T, M, g=g) > 0.5
            keep[
                torch.arange(T, device=DEVICE),
                torch.randint(0, M, (T,), generator=g, device=DEVICE),
            ] = True
            X_cat = X_cat * keep
        X_cat /= X_cat.sum(1, keepdim=True)
        if sparse:
            X_cat = X_cat.to_sparse_csr()
    Wt = rand(T, g=g)
    weight_scale = 0.5 + float(rand(g=g))  # positive scale
    xent_scratch = empty(T, K)
    return X_cat, logC, weight_scale, Wt, xent_scratch


def _cat_ref(X_cat, logC, *, integer_codes):
    """Naive xent oracle: out[t, k] = -Σ_m X̃[m, t]·logC[k, m] (or -logC[k, code_t])."""
    if X_cat.layout != torch.strided:  # sparse-format input: densify
        X_cat = X_cat.to_dense()
    T = X_cat.shape[0]
    K, M = logC.shape
    xent = torch.zeros(T, K, dtype=logC.dtype)
    for t in range(T):
        for k in range(K):
            if integer_codes:
                xent[t, k] = -float(logC[k, X_cat[t]])
            else:
                xent[t, k] = -sum(float(X_cat[t, m]) * float(logC[k, m]) for m in range(M))
    return xent


class TestAssembleCategoricalCost:
    @pytest.mark.parametrize("integer_codes", [False, True])
    @pytest.mark.parametrize("shape", CAT_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed, integer_codes):
        T, M, K = shape
        X_cat, logC, ws, Wt, xent_scratch = _make_cat(
            T, M, K, seed=seed, integer_codes=integer_codes
        )
        g = generator(seed + 7)
        cat_cost = randn(T, K, g=g)  # pre-filled: checks +=
        disc_cost = randn(T, K, g=g)
        cat_before, disc_before = cat_cost.clone(), disc_cost.clone()
        assert_zero_alloc(
            assemble_categorical_cost_, disc_cost, cat_cost, xent_scratch, X_cat, logC, ws, Wt
        )
        xent = _cat_ref(X_cat, logC, integer_codes=integer_codes)
        assert_close(cat_cost, cat_before.cpu() + ws * xent)
        assert_close(disc_cost, disc_before.cpu() + ws * Wt.cpu().unsqueeze(1) * xent)

    def test_accepts_a_device_scalar_without_allocation(self):
        X_cat, logC, scale, Wt, scratch = _make_cat(7, 4, 3, seed=0, integer_codes=True)
        cat_cost = empty(7, 3)
        disc_cost = empty(7, 3)
        cat_cost.zero_()
        disc_cost.zero_()

        assert_zero_alloc(
            assemble_categorical_cost_,
            disc_cost,
            cat_cost,
            scratch,
            X_cat,
            logC,
            torch.tensor(scale, dtype=DTYPE, device=DEVICE),
            Wt,
        )

        xent = _cat_ref(X_cat, logC, integer_codes=True)
        assert_close(cat_cost, scale * xent)
        assert_close(disc_cost, scale * Wt.cpu().unsqueeze(1) * xent)

    @pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta")
    @pytest.mark.parametrize("shape", CAT_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference_sparse_distribution(self, shape, seed):
        T, M, K = shape
        X_cat, logC, ws, Wt, xent_scratch = _make_cat(
            T, M, K, seed=seed, integer_codes=False, sparse=True
        )
        g = generator(seed + 7)
        cat_cost = randn(T, K, g=g)
        disc_cost = randn(T, K, g=g)
        cat_before, disc_before = cat_cost.clone(), disc_cost.clone()
        assemble_categorical_cost_(disc_cost, cat_cost, xent_scratch, X_cat, logC, ws, Wt)
        xent = _cat_ref(X_cat, logC, integer_codes=False)
        assert_close(cat_cost, cat_before.cpu() + ws * xent)
        assert_close(disc_cost, disc_before.cpu() + ws * Wt.cpu().unsqueeze(1) * xent)

    def test_int_codes_equal_onehot_distribution(self):
        # The int-code fast path must equal the distribution path on a one-hot.
        T, M, K = 6, 4, 3
        g = generator(123)
        codes = torch.randint(0, M, (T,), generator=g, device=DEVICE)
        C = stochastic(K, M, dim=1, g=g)
        logC = C.log()
        Wt = rand(T, g=g)
        ws = 0.7
        onehot = torch.zeros(T, M, dtype=DTYPE, device=DEVICE)
        onehot[torch.arange(T, device=DEVICE), codes] = 1.0
        results = []
        for x in (codes, onehot):
            cat_cost = torch.zeros(T, K, dtype=DTYPE, device=DEVICE)
            disc_cost = torch.zeros(T, K, dtype=DTYPE, device=DEVICE)
            xent_scratch = empty(T, K)
            assemble_categorical_cost_(disc_cost, cat_cost, xent_scratch, x, logC, ws, Wt)
            results.append((cat_cost, disc_cost))
        assert_close(results[0][0], results[1][0])
        assert_close(results[0][1], results[1][1])


def _xsqwd_ref(X_sq, Wd):
    """Naive oracle: out[t] = Σ_d Wd[d]·X_sq[t, d]."""
    T, D = X_sq.shape
    out = torch.zeros(T, dtype=X_sq.dtype)
    for t in range(T):
        out[t] = sum(float(Wd[d]) * float(X_sq[t, d]) for d in range(D))
    return out


class TestRefreshWeightedNorm:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, D, _K = shape
        g = generator(seed)
        X_sq = rand(T, D, g=g)  # X² is non-negative
        Wd = rand(D, g=g)
        out = empty(T)
        assert_zero_alloc(refresh_weighted_norm_, out, X_sq, Wd)
        assert_close(out, _xsqwd_ref(X_sq, Wd))


class TestWeightedSqNorm:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_the_cached_norm_path(self, shape, seed):
        T, D, _K = shape
        g = generator(seed)
        X = randn(T, D, g=g)
        Wd = rand(D, g=g)
        expected = empty(T)
        refresh_weighted_norm_(expected, X * X, Wd)
        out = empty(T)
        weighted_sq_norm_(out, X, Wd)
        assert_close(out, expected)

    def test_result_is_independent_of_the_block_size(self, monkeypatch):
        # The reduction routes through BLAS, whose round-off may differ per block
        # shape, so the block split is pinned to closeness rather than bit equality.
        from entlearn.primitives import distance as distance_module

        g = generator(7)
        X = randn(50, 6, g=g)
        Wd = rand(6, g=g)
        whole = empty(50)
        weighted_sq_norm_(whole, X, Wd)
        monkeypatch.setattr(distance_module, "_NORM_BLOCK_BYTES", 96)  # two rows per block
        blocked = empty(50).fill_(float("nan"))  # an unwritten row stays NaN
        weighted_sq_norm_(blocked, X, Wd)
        assert_close(blocked, whole)

    def test_empty_shapes_zero_the_output(self):
        out = empty(3)
        out.fill_(7.0)
        weighted_sq_norm_(out, empty(3, 0), empty(0))
        assert torch.equal(out, torch.zeros(3, dtype=DTYPE, device=DEVICE))
