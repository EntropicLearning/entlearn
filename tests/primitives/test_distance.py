"""Tests for the weighted squared-distance kernels and the seed dissimilarities."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.distance import (
    assemble_categorical_cost_,
    dissimilarity_to_rows,
    refresh_weighted_norm_,
    weighted_sq_distance_,
    weighted_sq_norm_,
)
from entlearn.primitives.normalise import _tiny
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn, stochastic

SHAPES = [(7, 4, 3), (10, 1, 4), (6, 3, 1), (5, 5, 5)]
CAT_SHAPES = [(7, 3, 4), (10, 2, 5), (6, 5, 3), (5, 4, 5)]  # (T, M_d, K); K <= T, M_d >= 2


class TestSeedingLogPrice:
    @pytest.mark.parametrize("distribution", [False, True])
    def test_missing_levels_use_the_same_tiny_log_floor_as_fitting(self, distribution):
        codes = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        feature = torch.nn.functional.one_hot(codes, 2).to(DTYPE) if distribution else codes
        X = torch.empty(2, 0, dtype=DTYPE, device=DEVICE)
        distance = dissimilarity_to_rows(X, [feature], X.new_empty(0), X.new_ones(1))
        actual = distance(torch.tensor([0], device=DEVICE))
        expected = torch.tensor([[0.0], [-math.log(_tiny(DTYPE))]], dtype=DTYPE, device=DEVICE)
        assert_close(actual, expected)


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

    def test_without_the_row_term_every_row_shifts_by_its_norm(self):
        A, B, w, A_sq_sum, out, scratch_K, scratch_KD = _make(7, 4, 3, seed=0, weighted=True)
        weighted_sq_distance_(out, A, B, None, scratch_K, scratch_KD, feature_weights=w)
        assert_close(out, _reference(A, B, w) - A_sq_sum.cpu().unsqueeze(1))


def _make_cat(T, M, K, *, seed, integer_codes):
    g = generator(seed)
    C = stochastic(K, M, dim=1, g=g)  # distribution over categories
    logC = C.clamp_min(_tiny(DTYPE)).log()
    if integer_codes:
        X_cat = torch.randint(0, M, (T,), generator=g, device=DEVICE)
    else:
        X_cat = rand(T, M, g=g)
        X_cat /= X_cat.sum(1, keepdim=True)
    weight_scale = 0.5 + float(rand(g=g))  # positive scale
    xent_scratch = empty(T, K)
    return X_cat, logC, weight_scale, xent_scratch, empty(M, K) if integer_codes else None


def _cat_ref(X_cat, logC, *, integer_codes):
    """Naive xent oracle: out[t, k] = -Σ_m X̃[m, t]·logC[k, m] (or -logC[k, code_t])."""
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
        X_cat, logC, ws, xent_scratch, scratch_MK = _make_cat(
            T, M, K, seed=seed, integer_codes=integer_codes
        )
        cost = randn(T, K, g=generator(seed + 7))  # pre-filled: checks +=
        before = cost.clone()
        assert_zero_alloc(
            assemble_categorical_cost_, cost, xent_scratch, X_cat, logC, ws, scratch_MK
        )
        added = ws * _cat_ref(X_cat, logC, integer_codes=integer_codes)
        assert_close(cost, before.cpu() + added)
        assert_close(xent_scratch, added)

    def test_accepts_a_device_scalar_without_allocation(self):
        X_cat, logC, scale, scratch, scratch_MK = _make_cat(7, 4, 3, seed=0, integer_codes=True)
        cost = empty(7, 3)
        cost.zero_()

        assert_zero_alloc(
            assemble_categorical_cost_,
            cost,
            scratch,
            X_cat,
            logC,
            torch.tensor(scale, dtype=DTYPE, device=DEVICE),
            scratch_MK,
        )

        assert_close(cost, scale * _cat_ref(X_cat, logC, integer_codes=True))

    def test_integer_codes_cost_exactly_what_their_one_hot_rows_cost(self):
        codes, logC, scale, scratch, scratch_MK = _make_cat(23, 5, 4, seed=0, integer_codes=True)
        scale = torch.tensor(scale, dtype=DTYPE, device=DEVICE)
        costs = []
        for rows in (codes, torch.nn.functional.one_hot(codes, 5).to(DTYPE)):
            cost = empty(23, 4).zero_()
            assemble_categorical_cost_(cost, scratch, rows, logC, scale, scratch_MK)
            costs.append(cost)

        assert torch.equal(costs[0], costs[1])


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

    def test_without_weights_is_the_squared_norm(self):
        X = randn(50, 6, g=generator(5))
        out = empty(50)
        weighted_sq_norm_(out, X)
        assert_close(out, (X * X).sum(dim=1))

    def test_result_is_independent_of_the_block_size(self, monkeypatch):
        # The reduction routes through BLAS, whose round-off may differ per block
        # shape, so the block split is pinned to closeness rather than bit equality.
        from entlearn.primitives import distance as distance_module

        g = generator(7)
        X = randn(50, 6, g=g)
        Wd = rand(6, g=g)
        whole = empty(50)
        weighted_sq_norm_(whole, X, Wd)
        # 96 bytes holds 2 rows of 6 features per block in float64, 4 in float32.
        monkeypatch.setattr(distance_module, "_NORM_BLOCK_BYTES", 96)
        blocked = empty(50).fill_(float("nan"))  # an unwritten row stays NaN
        weighted_sq_norm_(blocked, X, Wd)
        assert_close(blocked, whole)

    def test_empty_shapes_zero_the_output(self):
        out = empty(3)
        out.fill_(7.0)
        weighted_sq_norm_(out, empty(3, 0), empty(0))
        assert torch.equal(out, torch.zeros(3, dtype=DTYPE, device=DEVICE))


def _direct_dissimilarity(X, X_cat, Wd_cont, Wd_cat, seed):
    """Direct-difference oracle: ``Σ_d w_d (x_d - x_sd)² + Σ_i w_i·(-Σ_m p_im log max(p_sim, tiny))``."""
    difference = X - X[seed]
    d2 = (Wd_cont * difference * difference).sum(dim=1)
    for weight, rows in zip(Wd_cat, X_cat, strict=True):
        d2 += weight * -(rows * rows[seed].clamp_min(_tiny(DTYPE)).log()).sum(dim=1)
    return d2


class TestDissimilarityToRows:
    def test_dissimilarities_match_the_direct_difference(self):
        g = torch.Generator().manual_seed(2)
        X = torch.rand(50, 6, generator=g, dtype=DTYPE).to(DEVICE)
        Wd = torch.rand(6, generator=g, dtype=DTYPE).to(DEVICE)
        distribution = torch.softmax(torch.rand(50, 3, generator=g, dtype=DTYPE), dim=1).to(DEVICE)
        codes = torch.randint(0, 4, (50,), generator=g).to(DEVICE)
        one_hot = torch.nn.functional.one_hot(codes, 4).to(DTYPE)
        Wd_cat = torch.full((2,), 0.05, dtype=DTYPE, device=DEVICE)
        seeds = torch.arange(50, device=DEVICE)  # every row, so the self-dissimilarities too

        columns = dissimilarity_to_rows(X, [distribution, codes], Wd, Wd_cat)(seeds)

        direct = torch.stack(
            [_direct_dissimilarity(X, [distribution, one_hot], Wd, Wd_cat, s) for s in range(50)],
            dim=1,
        )
        # Every dissimilarity is below 8 and each term a weighted sum of at most six products,
        # so the rounding of the two forms stays within 64 eps.
        torch.testing.assert_close(columns, direct, rtol=0, atol=64 * torch.finfo(DTYPE).eps)

    def test_dissimilarities_are_never_negative(self):
        g = torch.Generator().manual_seed(2)
        X = torch.rand(50, 6, generator=g, dtype=DTYPE).to(DEVICE)
        Wd = torch.rand(6, generator=g, dtype=DTYPE).to(DEVICE)
        seeds = torch.arange(50, device=DEVICE)  # the expansion rounds some self-terms below 0

        columns = dissimilarity_to_rows(X, [], Wd, Wd[:0])(seeds)

        assert bool((columns >= 0).all())

    def test_codes_give_their_one_hot_rows_bits(self):
        codes = (torch.arange(40, device=DEVICE) * 7) % 4
        one_hot = torch.nn.functional.one_hot(codes, 4).to(DTYPE)
        X = torch.rand(40, 3, generator=torch.Generator().manual_seed(6), dtype=DTYPE).to(DEVICE)
        Wd = torch.full((3,), 0.2, dtype=DTYPE, device=DEVICE)
        seeds = torch.tensor([5, 6], device=DEVICE)  # codes 3 and 2

        from_codes = dissimilarity_to_rows(X, [codes], Wd, Wd[:1])(seeds)
        from_rows = dissimilarity_to_rows(X, [one_hot], Wd, Wd[:1])(seeds)

        assert torch.equal(from_codes, from_rows)
