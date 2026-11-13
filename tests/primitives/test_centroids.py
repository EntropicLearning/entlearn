"""Tests for the input-block update-step / feature-weight primitives (centroids.py)."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.centroids import (
    _ONE_HOT_BATCHES,
    compute_gamma_wt_,
    compute_wd_cost_categorical_,
    compute_wd_cost_euclidean_,
    update_categorical_centroids_,
    update_euclidean_centroids_,
)
from entlearn.primitives.normalise import _tiny
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn, stochastic

SHAPES = [(7, 4, 3), (10, 1, 4), (6, 3, 1), (5, 5, 5)]  # (T, D, K); K <= T
CAT_SHAPES = [(7, 3, [4, 2]), (10, 4, [2]), (6, 2, [5, 3, 4]), (5, 5, [3, 2])]  # (T, K, [M_d])
# (T, K, [M_d for each feature]); features 0, 2, … are int-coded and the others are
# distributions, so the cases mix both X̃ representations and several code features.
WD_CAT_SHAPES = [
    (7, 3, [4, 2]),
    (10, 4, [3, 5, 2]),
    (6, 2, [5]),
    (5, 5, [2, 4, 3]),
]


def _make(T, D, K, *, seed):
    g = generator(seed)
    X = randn(T, D, g=g)
    C = randn(K, D, g=g)
    gamma = stochastic(T, K, dim=1, g=g)
    Wt = rand(T, g=g)
    gamma_wt = gamma * Wt[:, None]  # Γ ∘ Wt
    denom = gamma_wt.sum(0)
    X_sq = X * X
    return X, C, gamma_wt, denom, X_sq


def _ref(X, C, gamma_wt):
    """Naive oracle for (b_d, numer) of compute_wd_cost_euclidean_."""
    T, D = X.shape
    K = C.shape[0]
    numer = torch.zeros(K, D, dtype=X.dtype)
    for k in range(K):
        for d in range(D):
            numer[k, d] = sum(float(gamma_wt[t, k]) * float(X[t, d]) for t in range(T))
    b_d = torch.zeros(D, dtype=X.dtype)
    for d in range(D):
        b_d[d] = sum(
            float(gamma_wt[t, k]) * (float(X[t, d]) - float(C[k, d])) ** 2
            for t in range(T)
            for k in range(K)
        )
    return b_d, numer


def _ref_update(numer, denom):
    """Naive implementation of update_euclidean_centroids_: numer/denom, mass <= tiny -> 0."""
    K, D = numer.shape
    C = torch.zeros(K, D, dtype=numer.dtype)
    for k in range(K):
        if float(denom[k]) > _tiny(numer.dtype):
            for d in range(D):
                C[k, d] = float(numer[k, d]) / float(denom[k])
    return C


def _ref_inputs(gamma, Wt):
    """Naive implementation of compute_gamma_wt_: gamma_wt = Γ∘Wt, denom = Σ_t."""
    T, K = gamma.shape
    gamma_wt = torch.zeros(T, K, dtype=gamma.dtype)
    for t in range(T):
        for k in range(K):
            gamma_wt[t, k] = float(gamma[t, k]) * float(Wt[t])
    denom = torch.zeros(K, dtype=gamma.dtype)
    for k in range(K):
        denom[k] = sum(float(gamma_wt[t, k]) for t in range(T))
    return gamma_wt, denom


def _make_cat(T, K, M_list, *, seed):
    """Realistic categorical numerators + mass with Σ_m numerator_i[m,k] = denom[k]."""
    g = generator(seed)
    gamma = stochastic(T, K, dim=1, g=g)
    Wt = rand(T, g=g)
    gamma_wt = gamma * Wt[:, None]
    # left-stochastic columns (Σ_m = 1) times gamma_wt: one (M_d, K) numerator per feature
    cat_numerators = [stochastic(M, T, dim=0, g=g) @ gamma_wt for M in M_list]
    denom = gamma_wt.sum(0)  # (K,)
    return cat_numerators, denom


def _ref_cat_centroids(cat_numerator, denom, M):
    """Naive implementation of update_categorical_centroids_: normalise + log-refresh."""
    tiny = _tiny(cat_numerator.dtype)
    K = denom.shape[0]
    C_cat = torch.zeros(K, M, dtype=cat_numerator.dtype)
    logC = torch.zeros(K, M, dtype=cat_numerator.dtype)
    for k in range(K):
        if float(denom[k]) > _tiny(cat_numerator.dtype):
            for m in range(M):
                C_cat[k, m] = float(cat_numerator[m, k]) / float(denom[k])
        else:
            for m in range(M):
                C_cat[k, m] = 1.0 / M  # uniform empty fallback
        for m in range(M):
            logC[k, m] = math.log(max(float(C_cat[k, m]), tiny))
    return C_cat, logC


def _buffers(T, D, K):
    return {
        "Wd_cost_slice": empty(D),
        "centroid_numerator": empty(K, D),
        "scratch_T": empty(T),
        "scratch_KD": empty(K, D),
        "scratch_D": empty(D),
    }


class TestComputeWdCostEuclidean:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, D, K = shape
        X, C, gamma_wt, denom, X_sq = _make(T, D, K, seed=seed)
        C_input = C.clone()
        buf = _buffers(T, D, K)
        assert_zero_alloc(
            compute_wd_cost_euclidean_,
            buf["Wd_cost_slice"],
            X,
            C,
            gamma_wt,
            denom,
            X_sq,
            buf["centroid_numerator"],
            buf["scratch_T"],
            buf["scratch_KD"],
            buf["scratch_D"],
        )
        b_ref, numer_ref = _ref(X, C_input, gamma_wt)
        assert_close(C, C_input)  # C read-only (not mutated)
        assert_close(buf["centroid_numerator"], numer_ref)
        assert_close(buf["Wd_cost_slice"], b_ref)

    def test_frozen_feature_weights_still_stage_the_complete_numerator(self):
        T, D, K = SHAPES[0]
        X, C, gamma_wt, denom, X_sq = _make(T, D, K, seed=0)
        buf = _buffers(T, D, K)
        sentinel = torch.full((D,), -7.5, dtype=DTYPE, device=DEVICE)
        buf["Wd_cost_slice"].copy_(sentinel)

        assert_zero_alloc(
            compute_wd_cost_euclidean_,
            buf["Wd_cost_slice"],
            X,
            C,
            gamma_wt,
            denom,
            X_sq,
            buf["centroid_numerator"],
            buf["scratch_T"],
            buf["scratch_KD"],
            buf["scratch_D"],
            need_cost=False,
        )

        _, numer_ref = _ref(X, C, gamma_wt)
        assert_close(buf["centroid_numerator"], numer_ref)
        torch.testing.assert_close(buf["Wd_cost_slice"], sentinel, rtol=0, atol=0)


def _code_scratch(T, max_M, K_max):
    """Return NaN-filled workspaces for summing codes in a fixed order."""
    return (
        torch.full((T, max_M), torch.nan, dtype=DTYPE, device=DEVICE),
        torch.full((_ONE_HOT_BATCHES * max_M * K_max,), torch.nan, dtype=DTYPE, device=DEVICE),
    )


def _make_wd_cat(T, K, M_list, *, seed):
    """Build a per-feature X_cat list alternating int-code features and distributions.

    Features 0, 2, … are ``(T,)`` long tensors of integer codes; the others are
    ``(T, M_d)`` row-normalised simplex distributions. Also builds gamma_wt,
    the per-feature logC_cat_list, combined_scales, the output buffers, and the
    ``(K, max M_d)`` Frobenius-product scratch.
    """
    g = generator(seed)
    gamma = stochastic(T, K, dim=1, g=g)
    Wt = rand(T, g=g)
    gamma_wt = gamma * Wt[:, None]  # Γ ∘ Wt
    X_cat = []
    logC_cat_list = []
    cat_numerator_list = []
    for i, M in enumerate(M_list):
        if i % 2 == 0:  # int-code feature
            X_cat.append(torch.randint(0, M, (T,), generator=g, device=DEVICE))  # (T,) long
        else:  # distribution feature
            X_cat.append(stochastic(T, M, dim=1, g=g))  # row-stochastic simplex
        C = stochastic(K, M, dim=1, g=g)  # distribution over categories
        logC_cat_list.append(C.log())
        cat_numerator_list.append(empty(M, K))
    D_cat = len(M_list)
    combined_scales = rand(D_cat, g=g)
    Wd_cost_slice_cat = empty(D_cat)
    scratch_KM = empty(K, max(M_list))
    return (
        Wd_cost_slice_cat,
        X_cat,
        gamma_wt,
        logC_cat_list,
        combined_scales,
        cat_numerator_list,
        scratch_KM,
    )


def _ref_wd_cat(X_cat, gamma_wt, logC_cat_list, combined_scales):
    """Naive-loop oracle for (Wd_cost_slice_cat, cat_numerator_list)."""
    T, K = gamma_wt.shape
    D_cat = len(X_cat)
    Wd_cost_slice_cat = torch.zeros(D_cat, dtype=gamma_wt.dtype)
    cat_numerator_list = []
    for i, (X_i, logC_i) in enumerate(zip(X_cat, logC_cat_list, strict=True)):
        M = logC_i.shape[1]
        numer = torch.zeros(M, K, dtype=gamma_wt.dtype)
        if X_i.dim() == 1:  # int codes: scatter-add
            for t in range(T):
                code = int(X_i[t])
                for k in range(K):
                    numer[code, k] += float(gamma_wt[t, k])
        else:  # distribution
            for m in range(M):
                for k in range(K):
                    numer[m, k] = sum(float(X_i[t, m]) * float(gamma_wt[t, k]) for t in range(T))
        cat_numerator_list.append(numer)
        acc = sum(float(logC_i[k, m]) * float(numer[m, k]) for k in range(K) for m in range(M))
        Wd_cost_slice_cat[i] = -float(combined_scales[i]) * acc
    return Wd_cost_slice_cat, cat_numerator_list


class TestComputeWdCostCategorical:
    @pytest.mark.parametrize("shape", WD_CAT_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, K, M_list = shape
        (
            Wd_cost_slice_cat,
            X_cat,
            gamma_wt,
            logC_cat_list,
            combined_scales,
            cat_numerator_list,
            scratch_KM,
        ) = _make_wd_cat(T, K, M_list, seed=seed)
        assert_zero_alloc(
            compute_wd_cost_categorical_,
            Wd_cost_slice_cat,
            X_cat,
            gamma_wt,
            logC_cat_list,
            combined_scales,
            cat_numerator_list,
            scratch_KM,
        )
        slice_ref, numer_ref = _ref_wd_cat(X_cat, gamma_wt, logC_cat_list, combined_scales)
        assert_close(Wd_cost_slice_cat, slice_ref)
        for num_i, num_ref_i in zip(cat_numerator_list, numer_ref, strict=True):
            assert_close(num_i, num_ref_i)

    # One-hot rows in row batches: fewer rows than batches, whole batches and a remainder,
    # and whole batches only
    @pytest.mark.parametrize("shape", [WD_CAT_SHAPES[1], (600, 3, [4, 2]), (512, 4, [3, 5])])
    def test_codes_summed_in_a_fixed_order_match_the_reference(self, shape):
        T, K, M_list = shape
        buffers = list(_make_wd_cat(T, K, M_list, seed=SEEDS[0]))
        code_scratch = _code_scratch(T, max(M_list), K)

        assert_zero_alloc(compute_wd_cost_categorical_, *buffers, code_scratch=code_scratch)

        slice_ref, numer_ref = _ref_wd_cat(buffers[1], buffers[2], buffers[3], buffers[4])
        assert_close(buffers[0], slice_ref)
        for num_i, num_ref_i in zip(buffers[5], numer_ref, strict=True):
            assert_close(num_i, num_ref_i)

    @pytest.mark.parametrize("summed_in_order", [False, True])
    def test_fills_the_leading_elements_of_larger_buffers(self, summed_in_order):
        buffers = list(_make_wd_cat(7, 3, [4, 2, 3], seed=0))
        original = buffers[5]
        numerators = [
            torch.full((value.shape[0], 5), torch.nan, dtype=DTYPE, device=DEVICE)
            for value in original
        ]
        buffers[5] = numerators
        buffers[6] = torch.full((5, 4), torch.nan, dtype=DTYPE, device=DEVICE)  # 5 > K rows
        code_scratch = _code_scratch(7, 4, 5) if summed_in_order else None

        assert_zero_alloc(compute_wd_cost_categorical_, *buffers, code_scratch=code_scratch)

        costs, expected = _ref_wd_cat(buffers[1], buffers[2], buffers[3], buffers[4])
        assert_close(buffers[0], costs)
        for actual, reference in zip(numerators, expected, strict=True):
            filled = reference.numel()
            assert_close(actual.view(-1)[:filled].view(reference.shape), reference)
            assert torch.isnan(actual.view(-1)[filled:]).all()

    @pytest.mark.parametrize("operand", ["numerator", "scratch_KM"])
    def test_a_strided_buffer_is_refused(self, operand):
        buffers = list(_make_wd_cat(7, 3, [4, 2], seed=0))
        if operand == "numerator":
            buffers[5] = [torch.empty(4, 6, dtype=DTYPE, device=DEVICE)[:, :5], buffers[5][1]]
            shape = r"shape \(4, 5\)"
        else:
            buffers[6] = torch.empty(3, 6, dtype=DTYPE, device=DEVICE)[:, :4]
            shape = r"shape \(3, 4\)"
        with pytest.raises(ValueError, match=rf"{shape} .* not contiguous"):
            compute_wd_cost_categorical_(*buffers)

    def test_frozen_feature_weights_still_stage_the_complete_numerators(self):
        T, K, M_list = WD_CAT_SHAPES[0]
        buffers = list(_make_wd_cat(T, K, M_list, seed=0))
        sentinel = torch.full((len(M_list),), -7.5, dtype=DTYPE, device=DEVICE)
        buffers[0].copy_(sentinel)

        assert_zero_alloc(compute_wd_cost_categorical_, *buffers, need_cost=False)

        _, numer_ref = _ref_wd_cat(buffers[1], buffers[2], buffers[3], buffers[4])
        for num_i, num_ref_i in zip(buffers[5], numer_ref, strict=True):
            assert_close(num_i, num_ref_i)
        torch.testing.assert_close(buffers[0], sentinel, rtol=0, atol=0)


class TestUpdateEuclideanCentroids:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        _T, D, K = shape
        g = generator(seed)
        numer = randn(K, D, g=g)
        denom = rand(K, g=g)  # all > 0
        C_out = empty(K, D)
        mask = torch.empty(K, dtype=torch.bool, device=DEVICE)
        assert_zero_alloc(update_euclidean_centroids_, C_out, numer, denom, mask)
        assert_close(C_out, _ref_update(numer, denom))

    def test_the_numerator_may_be_divided_in_place(self):
        numer = torch.tensor([[2.0, 4.0], [3.0, 9.0], [1.0, 1.0]], dtype=DTYPE, device=DEVICE)
        denom = torch.tensor([2.0, 3.0, 0.0], dtype=DTYPE, device=DEVICE)
        expected = torch.tensor([[1.0, 2.0], [1.0, 3.0], [0.0, 0.0]], dtype=DTYPE, device=DEVICE)
        mask = torch.empty(3, dtype=torch.bool, device=DEVICE)

        update_euclidean_centroids_(numer, numer, denom, mask)

        torch.testing.assert_close(numer, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 2.0])
    def test_tiny_mass_uses_the_fallback_without_changing_the_mass(self, scale):
        mass = _tiny(DTYPE) * scale
        denom = torch.tensor([mass, 0.0], dtype=DTYPE, device=DEVICE)
        numer = torch.tensor([[mass / 4, mass / 2], [0.0, 0.0]], dtype=DTYPE, device=DEVICE)
        out = empty(2, 2)
        before = denom.clone()
        mask = torch.empty(2, dtype=torch.bool, device=DEVICE)

        assert_zero_alloc(update_euclidean_centroids_, out, numer, denom, mask)

        row = [0.0, 0.0] if scale <= 1 else [0.25, 0.5]
        expected = torch.tensor([row, [0.0, 0.0]], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert torch.equal(denom, before)


class TestComputeCentroidInputs:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, _D, K = shape
        g = generator(seed)
        gamma = stochastic(T, K, dim=1, g=g)
        Wt = rand(T, g=g)
        gamma_wt = empty(T, K)
        denom = empty(K)
        assert_zero_alloc(compute_gamma_wt_, gamma_wt, denom, gamma, Wt)
        gw_ref, denom_ref = _ref_inputs(gamma, Wt)
        assert_close(gamma_wt, gw_ref)
        assert_close(denom, denom_ref)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_none_weights_are_uniform(self, shape, seed):
        # Wt=None is the uniform fast-path: gamma_wt = Γ, denom = Σ_t Γ.
        T, _D, K = shape
        gamma = stochastic(T, K, dim=1, g=generator(seed))
        gamma_wt = empty(T, K)
        denom = empty(K)
        compute_gamma_wt_(gamma_wt, denom, gamma, None)
        assert_close(gamma_wt, gamma)
        assert_close(denom, gamma.sum(0))


class TestUpdateCategoricalCentroids:
    @pytest.mark.parametrize("shape", CAT_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, K, M_list = shape
        cat_numerators, denom = _make_cat(T, K, M_list, seed=seed)
        C_cat = tuple(empty(K, M) for M in M_list)
        logC = tuple(empty(K, M) for M in M_list)
        mask = torch.empty(K, dtype=torch.bool, device=DEVICE)
        assert_zero_alloc(update_categorical_centroids_, C_cat, logC, cat_numerators, denom, mask)
        for C_i, logC_i, numerator_i, M in zip(C_cat, logC, cat_numerators, M_list, strict=True):
            C_ref, logC_ref = _ref_cat_centroids(numerator_i, denom, M)
            assert_close(C_i, C_ref)
            assert_close(logC_i, logC_ref)
            assert_close(C_i.sum(1), torch.ones(K, dtype=DTYPE))  # category distributions

    @pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 2.0])
    def test_tiny_mass_uses_a_uniform_level_distribution(self, scale):
        mass = _tiny(DTYPE) * scale
        denom = torch.tensor([mass, 0.0], dtype=DTYPE, device=DEVICE)
        numerator = torch.tensor([[mass / 4, 0.0], [3 * mass / 4, 0.0]], dtype=DTYPE, device=DEVICE)
        C, logs = (empty(2, 2),), (empty(2, 2),)
        mask = torch.empty(2, dtype=torch.bool, device=DEVICE)

        assert_zero_alloc(update_categorical_centroids_, C, logs, (numerator,), denom, mask)

        row = [0.5, 0.5] if scale <= 1 else [0.25, 0.75]
        expected = torch.tensor([row, [0.5, 0.5]], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(C[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(logs[0], expected.log())
