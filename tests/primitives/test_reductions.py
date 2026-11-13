"""Tests for the reductions over the rows (reductions.py)."""

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.reductions import (
    _CHUNK_ROWS,
    _CUDA_MIN_CHUNKS,
    _chunked_product_,
    compute_wt_cost_,
    reduce_input_cost_,
)
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn, stochastic

REDUCE_SHAPES = [(7, 3), (10, 4), (6, 2), (5, 5)]  # (T, K)
WT_COST_SHAPES = [(7, 3), (10, 4), (6, 2), (5, 5)]  # (T, K)
WT_COST_MODES = ["both", "cont_only", "cat_only"]  # which of sqdist_wd / cat_cost present


def _make_reduce(T, K, *, seed):
    """Build (disc_cost, gamma) for the input-cost reduction.

    ``disc_cost`` is an arbitrary ``(T, K)`` cost tensor; ``gamma`` is a ``(T, K)``
    row-stochastic affiliation tensor.
    """
    g = generator(seed)
    disc_cost = randn(T, K, g=g)
    gamma = stochastic(T, K, dim=1, g=g)
    return disc_cost, gamma


def _ref_reduce(disc_cost, gamma):
    """Naive-loop implementation for the added term Σ_{t,k} gamma·disc_cost."""
    T, K = disc_cost.shape
    return sum(float(gamma[t, k]) * float(disc_cost[t, k]) for t in range(T) for k in range(K))


class TestReduceInputCost:
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, K = shape
        disc_cost, gamma = _make_reduce(T, K, seed=seed)
        out_scalar = empty()
        out_scalar.fill_(0.5)  # pre-populated accumulator
        before = float(out_scalar)
        scratch_TK = empty(T, K)
        scratch_scalar = empty()
        assert_zero_alloc(
            reduce_input_cost_, out_scalar, disc_cost, gamma, scratch_TK, scratch_scalar
        )
        expected = before + _ref_reduce(disc_cost, gamma)
        assert_close(out_scalar, torch.tensor(expected, dtype=DTYPE))


def _make_wt_cost(T, K, mode, *, seed):
    """Build (gamma, sqdist_wd, cat_cost, Wt_cost, scratch_TK) for the Wt-cost composition.

    ``gamma`` is a ``(T, K)`` row-stochastic affiliation tensor. ``sqdist_wd``
    and ``cat_cost`` are arbitrary non-negative ``(T, K)`` cost caches; ``mode``
    selects which are present (the other is ``None``), exercising the
    ``D_cont == 0`` / ``D_cat == 0`` branches.
    """
    g = generator(seed)
    gamma = stochastic(T, K, dim=1, g=g)
    sqdist_wd = rand(T, K, g=g) if mode != "cat_only" else None
    cat_cost = rand(T, K, g=g) if mode != "cont_only" else None
    Wt_cost = empty(T)
    scratch_TK = empty(T, K)
    return gamma, sqdist_wd, cat_cost, Wt_cost, scratch_TK


def _ref_wt_cost(gamma, sqdist_wd, cat_cost):
    """Naive-loop oracle for Wt_cost[t] = Σ_k gamma·(sqdist_wd + cat_cost)."""
    T, K = gamma.shape
    out = torch.zeros(T, dtype=gamma.dtype)
    for t in range(T):
        s = 0.0
        for k in range(K):
            d = 0.0
            if sqdist_wd is not None:
                d += float(sqdist_wd[t, k])
            if cat_cost is not None:
                d += float(cat_cost[t, k])
            s += float(gamma[t, k]) * d
        out[t] = s
    return out


class TestComputeWtCost:
    @pytest.mark.parametrize("mode", WT_COST_MODES)
    @pytest.mark.parametrize("shape", WT_COST_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, mode, seed):
        T, K = shape
        gamma, sqdist_wd, cat_cost, Wt_cost, scratch_TK = _make_wt_cost(T, K, mode, seed=seed)
        assert_zero_alloc(compute_wt_cost_, Wt_cost, gamma, sqdist_wd, cat_cost, scratch_TK)
        ref = _ref_wt_cost(gamma, sqdist_wd, cat_cost)
        assert_close(Wt_cost, ref)


class TestChunkedProduct:
    @pytest.mark.parametrize("rows", [_CHUNK_ROWS - 1, 3 * _CHUNK_ROWS + 5])
    def test_matches_the_product_and_allocates_nothing(self, rows):
        g = generator(SEEDS[0])
        A, B = rand(rows, 3, g=g), stochastic(rows, 4, dim=1, g=g)
        out, scratch = empty(3, 4), empty(rows * 12)
        assert_zero_alloc(_chunked_product_, out, A, B, scratch)
        assert_close(out, A.T @ B)

    def test_a_scratch_too_small_for_the_chunks_takes_one_product(self):
        g = generator(SEEDS[0])
        A, B = rand(3 * _CHUNK_ROWS, 3, g=g), stochastic(3 * _CHUNK_ROWS, 4, dim=1, g=g)
        out = empty(3, 4)
        _chunked_product_(out, A, B, empty(2 * 12))  # room for two of the three chunk products
        assert_close(out, A.T @ B)

    @pytest.mark.parametrize("chunks", [3, _CUDA_MIN_CHUNKS])
    def test_cuda_chunks_only_float32_products_of_many_chunks(self, chunks):
        g = generator(SEEDS[0])
        A, B = rand(chunks * _CHUNK_ROWS, 3, g=g), stochastic(chunks * _CHUNK_ROWS, 4, dim=1, g=g)
        scratch = torch.full((chunks * 12,), torch.nan, dtype=DTYPE, device=DEVICE)
        _chunked_product_(empty(3, 4), A, B, scratch)
        one = DEVICE.type == "cuda" and (DTYPE is torch.float64 or chunks < _CUDA_MIN_CHUNKS)
        assert bool(scratch.isnan().all()) == one

    def test_float32_error_stays_near_that_of_one_chunk(self):
        # One float32 product over these 400,000 rows is off by about 300 eps here; the
        # pairwise sum of the chunk products adds only seven levels to one chunk's error.
        g = torch.Generator(device=DEVICE).manual_seed(SEEDS[0])
        A = torch.rand(400_000, 3, generator=g, dtype=torch.float64, device=DEVICE)
        B = torch.rand(400_000, 4, generator=g, dtype=torch.float64, device=DEVICE)
        B /= B.sum(1, keepdim=True)
        out = torch.empty(3, 4, dtype=torch.float32, device=DEVICE)
        scratch = torch.empty(400_000 * 12, dtype=torch.float32, device=DEVICE)
        _chunked_product_(out, A.float(), B.float(), scratch)
        reference = A.T @ B
        error = ((out.double() - reference).abs() / reference).max()
        assert error <= 64 * torch.finfo(torch.float32).eps
