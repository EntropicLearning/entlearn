"""Tests for the affiliation cost reductions (reductions.py)."""

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DTYPE

from entlearn.primitives.reductions import compute_wt_cost_, reduce_input_cost_
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
