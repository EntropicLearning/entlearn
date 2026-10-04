"""Tests for the cross-block coupling primitives (coupling.py)."""

import pytest
import torch
from _alloc import assert_zero_alloc

from entlearn.primitives.coupling import (
    accumulate_coupling_,
    transition_partial_loss_,
)
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn, stochastic

SHAPES = [(7, 3, 4), (10, 4, 2), (6, 2, 5), (5, 5, 3)]  # (T, K_target, K_source)
DELTA_EFF = 0.37


def _make_case(T, Kt, Ks, *, seed, weighted):
    """Seeded log transition matrix, both endpoints' affiliations and optional weights."""
    g = generator(seed)
    theta = stochastic(Kt, Ks, dim=0, g=g)  # left-stochastic columns
    log_theta = theta.clamp_min(torch.finfo(theta.dtype).eps).log()
    gamma_target = stochastic(T, Kt, dim=1, g=g)
    gamma_source = stochastic(T, Ks, dim=1, g=g)
    sw = rand(T, g=g) if weighted else None
    return log_theta, gamma_target, gamma_source, sw


def _ref_accumulate(cost, log_theta, gamma, delta_eff, sw):
    """Naive implementation of accumulate_coupling_, given the already-oriented log_theta."""
    T, K_own = cost.shape
    K = gamma.shape[1]
    out = cost.clone()
    for t in range(T):
        w = 1.0 if sw is None else float(sw[t])
        for j in range(K_own):
            acc = sum(float(gamma[t, k]) * float(log_theta[k, j]) for k in range(K))
            out[t, j] += -delta_eff * w * acc
    return out


def _ref_loss(log_theta, gamma_target, gamma_source, delta_eff, sw):
    """Naive implementation of transition_partial_loss_ (the added contribution)."""
    T, Kt = gamma_target.shape
    Ks = gamma_source.shape[1]
    acc = 0.0
    for t in range(T):
        w = 1.0 if sw is None else float(sw[t])
        for kt in range(Kt):
            for ks in range(Ks):
                acc += (
                    w
                    * float(gamma_target[t, kt])
                    * float(gamma_source[t, ks])
                    * float(log_theta[kt, ks])
                )
    return -delta_eff * acc


class TestAccumulateCoupling:
    """``accumulate_coupling_`` from either end. The target side passes the transposed
    log_theta and the source's Γ; the source side the untransposed one and the target's Γ."""

    @pytest.mark.parametrize("side", ["target", "source"])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed, weighted, side):
        T, Kt, Ks = shape
        log_theta, gamma_target, gamma_source, sw = _make_case(
            T, Kt, Ks, seed=seed, weighted=weighted
        )
        if side == "target":
            oriented, gamma = log_theta.transpose(0, 1), gamma_source
        else:
            oriented, gamma = log_theta, gamma_target
        cost = randn(T, oriented.shape[1], g=generator(seed + 1))  # pre-populated accumulator
        ref = _ref_accumulate(cost, oriented, gamma, DELTA_EFF, sw)
        scratch = torch.empty_like(gamma) if weighted else None
        assert_zero_alloc(
            accumulate_coupling_,
            cost,
            oriented,
            gamma,
            DELTA_EFF,
            sample_weights=sw,
            scratch_TK=scratch,
        )
        assert_close(cost, ref)


class TestTransitionPartialLoss:
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed, weighted):
        T, Kt, Ks = shape
        log_theta, gamma_target, gamma_source, sw = _make_case(
            T, Kt, Ks, seed=seed, weighted=weighted
        )
        out_scalar = empty()
        out_scalar.fill_(0.5)  # pre-populated accumulator
        before = float(out_scalar)
        scratch = empty(T, Ks)
        scratch_scalar = empty()
        assert_zero_alloc(
            transition_partial_loss_,
            out_scalar,
            log_theta,
            gamma_target,
            gamma_source,
            DELTA_EFF,
            sample_weights=sw,
            scratch_TK_source=scratch,
            scratch_scalar=scratch_scalar,
        )
        expected = before + _ref_loss(log_theta, gamma_target, gamma_source, DELTA_EFF, sw)
        assert_close(out_scalar, torch.tensor(expected, dtype=out_scalar.dtype))
