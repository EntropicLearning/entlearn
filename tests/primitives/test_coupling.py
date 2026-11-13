"""Tests for the cross-block coupling primitives (coupling.py)."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.coupling import (
    accumulate_coupling_,
    transition_partial_loss_,
)
from entlearn.primitives.normalise import _tiny, floored_log_
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn, stochastic

SHAPES = [(7, 3, 4), (10, 4, 2), (6, 2, 5), (5, 5, 3)]  # (T, K_target, K_source)
DELTA_EFF = 0.37


def _make_case(T, Kt, Ks, *, seed, weighted):
    """Seeded log transition matrix, both endpoints' affiliations and optional weights."""
    g = generator(seed)
    theta = stochastic(Kt, Ks, dim=0, g=g)  # left-stochastic columns
    log_theta = theta.clamp_min(_tiny(theta.dtype)).log()
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
    def test_zero_probabilities_and_inactive_weights_stay_finite(self):
        theta = torch.eye(2, dtype=DTYPE, device=DEVICE)
        logs = torch.empty_like(theta)
        floored_log_(logs, theta)
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=DTYPE, device=DEVICE)
        source = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=DTYPE, device=DEVICE)
        weights = torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE)
        cost = torch.zeros_like(source)
        loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
        scratch, scalar = torch.empty_like(source), empty()

        assert_zero_alloc(accumulate_coupling_, cost, logs, target, 1.0, weights, scratch)
        assert_zero_alloc(
            transition_partial_loss_, loss, logs, target, source, 1.0, scratch, scalar, weights
        )

        # One active row assigns positive weight to p = 0: the fallback prices it finitely.
        # The zero sample weight and zero affiliations contribute exactly zero.
        mismatch = -math.log(_tiny(DTYPE))
        expected = torch.tensor(
            [[0.0, mismatch], [0.0, mismatch], [0.0, 0.0]], dtype=DTYPE, device=DEVICE
        )
        assert torch.isfinite(cost).all() and torch.isfinite(loss)
        torch.testing.assert_close(cost, expected, rtol=4 * torch.finfo(DTYPE).eps, atol=0)
        assert float(loss) == pytest.approx(mismatch, rel=4 * torch.finfo(DTYPE).eps, abs=0)

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
