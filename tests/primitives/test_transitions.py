"""Tests for the transition update step (transitions.py)."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc

from entlearn.primitives.transitions import update_theta_
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, stochastic

SHAPES = [(7, 3, 4), (10, 4, 2), (6, 2, 5), (5, 5, 3)]  # (T, K_target, K_source)


def _make(T, Kt, Ks, *, seed, weighted):
    g = generator(seed)
    gamma_target = stochastic(T, Kt, dim=1, g=g)
    gamma_source = stochastic(T, Ks, dim=1, g=g)
    sw = rand(T, g=g) if weighted else None
    return gamma_target, gamma_source, sw


def _ref(gamma_target, gamma_source, sw):
    """Naive implementation of update_theta_: weighted co-occurrence, lstoch, log."""
    eps = torch.finfo(gamma_target.dtype).eps
    T, Kt = gamma_target.shape
    Ks = gamma_source.shape[1]
    theta = torch.zeros(Kt, Ks, dtype=gamma_target.dtype)
    for kt in range(Kt):
        for ks in range(Ks):
            s = 0.0
            for t in range(T):
                w = 1.0 if sw is None else float(sw[t])
                s += float(gamma_target[t, kt]) * w * float(gamma_source[t, ks])
            theta[kt, ks] = s
    for ks in range(Ks):  # left-stochastic columns
        col = sum(float(theta[kt, ks]) for kt in range(Kt))
        for kt in range(Kt):
            theta[kt, ks] = (float(theta[kt, ks]) / col) if col > 0.0 else 1.0 / Kt
    log_theta = torch.zeros(Kt, Ks, dtype=gamma_target.dtype)
    for kt in range(Kt):
        for ks in range(Ks):
            log_theta[kt, ks] = math.log(max(float(theta[kt, ks]), eps))
    return theta, log_theta


class TestUpdateTheta:
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed, weighted):
        T, Kt, Ks = shape
        gamma_target, gamma_source, sw = _make(T, Kt, Ks, seed=seed, weighted=weighted)
        theta = empty(Kt, Ks)
        log_theta = empty(Kt, Ks)
        scratch_K_source = empty(Ks)
        scratch_TK_source = empty(T, Ks) if weighted else None
        assert_zero_alloc(
            update_theta_,
            theta,
            log_theta,
            gamma_target,
            gamma_source,
            scratch_K_source,
            sample_weights=sw,
            scratch_TK_source=scratch_TK_source,
        )
        theta_ref, log_ref = _ref(gamma_target, gamma_source, sw)
        assert_close(theta, theta_ref)
        assert_close(log_theta, log_ref)
        assert_close(theta.sum(0), torch.ones(Ks, dtype=theta.dtype))  # left-stochastic

    @pytest.mark.parametrize("norm_dim", [0, 1])
    def test_pseudocount_and_norm_dim_match_reference(self, norm_dim):
        T, Kt, Ks = 7, 3, 4
        pseudocount = 0.7
        gamma_target, gamma_source, _ = _make(T, Kt, Ks, seed=SEEDS[0], weighted=False)
        theta = empty(Kt, Ks)
        log_theta = empty(Kt, Ks)
        scratch_norm = empty(Ks if norm_dim == 0 else Kt)
        update_theta_(
            theta,
            log_theta,
            gamma_target,
            gamma_source,
            scratch_norm,
            pseudocount=pseudocount,
            norm_dim=norm_dim,
        )
        counts = torch.zeros(Kt, Ks, dtype=theta.dtype)
        for kt in range(Kt):
            for ks in range(Ks):
                s = pseudocount
                for t in range(T):
                    s += float(gamma_target[t, kt]) * float(gamma_source[t, ks])
                counts[kt, ks] = s
        ref = counts / counts.sum(norm_dim, keepdim=True)  # every slice has positive mass
        eps = torch.finfo(theta.dtype).eps
        assert_close(theta, ref)
        assert_close(log_theta, torch.log(torch.clamp_min(ref, eps)))
        assert_close(
            theta.sum(norm_dim),
            torch.ones(Kt if norm_dim == 1 else Ks, dtype=theta.dtype),
        )
