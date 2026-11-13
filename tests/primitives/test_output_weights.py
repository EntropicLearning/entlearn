"""Tests for output-weight and output-gate primitives."""

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.output import apply_output_gate_, update_output_weights_
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, stochastic

SHAPES = [(7, 3, 4), (10, 2, 5), (6, 5, 2), (5, 4, 3)]  # (T, M, K_source)
DELTA = 0.5


def _make_wm(T, M, K, *, seed):
    g = generator(seed)
    Y = rand(T, M, g=g)
    Cy = rand(M, K, g=g)
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    eow *= rand(T, g=g) > 0.25  # mostly labelled; unlabelled rows have zero weight
    return Y, Cy, gamma_source, eow


def _ref_wm_cost(Y, Cy, gamma_source, eow, delta):
    """Naive per-dimension residual masses q_m = delta·Σ_t ω̃·Σ_k Γ·(Y-Cy)²."""
    T, M = Y.shape
    K = Cy.shape[1]
    q = torch.zeros(M, dtype=Y.dtype, device=Y.device)
    for t in range(T):
        for k in range(K):
            w = float(eow[t]) * float(gamma_source[t, k])
            for m in range(M):
                q[m] += w * (float(Y[t, m]) - float(Cy[m, k])) ** 2
    return delta * q


def _run_update(T, M, K, *, seed, delta, epsilon_M, zero_alloc=False):
    """Update a uniform ``Wm`` on seeded data; return ``Wm``, ``Wm_cost`` and the reference."""
    Y, Cy, gamma_source, eow = _make_wm(T, M, K, seed=seed)
    Wm = torch.full((M,), 1.0 / M, dtype=DTYPE, device=DEVICE)
    Wm_cost = empty(M)
    gated = eow.unsqueeze(1) * gamma_source
    counts = (Y.T @ gated, gated.sum(0), (Y * Y).T @ eow)  # numerator, masses, moments
    scratch = (empty(M, K), empty(1), torch.empty(1, dtype=torch.int64, device=DEVICE))
    args = (Wm, Cy, *counts, delta, epsilon_M, Wm_cost, *scratch)
    if zero_alloc:
        assert_zero_alloc(update_output_weights_, *args)
    else:
        update_output_weights_(*args)
    return Wm, Wm_cost, _ref_wm_cost(Y, Cy, gamma_source, eow, delta)


class TestUpdateOutputWeights:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference_softmax(self, shape, seed):
        epsilon_M = 0.05
        Wm, Wm_cost, q = _run_update(
            *shape, seed=seed, delta=DELTA, epsilon_M=epsilon_M, zero_alloc=True
        )
        assert_close(Wm_cost, q)
        assert_close(Wm, torch.softmax(-q / epsilon_M, dim=0))

    def test_hard_regime_one_hot(self):
        Wm, _, q = _run_update(8, 3, 4, seed=2, delta=DELTA, epsilon_M=0.0)
        assert int(Wm.argmax()) == int(q.argmin())
        assert float(Wm.sum()) == 1.0
        assert int((Wm > 0).sum()) == 1


class TestApplyOutputGate:
    @pytest.mark.parametrize("aliased", [False, True], ids=["separate", "aliased"])
    def test_scales_rows_by_weight(self, aliased):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=DTYPE, device=DEVICE)
        # The middle row is unlabelled, so its weight is zero.
        eow = torch.tensor([0.5, 0.0, 0.25], dtype=DTYPE, device=DEVICE)
        out = source if aliased else torch.empty_like(source)
        result = apply_output_gate_(out, source, eow)
        expected = torch.tensor([[0.5, 1.0], [0.0, 0.0], [1.25, 1.5]], dtype=DTYPE, device=DEVICE)
        assert result is out
        assert torch.equal(out, expected)

    def test_without_weights_returns_the_source_and_leaves_out_untouched(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE, device=DEVICE)
        out = torch.full_like(source, 7.0)
        result = apply_output_gate_(out, source, None)
        assert result is source
        assert torch.equal(out, torch.full_like(source, 7.0))

    def test_inputs_read_only_and_zero_alloc(self):
        g = generator(SEEDS[0])
        src = rand(9, 4, g=g)
        eow = rand(9, g=g)
        out = torch.empty_like(src)
        src_c, eow_c = src.clone(), eow.clone()
        assert_zero_alloc(apply_output_gate_, out, src, eow)
        assert torch.equal(src, src_c) and torch.equal(eow, eow_c)
