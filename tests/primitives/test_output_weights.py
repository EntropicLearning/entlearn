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
    labelled = rand(T, g=g) > 0.25  # mostly labelled, some not
    return Y, Cy, gamma_source, eow, labelled


def _ref_wm_cost(Y, Cy, gamma_source, eow, labelled, delta):
    """Naive per-dimension residual masses q_m = delta·Σ_t ω̃·Σ_k Γ·(Y-Cy)²."""
    T, M = Y.shape
    K = Cy.shape[1]
    q = torch.zeros(M, dtype=Y.dtype, device=Y.device)
    for t in range(T):
        if not bool(labelled[t]):
            continue
        for k in range(K):
            w = float(eow[t]) * float(gamma_source[t, k])
            for m in range(M):
                q[m] += w * (float(Y[t, m]) - float(Cy[m, k])) ** 2
    return delta * q


def _run_update(T, M, K, *, seed, delta, epsilon_M, zero_alloc=False):
    """Update a uniform ``Wm`` on seeded data; return ``Wm``, ``Wm_cost`` and the reference."""
    Y, Cy, gamma_source, eow, labelled = _make_wm(T, M, K, seed=seed)
    Wm = torch.full((M,), 1.0 / M, dtype=DTYPE, device=DEVICE)
    Wm_cost = empty(M)
    scratch = (
        empty(T, K),  # masked_TK
        empty(M, K),  # scratch_MK
        empty(K),  # scratch_K
        empty(T),  # scratch_T
        empty(1),  # scratch_keepdim
        torch.empty(1, dtype=torch.int64, device=DEVICE),  # scratch_idx
        empty(K, M),  # scratch_KM
        empty(M),  # scratch_M
    )
    args = (Wm, Y, Y * Y, Cy, gamma_source, eow, labelled, delta, epsilon_M, Wm_cost, *scratch)
    if zero_alloc:
        assert_zero_alloc(update_output_weights_, *args)
    else:
        update_output_weights_(*args)
    return Wm, Wm_cost, _ref_wm_cost(Y, Cy, gamma_source, eow, labelled, delta)


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

    def test_simplex_invariant(self):
        Wm, _, _ = _run_update(9, 4, 3, seed=1, delta=1.3, epsilon_M=0.2)
        assert bool((Wm >= 0).all())
        assert_close(Wm.sum(), torch.tensor(1.0, dtype=DTYPE, device=DEVICE))

    def test_hard_regime_one_hot(self):
        Wm, _, q = _run_update(8, 3, 4, seed=2, delta=DELTA, epsilon_M=0.0)
        assert int(Wm.argmax()) == int(q.argmin())
        assert float(Wm.sum()) == 1.0
        assert int((Wm > 0).sum()) == 1

    def test_delta_zero_gives_uniform(self):
        M = 3
        Wm, _, _ = _run_update(8, M, 4, seed=3, delta=0.0, epsilon_M=0.1)
        assert_close(Wm, torch.full((M,), 1.0 / M, dtype=DTYPE, device=DEVICE))


class TestApplyOutputGate:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_inline_prologue_bitwise(self, seed):
        g = generator(seed)
        src = rand(9, 4, g=g)
        eow = rand(9, g=g)
        lab = rand(9, g=g) > 0.4
        out = torch.empty_like(src)
        expected = torch.empty_like(src)
        apply_output_gate_(out, src, eow, lab)
        torch.mul(src, eow.unsqueeze(1), out=expected)
        expected.mul_(lab.unsqueeze(1))
        assert torch.equal(out, expected)

    def test_aliased_in_place_form(self):
        g = generator(SEEDS[0])
        buf = rand(9, 4, g=g)
        eow = rand(9, g=g)
        lab = rand(9, g=g) > 0.4
        expected = buf.clone()
        expected.mul_(eow.unsqueeze(1))
        expected.mul_(lab.unsqueeze(1))
        apply_output_gate_(buf, buf, eow, lab)  # out aliases source
        assert torch.equal(buf, expected)

    def test_inputs_read_only_and_zero_alloc(self):
        g = generator(SEEDS[0])
        src = rand(9, 4, g=g)
        eow = rand(9, g=g)
        lab = rand(9, g=g) > 0.4
        out = torch.empty_like(src)
        src_c, eow_c, lab_c = src.clone(), eow.clone(), lab.clone()
        assert_zero_alloc(apply_output_gate_, out, src, eow, lab)
        assert torch.equal(src, src_c) and torch.equal(eow, eow_c) and torch.equal(lab, lab_c)
