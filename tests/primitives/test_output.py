"""Tests for the output-block primitives (output.py)."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.output import (
    classification_output_accumulate_into_source_cost_,
    classification_output_loss_,
    geometric_probabilities_,
    propagate_classification_output_,
    propagate_regression_output_,
    regression_output_accumulate_into_source_cost_,
    regression_output_loss_,
    update_classification_output_,
    update_regression_output_,
)
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn, stochastic

SHAPES = [(7, 3, 4), (10, 2, 5), (6, 5, 2), (5, 4, 3)]  # (T, M, K_source)
DELTA = 0.5
# (delta, epsilon_P, expected) for logits whose two entries differ by log 3. The gate
# ignores delta; each soft case holds epsilon_P fixed so that delta alone sets the
# combined inverse temperature beta = delta / epsilon_P, giving 3^-beta / (1 + 3^-beta).
READOUT_CASES = [
    (1.0, torch.finfo(DTYPE).eps, [1.0, 0.0]),
    (1e-3, 1e-3, [0.75, 0.25]),
    (3e-3, 1e-3, [27 / 28, 1 / 28]),
]


class TestGeometricReadout:
    @pytest.mark.parametrize(("delta", "epsilon", "expected_row"), READOUT_CASES)
    def test_centred_readout_preserves_the_gate_and_reuses_scratch(
        self, delta, epsilon, expected_row
    ):
        logits = torch.tensor([[-10.0, -10.0 - math.log(3)]], dtype=DTYPE, device=DEVICE)
        before = logits.clone()
        out = torch.empty_like(logits)
        scratch = torch.empty(1, 1, dtype=DTYPE, device=DEVICE)
        indices = torch.empty(1, 1, dtype=torch.int64, device=DEVICE)
        assert_zero_alloc(geometric_probabilities_, out, logits, delta, epsilon, scratch, indices)
        expected = torch.tensor([expected_row], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(logits, before, rtol=0, atol=0)
        geometric_probabilities_(logits, logits, delta, epsilon, scratch, indices)
        torch.testing.assert_close(logits, expected)

    def test_extreme_soft_inverse_temperature_keeps_tied_maxima_soft(self):
        logits = torch.tensor([[-1.0, -1.0, -2.0]], dtype=DTYPE, device=DEVICE)
        scratch = torch.empty(1, 1, dtype=DTYPE, device=DEVICE)
        indices = torch.empty(1, 1, dtype=torch.int64, device=DEVICE)
        geometric_probabilities_(logits, logits, 1e308, 1e-3, scratch, indices)
        torch.testing.assert_close(
            logits, torch.tensor([[0.5, 0.5, 0.0]], dtype=DTYPE, device=DEVICE)
        )


def _make_clf(T, M, K, *, seed):
    g = generator(seed)
    Pi = stochastic(T, M, dim=1, g=g)  # simplex rows (distribution target)
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    labelled = rand(T, g=g) > 0.25  # mostly labelled, some not
    return Pi, gamma_source, eow, labelled


def _ref_clf(Pi, gamma_source, eow, labelled, M, norm_dim=0):
    """Naive implementation of update_classification_output_: gated numer/denom + log.

    ``norm_dim=0`` divides each column by the cluster's labelled mass (M coupling);
    ``norm_dim=1`` divides each row by the class's counted mass (S coupling). Same
    counts either way, and an empty slice takes the uniform over the axis normalised.
    """
    eps = torch.finfo(Pi.dtype).eps
    T = Pi.shape[0]
    Ks = gamma_source.shape[1]
    numer = torch.zeros(M, Ks, dtype=Pi.dtype)
    denom = torch.zeros(Ks, dtype=Pi.dtype)
    row_denom = torch.zeros(M, dtype=Pi.dtype)
    for t in range(T):
        if not bool(labelled[t]):
            continue
        for k in range(Ks):
            denom[k] += float(eow[t]) * float(gamma_source[t, k])
            for m in range(M):
                numer[m, k] += float(eow[t]) * float(Pi[t, m]) * float(gamma_source[t, k])
    for m in range(M):
        for k in range(Ks):
            row_denom[m] += float(numer[m, k])
    theta = torch.zeros(M, Ks, dtype=Pi.dtype)
    logt = torch.zeros(M, Ks, dtype=Pi.dtype)
    uniform = 1.0 / M if norm_dim == 0 else 1.0 / Ks
    for k in range(Ks):
        for m in range(M):
            d = float(denom[k]) if norm_dim == 0 else float(row_denom[m])
            theta[m, k] = float(numer[m, k]) / d if d > 0.0 else uniform
            logt[m, k] = math.log(max(float(theta[m, k]), eps))
    return theta, logt


def _make_reg(T, M, K, *, seed):
    g = generator(seed)
    Y = randn(T, M, g=g)  # regression target
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    labelled = rand(T, g=g) > 0.25  # mostly labelled, some not
    w = eow * labelled  # labelled-only weights
    Y_mean_weighted = (w.unsqueeze(1) * Y).sum(0) / w.sum()  # (M,) eow-weighted labelled mean
    return Y, gamma_source, eow, labelled, Y_mean_weighted


def _ref_reg(Y, gamma_source, eow, labelled, Y_mean_weighted):
    """Naive implementation of update_regression_output_: gated numer/denom, empty -> Y_mean."""
    T, M = Y.shape
    Ks = gamma_source.shape[1]
    numer = torch.zeros(M, Ks, dtype=Y.dtype)
    denom = torch.zeros(Ks, dtype=Y.dtype)
    for t in range(T):
        if not bool(labelled[t]):
            continue
        for k in range(Ks):
            denom[k] += float(eow[t]) * float(gamma_source[t, k])
            for m in range(M):
                numer[m, k] += float(eow[t]) * float(gamma_source[t, k]) * float(Y[t, m])
    Cy = torch.zeros(M, Ks, dtype=Y.dtype)
    for k in range(Ks):
        for m in range(M):
            if float(denom[k]) > 0.0:
                Cy[m, k] = float(numer[m, k]) / float(denom[k])
            else:
                Cy[m, k] = float(Y_mean_weighted[m])
    return Cy


def _reg_buffers(T, M, K):
    return (
        empty(M, K),  # Cy
        empty(K),  # denom_buf
        empty(M, K),  # scratch_MK
        empty(T, K),  # scratch_TK_source
    )


def _buffers(T, M, K):
    return (
        empty(M, K),  # theta_out
        empty(M, K),  # log_theta_out
        empty(K),  # scratch_K
        empty(T, K),  # scratch_TK_source
    )


def _make_clf_cost(T, M, K, *, seed):
    g = generator(seed)
    cost_source = randn(T, K, g=g)  # pre-populated accumulator
    theta_out = stochastic(M, K, dim=0, g=g)  # left-stochastic over m
    log_theta_out = theta_out.clamp_min(torch.finfo(theta_out.dtype).eps).log()
    Pi = stochastic(T, M, dim=1, g=g)  # simplex rows
    eow = rand(T, g=g)
    labelled = rand(T, g=g) > 0.25
    return cost_source, log_theta_out, Pi, eow, labelled


def _ref_clf_cost(cost_source, log_theta_out, Pi, delta, eow, labelled):
    """Naive implementation of classification_output_accumulate_into_source_cost_."""
    T, Ks = cost_source.shape
    M = Pi.shape[1]
    out = cost_source.clone()
    for t in range(T):
        if not bool(labelled[t]):
            continue
        for ks in range(Ks):
            acc = sum(float(Pi[t, m]) * float(log_theta_out[m, ks]) for m in range(M))
            out[t, ks] += -delta * float(eow[t]) * acc
    return out


def _make_reg_cost(T, M, K, *, seed):
    g = generator(seed)
    cost_source = randn(T, K, g=g)  # pre-populated accumulator
    Cy = randn(M, K, g=g)  # regression centroids
    Y = randn(T, M, g=g)
    target_sq = (Y * Y).sum(1)  # (T,) Σ_m Y²
    eow = rand(T, g=g)
    labelled = rand(T, g=g) > 0.25
    return cost_source, Cy, Y, target_sq, eow, labelled


def _ref_reg_cost(cost_source, Cy, Y, delta, eow, labelled):
    """Naive implementation of regression_output_accumulate_into_source_cost_."""
    T, Ks = cost_source.shape
    M = Y.shape[1]
    out = cost_source.clone()
    for t in range(T):
        if not bool(labelled[t]):
            continue
        for ks in range(Ks):
            resid = sum((float(Y[t, m]) - float(Cy[m, ks])) ** 2 for m in range(M))
            out[t, ks] += delta * float(eow[t]) * resid
    return out


def _make_clf_loss(T, M, K, *, seed):
    g = generator(seed)
    theta_out = stochastic(M, K, dim=0, g=g)  # left-stochastic over m
    log_theta_out = theta_out.clamp_min(torch.finfo(theta_out.dtype).eps).log()
    Pi = stochastic(T, M, dim=1, g=g)  # simplex rows
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    labelled = rand(T, g=g) > 0.25
    return log_theta_out, Pi, gamma_source, eow, labelled


# Each loss kernel equals Σ_{t,k} gamma_source[t, k]·cost[t, k] over its cost kernel's output.
# The loss and cost oracles pin both sides of that identity, so it has no test of its own.
def _ref_clf_loss(log_theta_out, Pi, gamma_source, delta, eow, labelled):
    """Naive implementation of classification_output_loss_ (the added contribution)."""
    T, M = Pi.shape
    Ks = gamma_source.shape[1]
    acc = 0.0
    for t in range(T):
        if not bool(labelled[t]):
            continue
        for m in range(M):
            for ks in range(Ks):
                acc += (
                    float(eow[t])
                    * float(Pi[t, m])
                    * float(gamma_source[t, ks])
                    * float(log_theta_out[m, ks])
                )
    return -delta * acc


def _make_reg_loss(T, M, K, *, seed):
    g = generator(seed)
    Cy = randn(M, K, g=g)  # regression centroids
    Y = randn(T, M, g=g)
    target_sq = (Y * Y).sum(1)  # (T,) Σ_m Y²
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    labelled = rand(T, g=g) > 0.25
    return Cy, Y, target_sq, gamma_source, eow, labelled


def _ref_reg_loss(Cy, Y, gamma_source, delta, eow, labelled):
    """Naive implementation of regression_output_loss_: gated weighted squared residual."""
    T, M = Y.shape
    Ks = gamma_source.shape[1]
    acc = 0.0
    for t in range(T):
        if not bool(labelled[t]):
            continue
        for k in range(Ks):
            sq = sum((float(Y[t, m]) - float(Cy[m, k])) ** 2 for m in range(M))
            acc += float(eow[t]) * float(gamma_source[t, k]) * sq
    return delta * acc


def _make_clf_propagate(T, M, K, *, seed):
    g = generator(seed)
    gamma_source = stochastic(T, K, dim=1, g=g)
    theta_out = stochastic(M, K, dim=0, g=g)  # left-stochastic over m
    return gamma_source, theta_out


def _ref_clf_propagate(gamma_source, theta_out, eps):
    """Naive implementation of propagate_classification_output_: GEMM + defensive row-renorm."""
    T = gamma_source.shape[0]
    Ks = gamma_source.shape[1]
    M = theta_out.shape[0]
    out = [[0.0] * M for _ in range(T)]
    for t in range(T):
        for m in range(M):
            out[t][m] = sum(float(gamma_source[t, k]) * float(theta_out[m, k]) for k in range(Ks))
        row = max(sum(out[t]), eps)  # defensive row-renorm (floored at eps)
        for m in range(M):
            out[t][m] /= row
    return torch.tensor(out, dtype=gamma_source.dtype)


def _make_reg_propagate(T, M, K, *, seed):
    g = generator(seed)
    gamma_source = stochastic(T, K, dim=1, g=g)
    Cy = randn(M, K, g=g)  # regression centroids
    return gamma_source, Cy


def _ref_reg_propagate(gamma_source, Cy):
    """Naive implementation of propagate_regression_output_: out[t,m] = Σ_k gamma·Cy (no renorm)."""
    T, Ks = gamma_source.shape
    M = Cy.shape[0]
    out = torch.zeros(T, M, dtype=gamma_source.dtype)
    for t in range(T):
        for m in range(M):
            out[t, m] = sum(float(gamma_source[t, k]) * float(Cy[m, k]) for k in range(Ks))
    return out


class TestUpdateClassificationOutput:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        Pi, gamma_source, eow, labelled = _make_clf(T, M, K, seed=seed)
        theta_out, log_theta_out, scratch_K, scratch_TK = _buffers(T, M, K)
        assert_zero_alloc(
            update_classification_output_,
            theta_out,
            log_theta_out,
            Pi,
            gamma_source,
            eow,
            labelled,
            scratch_K,
            scratch_TK,
        )
        theta_ref, log_ref = _ref_clf(Pi, gamma_source, eow, labelled, M)
        assert_close(theta_out, theta_ref)
        assert_close(log_theta_out, log_ref)
        assert_close(theta_out.sum(0), torch.ones(K, dtype=DTYPE))  # left-stochastic

    def test_empty_cluster_uniform(self):
        T, M, K = 7, 3, 4
        Pi, gamma_source, eow, labelled = _make_clf(T, M, K, seed=11)
        gamma_source[:, 1] = 0.0  # cluster 1 carries no labelled mass
        theta_out, log_theta_out, scratch_K, scratch_TK = _buffers(T, M, K)
        update_classification_output_(
            theta_out,
            log_theta_out,
            Pi,
            gamma_source,
            eow,
            labelled,
            scratch_K,
            scratch_TK,
        )
        theta_ref, _ = _ref_clf(Pi, gamma_source, eow, labelled, M)
        assert_close(theta_out, theta_ref)
        assert_close(theta_out[:, 1], torch.full((M,), 1.0 / M, dtype=DTYPE))

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_s_convention_matches_reference(self, shape, seed):
        """S normalises the same counts along rows, so each row is a class's prototype."""
        T, M, K = shape
        Pi, gamma_source, eow, labelled = _make_clf(T, M, K, seed=seed)
        theta_out, log_theta_out, _, scratch_TK = _buffers(T, M, K)
        scratch_norm = empty(M)  # one sum per class, not per cluster
        assert_zero_alloc(
            update_classification_output_,
            theta_out,
            log_theta_out,
            Pi,
            gamma_source,
            eow,
            labelled,
            scratch_norm,
            scratch_TK,
            norm_dim=1,
        )
        theta_ref, log_ref = _ref_clf(Pi, gamma_source, eow, labelled, M, norm_dim=1)
        assert_close(theta_out, theta_ref)
        assert_close(log_theta_out, log_ref)
        assert_close(theta_out.sum(1), torch.ones(M, dtype=DTYPE))  # right-stochastic

    def test_s_convention_empty_class_uniform(self):
        """A class with no counted mass takes the uniform row over the source clusters."""
        T, M, K = 7, 3, 4
        Pi, gamma_source, eow, labelled = _make_clf(T, M, K, seed=11)
        Pi[:, 1] = 0.0  # class 1 carries no mass at all
        Pi /= Pi.sum(1, keepdim=True)
        theta_out, log_theta_out, _, scratch_TK = _buffers(T, M, K)
        scratch_norm = empty(M)
        update_classification_output_(
            theta_out,
            log_theta_out,
            Pi,
            gamma_source,
            eow,
            labelled,
            scratch_norm,
            scratch_TK,
            norm_dim=1,
        )
        theta_ref, _ = _ref_clf(Pi, gamma_source, eow, labelled, M, norm_dim=1)
        assert_close(theta_out, theta_ref)
        assert_close(theta_out[1], torch.full((K,), 1.0 / K, dtype=DTYPE))


class TestUpdateRegressionOutput:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        Y, gamma_source, eow, labelled, ymw = _make_reg(T, M, K, seed=seed)
        Cy, denom_buf, scratch_MK, scratch_TK = _reg_buffers(T, M, K)
        assert_zero_alloc(
            update_regression_output_,
            Cy,
            Y,
            gamma_source,
            eow,
            labelled,
            ymw,
            denom_buf,
            scratch_MK,
            scratch_TK,
        )
        assert_close(Cy, _ref_reg(Y, gamma_source, eow, labelled, ymw))

    def test_empty_cluster_fallback(self):
        T, M, K = 7, 3, 4
        Y, gamma_source, eow, labelled, ymw = _make_reg(T, M, K, seed=11)
        gamma_source[:, 1] = 0.0  # cluster 1 carries no labelled mass
        Cy, denom_buf, scratch_MK, scratch_TK = _reg_buffers(T, M, K)
        update_regression_output_(
            Cy, Y, gamma_source, eow, labelled, ymw, denom_buf, scratch_MK, scratch_TK
        )
        assert_close(Cy, _ref_reg(Y, gamma_source, eow, labelled, ymw))
        assert_close(Cy[:, 1], ymw)  # zero-mass cluster -> Y_mean_weighted


class TestClassificationOutputAccumulateIntoSourceCost:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        cost_source, log_theta_out, Pi, eow, labelled = _make_clf_cost(T, M, K, seed=seed)
        ref = _ref_clf_cost(cost_source, log_theta_out, Pi, DELTA, eow, labelled)
        scratch_TM = empty(T, M)
        assert_zero_alloc(
            classification_output_accumulate_into_source_cost_,
            cost_source,
            log_theta_out,
            Pi,
            DELTA,
            eow,
            labelled,
            scratch_TM,
        )
        assert_close(cost_source, ref)


class TestRegressionOutputAccumulateIntoSourceCost:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        cost_source, Cy, Y, target_sq, eow, labelled = _make_reg_cost(T, M, K, seed=seed)
        ref = _ref_reg_cost(cost_source, Cy, Y, DELTA, eow, labelled)
        scratch_TK = empty(T, K)
        scratch_K = empty(K)
        scratch_KD = empty(K, M)
        assert_zero_alloc(
            regression_output_accumulate_into_source_cost_,
            cost_source,
            Cy,
            Y,
            target_sq,
            DELTA,
            eow,
            labelled,
            scratch_TK,
            scratch_K,
            scratch_KD,
        )
        assert_close(cost_source, ref)

    def test_wm_weighted_matches_reference(self):
        T, M, K = 9, 3, 4
        cost_source, Cy, Y, _, eow, labelled = _make_reg_cost(T, M, K, seed=17)
        Wm = stochastic(M, dim=0, g=generator(99))  # simplex weights
        target_sq_wm = (Y * Y) @ Wm  # weighting-matched A_sq_sum
        out = cost_source.clone()
        regression_output_accumulate_into_source_cost_(
            out,
            Cy,
            Y,
            target_sq_wm,
            DELTA,
            eow,
            labelled,
            empty(T, K),
            empty(K),
            empty(K, M),
            Wm=Wm,
        )
        ref = cost_source.clone()
        for t in range(T):
            if not bool(labelled[t]):
                continue
            for ks in range(K):
                resid = sum(
                    float(Wm[m]) * (float(Y[t, m]) - float(Cy[m, ks])) ** 2 for m in range(M)
                )
                ref[t, ks] += DELTA * float(eow[t]) * resid
        assert_close(out, ref)


class TestClassificationOutputLoss:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        log_theta_out, Pi, gamma_source, eow, labelled = _make_clf_loss(T, M, K, seed=seed)
        out_scalar = empty()
        out_scalar.fill_(0.5)  # pre-populated accumulator
        before = float(out_scalar)
        scratch_TM = empty(T, M)
        scratch_TK = empty(T, K)
        scratch_scalar = empty()
        assert_zero_alloc(
            classification_output_loss_,
            out_scalar,
            log_theta_out,
            Pi,
            gamma_source,
            DELTA,
            eow,
            labelled,
            scratch_TM,
            scratch_TK,
            scratch_scalar,
        )
        expected = before + _ref_clf_loss(log_theta_out, Pi, gamma_source, DELTA, eow, labelled)
        assert_close(out_scalar, torch.tensor(expected, dtype=DTYPE))


class TestRegressionOutputLoss:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        Cy, Y, target_sq, gamma_source, eow, labelled = _make_reg_loss(T, M, K, seed=seed)
        before = 0.5
        out_scalar = empty()
        out_scalar.fill_(before)
        scratch_TK = empty(T, K)
        scratch_K = empty(K)
        scratch_KD = empty(K, M)
        scratch_scalar = empty()
        assert_zero_alloc(
            regression_output_loss_,
            out_scalar,
            Cy,
            Y,
            target_sq,
            gamma_source,
            DELTA,
            eow,
            labelled,
            scratch_TK,
            scratch_K,
            scratch_KD,
            scratch_scalar,
        )
        ref = before + _ref_reg_loss(Cy, Y, gamma_source, DELTA, eow, labelled)
        assert_close(out_scalar, torch.tensor(ref, dtype=DTYPE))

    def test_wm_weighted_matches_reference(self):
        T, M, K = 9, 3, 4
        Cy, Y, _, gamma_source, eow, labelled = _make_reg_loss(T, M, K, seed=23)
        Wm = stochastic(M, dim=0, g=generator(99))  # simplex weights
        target_sq_wm = (Y * Y) @ Wm  # weighting-matched A_sq_sum
        before = 0.5
        out_scalar = empty()
        out_scalar.fill_(before)
        regression_output_loss_(
            out_scalar,
            Cy,
            Y,
            target_sq_wm,
            gamma_source,
            DELTA,
            eow,
            labelled,
            empty(T, K),
            empty(K),
            empty(K, M),
            empty(),
            Wm=Wm,
        )
        acc = 0.0
        for t in range(T):
            if not bool(labelled[t]):
                continue
            for k in range(K):
                sq = sum(float(Wm[m]) * (float(Y[t, m]) - float(Cy[m, k])) ** 2 for m in range(M))
                acc += float(eow[t]) * float(gamma_source[t, k]) * sq
        ref = before + DELTA * acc
        assert_close(out_scalar, torch.tensor(ref, dtype=DTYPE))


class TestPropagateClassificationOutput:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        gamma_source, theta_out = _make_clf_propagate(T, M, K, seed=seed)
        out_TM = empty(T, M)
        scratch_T1 = empty(T, 1)
        assert_zero_alloc(
            propagate_classification_output_, out_TM, gamma_source, theta_out, scratch_T1
        )
        eps = torch.finfo(DTYPE).eps
        ref = _ref_clf_propagate(gamma_source, theta_out, eps)
        assert_close(out_TM, ref)
        assert_close(out_TM.sum(1), torch.ones(T, dtype=DTYPE))  # rows simplex

    def test_zero_row_stays_zero(self):
        T, M, K = 7, 3, 4
        gamma_source, theta_out = _make_clf_propagate(T, M, K, seed=11)
        gamma_source[0] = 0.0  # degenerate all-zero source row
        out_TM = empty(T, M)
        scratch_T1 = empty(T, 1)
        propagate_classification_output_(out_TM, gamma_source, theta_out, scratch_T1)
        eps = torch.finfo(DTYPE).eps
        ref = _ref_clf_propagate(gamma_source, theta_out, eps)
        assert_close(out_TM, ref)
        assert_close(out_TM[0], torch.zeros(M, dtype=DTYPE))  # 0 / eps -> 0


class TestPropagateRegressionOutput:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        gamma_source, Cy = _make_reg_propagate(T, M, K, seed=seed)
        out_TM = empty(T, M)
        assert_zero_alloc(propagate_regression_output_, out_TM, gamma_source, Cy)
        assert_close(out_TM, _ref_reg_propagate(gamma_source, Cy))
