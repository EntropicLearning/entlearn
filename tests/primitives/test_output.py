"""Tests for the output-block primitives (output.py)."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.centroids import update_euclidean_centroids_
from entlearn.primitives.normalise import _tiny
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
    def test_an_overflowing_float32_ratio_can_still_have_a_finite_scaled_gap(self):
        logits = torch.tensor([[0.0, -1e-40]], dtype=torch.float32, device=DEVICE)
        delta, temperature = 1e38, 0.01
        scaled_gap = -float(logits[0, 1]) * delta / temperature
        p = 1.0 / (1.0 + math.exp(-scaled_gap))
        expected = torch.tensor([[p, 1.0 - p]], dtype=torch.float32, device=DEVICE)
        scratch = torch.empty(1, 1, dtype=torch.float32, device=DEVICE)
        indices = torch.empty(1, 1, dtype=torch.int64, device=DEVICE)

        assert_zero_alloc(
            geometric_probabilities_, logits, logits, delta, temperature, scratch, indices
        )

        torch.testing.assert_close(logits, expected)

    def test_an_all_negative_infinity_row_reads_as_uniform(self):
        logits = torch.tensor([[-math.inf, -math.inf], [0.0, -1.0]], dtype=DTYPE, device=DEVICE)
        scratch = empty(2, 1)
        indices = torch.empty(2, 1, dtype=torch.int64, device=DEVICE)

        assert_zero_alloc(geometric_probabilities_, logits, logits, 1.0, 0.5, scratch, indices)

        p = 1 / (1 + math.exp(-2))
        expected = torch.tensor([[0.5, 0.5], [p, 1 - p]], dtype=DTYPE, device=DEVICE)
        assert_close(logits, expected)

    @pytest.mark.parametrize(
        "temperature",
        [2 * _tiny(DTYPE), _tiny(DTYPE) / 4],
        ids=["normal", "subnormal"],
    )
    def test_a_positive_temperature_keeps_a_small_logit_gap_soft(self, temperature):
        logits = torch.tensor([[0.0, -temperature]], dtype=DTYPE, device=DEVICE)
        scratch = empty(1, 1)
        indices = torch.empty(1, 1, dtype=torch.int64, device=DEVICE)

        assert_zero_alloc(
            geometric_probabilities_, logits, logits, 1.0, temperature, scratch, indices
        )

        p = 1 / (1 + math.exp(-1))
        expected = torch.tensor([[p, 1 - p]], dtype=DTYPE, device=DEVICE)
        assert_close(logits, expected)

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
    eow *= rand(T, g=g) > 0.25  # mostly labelled; unlabelled rows have zero weight
    return Pi, gamma_source, eow


def _ref_clf(Pi, gamma_source, eow, M, norm_dim=0):
    """Naive implementation of update_classification_output_: gated numer/denom + log.

    ``norm_dim=0`` divides each column by the cluster's labelled mass (M coupling);
    ``norm_dim=1`` divides each row by the class's counted mass (S coupling). Same
    counts either way, and an empty slice takes the uniform over the axis normalised.
    """
    tiny = _tiny(Pi.dtype)
    T = Pi.shape[0]
    Ks = gamma_source.shape[1]
    numer = torch.zeros(M, Ks, dtype=Pi.dtype)
    denom = torch.zeros(Ks, dtype=Pi.dtype)
    row_denom = torch.zeros(M, dtype=Pi.dtype)
    for t in range(T):
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
            logt[m, k] = math.log(max(float(theta[m, k]), tiny))
    return theta, logt


def _make_reg(T, M, K, *, seed):
    g = generator(seed)
    Y = randn(T, M, g=g)  # regression target
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    eow *= rand(T, g=g) > 0.25  # mostly labelled; unlabelled rows have zero weight
    Y_mean_weighted = (eow.unsqueeze(1) * Y).sum(0) / eow.sum()  # (M,) eow-weighted labelled mean
    return Y, gamma_source, eow, Y_mean_weighted


def _ref_reg(Y, gamma_source, eow, Y_mean_weighted):
    """Naive implementation of update_regression_output_: gated numer/denom, mass <= tiny -> Y_mean."""
    T, M = Y.shape
    Ks = gamma_source.shape[1]
    numer = torch.zeros(M, Ks, dtype=Y.dtype)
    denom = torch.zeros(Ks, dtype=Y.dtype)
    for t in range(T):
        for k in range(Ks):
            denom[k] += float(eow[t]) * float(gamma_source[t, k])
            for m in range(M):
                numer[m, k] += float(eow[t]) * float(gamma_source[t, k]) * float(Y[t, m])
    Cy = torch.zeros(M, Ks, dtype=Y.dtype)
    for k in range(Ks):
        for m in range(M):
            if float(denom[k]) > _tiny(Y.dtype):
                Cy[m, k] = float(numer[m, k]) / float(denom[k])
            else:
                Cy[m, k] = float(Y_mean_weighted[m])
    return Cy


def _counts(target, gamma_source, eow):
    """Return the target sums ``targetᵀ (eow ∘ Γ)`` and the cluster masses ``eowᵀ Γ``."""
    gated = eow.unsqueeze(1) * gamma_source
    return target.T @ gated, gated.sum(0)


def _make_clf_cost(T, M, K, *, seed):
    g = generator(seed)
    cost_source = randn(T, K, g=g)  # pre-populated accumulator
    theta_out = stochastic(M, K, dim=0, g=g)  # left-stochastic over m
    log_theta_out = theta_out.clamp_min(_tiny(theta_out.dtype)).log()
    Pi = stochastic(T, M, dim=1, g=g)  # simplex rows
    eow = rand(T, g=g)
    eow *= rand(T, g=g) > 0.25  # unlabelled rows have zero weight
    return cost_source, log_theta_out, Pi, eow


def _ref_clf_cost(cost_source, log_theta_out, Pi, delta, eow):
    """Naive implementation of classification_output_accumulate_into_source_cost_."""
    T, Ks = cost_source.shape
    M = Pi.shape[1]
    out = cost_source.clone()
    for t in range(T):
        for ks in range(Ks):
            acc = sum(float(Pi[t, m]) * float(log_theta_out[m, ks]) for m in range(M))
            out[t, ks] += -delta * float(eow[t]) * acc
    return out


def _make_reg_cost(T, M, K, *, seed):
    g = generator(seed)
    cost_source = randn(T, K, g=g)  # pre-populated accumulator
    Cy = randn(M, K, g=g)  # regression centroids
    Y = randn(T, M, g=g)
    eow = rand(T, g=g)
    eow *= rand(T, g=g) > 0.25  # unlabelled rows have zero weight
    return cost_source, Cy, Y, eow


def _ref_reg_cost(cost_source, Cy, Y, delta, eow, Wm):
    """Naive regression coupling: the residual less its row constant, δ·eow·Σ_m Wm·(Cy² - 2·Y·Cy)."""
    T, Ks = cost_source.shape
    M = Y.shape[1]
    out = cost_source.clone()
    for t in range(T):
        for ks in range(Ks):
            resid = sum(
                float(Wm[m]) * (float(Cy[m, ks]) ** 2 - 2 * float(Y[t, m]) * float(Cy[m, ks]))
                for m in range(M)
            )
            out[t, ks] += delta * float(eow[t]) * resid
    return out


def _make_clf_loss(T, M, K, *, seed):
    g = generator(seed)
    theta_out = stochastic(M, K, dim=0, g=g)  # left-stochastic over m
    log_theta_out = theta_out.clamp_min(_tiny(theta_out.dtype)).log()
    Pi = stochastic(T, M, dim=1, g=g)  # simplex rows
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    eow *= rand(T, g=g) > 0.25  # unlabelled rows have zero weight
    return log_theta_out, Pi, gamma_source, eow


# Each loss kernel equals Σ_{t,k} gamma_source[t, k]·cost[t, k] over its cost kernel's output.
# The loss and cost oracles pin both sides of that identity, so it has no test of its own.
def _ref_clf_loss(log_theta_out, Pi, gamma_source, delta, eow):
    """Naive implementation of classification_output_loss_ (the added contribution)."""
    T, M = Pi.shape
    Ks = gamma_source.shape[1]
    acc = 0.0
    for t in range(T):
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
    gamma_source = stochastic(T, K, dim=1, g=g)
    eow = rand(T, g=g)
    eow *= rand(T, g=g) > 0.25  # unlabelled rows have zero weight
    target_moment = eow @ (Y * Y)  # (M,) Σ_t eow[t]·Y[t, m]²
    return Cy, Y, target_moment, gamma_source, eow


def _ref_reg_loss(Cy, Y, gamma_source, delta, eow):
    """Naive implementation of regression_output_loss_: gated weighted squared residual."""
    T, M = Y.shape
    Ks = gamma_source.shape[1]
    acc = 0.0
    for t in range(T):
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
        Pi, gamma_source, eow = _make_clf(T, M, K, seed=seed)
        counts, _ = _counts(Pi, gamma_source, eow)
        theta_out, log_theta_out = empty(M, K), empty(M, K)
        assert_zero_alloc(update_classification_output_, theta_out, log_theta_out, counts, empty(K))
        theta_ref, log_ref = _ref_clf(Pi, gamma_source, eow, M)
        assert_close(theta_out, theta_ref)
        assert_close(log_theta_out, log_ref)
        assert_close(theta_out.sum(0), torch.ones(K, dtype=DTYPE))  # left-stochastic

    def test_empty_cluster_uniform(self):
        T, M, K = 7, 3, 4
        Pi, gamma_source, eow = _make_clf(T, M, K, seed=11)
        gamma_source[:, 1] = 0.0  # cluster 1 carries no labelled mass
        theta_out, log_theta_out = empty(M, K), empty(M, K)
        counts, _ = _counts(Pi, gamma_source, eow)
        update_classification_output_(theta_out, log_theta_out, counts, empty(K))
        theta_ref, _ = _ref_clf(Pi, gamma_source, eow, M)
        assert_close(theta_out, theta_ref)
        assert_close(theta_out[:, 1], torch.full((M,), 1.0 / M, dtype=DTYPE))

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_s_convention_matches_reference(self, shape, seed):
        """S normalises the same counts along rows, so each row is a class's prototype."""
        T, M, K = shape
        Pi, gamma_source, eow = _make_clf(T, M, K, seed=seed)
        counts, _ = _counts(Pi, gamma_source, eow)
        theta_out, log_theta_out = empty(M, K), empty(M, K)
        scratch_norm = empty(M)  # one sum per class, not per cluster
        assert_zero_alloc(
            update_classification_output_,
            theta_out,
            log_theta_out,
            counts,
            scratch_norm,
            norm_dim=1,
        )
        theta_ref, log_ref = _ref_clf(Pi, gamma_source, eow, M, norm_dim=1)
        assert_close(theta_out, theta_ref)
        assert_close(log_theta_out, log_ref)
        assert_close(theta_out.sum(1), torch.ones(M, dtype=DTYPE))  # right-stochastic

    def test_s_convention_empty_class_uniform(self):
        """A class with no counted mass takes the uniform row over the source clusters."""
        T, M, K = 7, 3, 4
        Pi, gamma_source, eow = _make_clf(T, M, K, seed=11)
        Pi[:, 1] = 0.0  # class 1 carries no mass at all
        Pi /= Pi.sum(1, keepdim=True)
        counts, _ = _counts(Pi, gamma_source, eow)
        theta_out, log_theta_out = empty(M, K), empty(M, K)
        update_classification_output_(theta_out, log_theta_out, counts, empty(M), norm_dim=1)
        theta_ref, _ = _ref_clf(Pi, gamma_source, eow, M, norm_dim=1)
        assert_close(theta_out, theta_ref)
        assert_close(theta_out[1], torch.full((K,), 1.0 / K, dtype=DTYPE))


class TestUpdateRegressionOutput:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        Y, gamma_source, eow, ymw = _make_reg(T, M, K, seed=seed)
        numerator, denom = _counts(Y, gamma_source, eow)
        Cy = empty(M, K)
        assert_zero_alloc(update_regression_output_, Cy, numerator, denom, ymw, empty(K))
        assert_close(Cy, _ref_reg(Y, gamma_source, eow, ymw))

    @pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 2.0])
    def test_tiny_mass_takes_the_global_target_mean(self, scale):
        mass = _tiny(DTYPE) * scale
        denom = torch.tensor([mass, 0.0], dtype=DTYPE, device=DEVICE)
        numerator = torch.tensor([[mass / 4, 0.0], [mass / 2, 0.0]], dtype=DTYPE, device=DEVICE)
        fallback = torch.tensor([0.75, 1.0], dtype=DTYPE, device=DEVICE)
        out = empty(2, 2)

        assert_zero_alloc(update_regression_output_, out, numerator, denom, fallback, empty(2))

        first = [0.75, 1.0] if scale <= 1 else [0.25, 0.5]
        expected = torch.tensor([[first[0], 0.75], [first[1], 1.0]], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_equals_the_input_centroid_update_except_on_an_empty_cluster(self):
        Y, gamma_source, eow, ymw = _make_reg(7, 3, 4, seed=0)
        gamma_source[:, 1] = 0.0
        numerator, denom = _counts(Y, gamma_source, eow)
        Cy, C = empty(3, 4), empty(4, 3)
        empty_clusters = torch.empty(4, dtype=torch.bool, device=DEVICE)

        update_regression_output_(Cy, numerator, denom, ymw, empty(4))
        update_euclidean_centroids_(C, numerator.T.contiguous(), denom, empty_clusters)

        filled = [0, 2, 3]
        assert torch.equal(Cy[:, filled], C.T[:, filled])
        assert_close(Cy[:, 1], ymw)  # the head takes the labelled mean, the input zero
        assert_close(C[1], torch.zeros(3, dtype=DTYPE))


class TestClassificationOutputAccumulateIntoSourceCost:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        cost_source, log_theta_out, Pi, eow = _make_clf_cost(T, M, K, seed=seed)
        ref = _ref_clf_cost(cost_source, log_theta_out, Pi, DELTA, eow)
        scratch_TM = empty(T, M)
        assert_zero_alloc(
            classification_output_accumulate_into_source_cost_,
            cost_source,
            log_theta_out,
            Pi,
            DELTA,
            eow,
            scratch_TM,
        )
        assert_close(cost_source, ref)


class TestRegressionOutputAccumulateIntoSourceCost:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        cost_source, Cy, Y, eow = _make_reg_cost(T, M, K, seed=seed)
        ref = _ref_reg_cost(cost_source, Cy, Y, DELTA, eow, torch.ones(M, dtype=DTYPE))
        weighted_Y = eow.unsqueeze(1) * Y
        args = (cost_source, Cy, weighted_Y, DELTA, eow, empty(K), empty(K, M))
        assert_zero_alloc(regression_output_accumulate_into_source_cost_, *args)
        assert_close(cost_source, ref)

    def test_wm_weighted_matches_reference(self):
        T, M, K = 9, 3, 4
        cost_source, Cy, Y, eow = _make_reg_cost(T, M, K, seed=17)
        Wm = stochastic(M, dim=0, g=generator(99))  # simplex weights
        out = cost_source.clone()
        weighted_Y = eow.unsqueeze(1) * Y
        regression_output_accumulate_into_source_cost_(
            out, Cy, weighted_Y, DELTA, eow, empty(K), empty(K, M), Wm=Wm
        )
        assert_close(out, _ref_reg_cost(cost_source, Cy, Y, DELTA, eow, Wm))

    def test_without_weights_every_row_weighs_one(self):
        T, M, K = 9, 3, 4
        cost_source, Cy, Y, _ = _make_reg_cost(T, M, K, seed=19)
        out = cost_source.clone()
        regression_output_accumulate_into_source_cost_(
            out, Cy, Y, DELTA, None, empty(K), empty(K, M)
        )
        ones = torch.ones(T, dtype=DTYPE)
        assert_close(
            out, _ref_reg_cost(cost_source, Cy, Y, DELTA, ones, torch.ones(M, dtype=DTYPE))
        )


class TestClassificationOutputLoss:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        log_theta_out, Pi, gamma_source, eow = _make_clf_loss(T, M, K, seed=seed)
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
            scratch_TM,
            scratch_TK,
            scratch_scalar,
        )
        expected = before + _ref_clf_loss(log_theta_out, Pi, gamma_source, DELTA, eow)
        assert_close(out_scalar, torch.tensor(expected, dtype=DTYPE))


class TestRegressionOutputLoss:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, M, K = shape
        Cy, Y, target_moment, gamma_source, eow = _make_reg_loss(T, M, K, seed=seed)
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
            target_moment,
            gamma_source,
            DELTA,
            eow,
            scratch_TK,
            scratch_K,
            scratch_KD,
            scratch_scalar,
        )
        ref = before + _ref_reg_loss(Cy, Y, gamma_source, DELTA, eow)
        assert_close(out_scalar, torch.tensor(ref, dtype=DTYPE))

    def test_wm_weighted_matches_reference(self):
        T, M, K = 9, 3, 4
        Cy, Y, target_moment, gamma_source, eow = _make_reg_loss(T, M, K, seed=23)
        Wm = stochastic(M, dim=0, g=generator(99))  # simplex weights
        before = 0.5
        out_scalar = empty()
        out_scalar.fill_(before)
        regression_output_loss_(
            out_scalar,
            Cy,
            Y,
            target_moment,
            gamma_source,
            DELTA,
            eow,
            empty(T, K),
            empty(K),
            empty(K, M),
            empty(),
            Wm=Wm,
        )
        acc = 0.0
        for t in range(T):
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
