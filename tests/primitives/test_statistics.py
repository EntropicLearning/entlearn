"""Tests for state-free statistical primitives."""

import math

import pytest
import torch
from _alloc import assert_no_float64, assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.normalise import _tiny
from entlearn.primitives.softmax import softmax_with_temp_
from entlearn.primitives.statistics import (
    effective_dimension,
    entropy,
    entropy_penalty_,
    inlier_scores_,
)
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand

ENTROPY_SHAPES = [(7, 4), (5, 3), (6,), (4,)]  # mix of 2-D (T, K) and 1-D (T,)/(D,)
COEF = 0.3  # fixed positive entropy coefficient


class TestEntropy:
    def test_zero_mass_convention_and_effective_dimension(self) -> None:
        # The last row carries no mass: its entropy is 0, not NaN.
        probabilities = torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]], dtype=DTYPE, device=DEVICE
        )

        entropies = entropy(probabilities, dim=1)
        raw = effective_dimension(probabilities, dim=1, normalise=False)
        normalised = effective_dimension(probabilities, dim=1)

        assert_close(entropies, torch.tensor([0.0, math.log(2.0), 0.0], dtype=DTYPE))
        assert_close(raw, torch.tensor([1.0, 2.0, 1.0], dtype=DTYPE))
        assert_close(normalised, torch.tensor([0.5, 1.0, 0.5], dtype=DTYPE))

        # ``dim`` must reach the entropy itself, not only the normalising axis length.
        columns = probabilities.transpose(0, 1).contiguous()
        assert_close(effective_dimension(columns, dim=0, normalise=False), raw)

    def test_positive_probabilities_take_the_exact_entropy(self) -> None:
        info = torch.finfo(DTYPE)
        values = [0.0, _tiny(DTYPE) / 16, _tiny(DTYPE), info.eps / 1024, 0.25]
        p = torch.tensor(values, dtype=DTYPE, device=DEVICE)

        contributions = torch.stack([entropy(v.unsqueeze(0)) for v in p])

        # Subnormal values survive in the tensor unless the device flushes them to zero
        expected = [-float(v) * math.log(float(v)) if float(v) > 0 else 0.0 for v in p]
        assert torch.isfinite(contributions).all()
        # Relative only: the values themselves lie far below any absolute tolerance
        torch.testing.assert_close(
            contributions.cpu(),
            torch.tensor(expected, dtype=DTYPE),
            rtol=8 * torch.finfo(DTYPE).eps,
            atol=0.0,
        )


class TestInlierScores:
    def test_writes_the_right_continuous_ecdf(self) -> None:
        reference = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        query = torch.tensor([1.999, 2.0, 4.0], dtype=DTYPE, device=DEVICE)
        result = torch.full((3,), -1.0, dtype=DTYPE, device=DEVICE)

        inlier_scores_(result, reference, query)

        assert_close(result, torch.tensor([0.25, 0.75, 1.0], dtype=DTYPE))

    def test_empty_reference_raises(self) -> None:
        result = empty(1)
        with pytest.raises(ValueError, match="empty"):
            inlier_scores_(result, empty(0), torch.tensor([1.0], dtype=DTYPE, device=DEVICE))


def _make_entropy(shape, *, seed):
    """Build a non-negative ``p`` of the given shape via a seeded generator.

    ``p`` need not be normalised: the penalty sums ``p·log p`` over every element
    regardless. For one shape a single entry is forced to ``0.0`` to exercise the
    convention ``0·log 0 = 0``, and another to a positive value below machine precision.
    """
    p = rand(*shape, g=generator(seed))  # non-negative
    if shape == (6,):  # a zero entry and a positive entry below eps
        p[0] = 0.0
        p[1] = torch.finfo(p.dtype).eps / 64
    return p


def _ref_entropy(p, coefficient):
    """Naive-loop oracle for the scalar contribution added to ``out_scalar``.

    Returns ``coefficient·Σ_elements p·log p = -coefficient·H``, with ``0·log 0 = 0``,
    i.e. the amount the penalty adds to ``out_scalar`` (decreasing it by ``coef·H``).
    """
    acc = sum(float(v) * math.log(float(v)) for v in p.flatten() if float(v) > 0)
    return coefficient * acc


class TestEntropyPenalty:
    @pytest.mark.parametrize("probability", ["below-eps", "subnormal"])
    def test_an_isolated_small_contribution_uses_its_own_log(self, probability):
        info = torch.finfo(DTYPE)
        value = info.eps / 1024 if probability == "below-eps" else _tiny(DTYPE) / 16
        p = torch.tensor([value, 0.0], dtype=DTYPE, device=DEVICE)
        loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
        before = p.clone()

        assert_zero_alloc(entropy_penalty_, loss, p, 1.0, torch.empty_like(p), empty())

        stored = float(p[0])
        expected = stored * math.log(stored) if stored > 0 else 0.0
        assert float(loss) == pytest.approx(expected, rel=8 * info.eps, abs=0)
        assert torch.equal(p, before)

    @pytest.mark.parametrize("shape", ENTROPY_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        p = _make_entropy(shape, seed=seed)
        out_scalar = empty()
        out_scalar.fill_(0.5)  # pre-populated accumulator
        before = float(out_scalar)
        p_in = p.clone()
        log_buffer = torch.empty_like(p)
        scratch_scalar = empty()
        assert_zero_alloc(entropy_penalty_, out_scalar, p, COEF, log_buffer, scratch_scalar)
        expected = before + _ref_entropy(p, COEF)
        assert_close(out_scalar, torch.tensor(expected, dtype=DTYPE))
        assert_close(p, p_in)  # p is read-only

    def test_row_weights_change_importance_not_entropy_temperature(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]],
            dtype=DTYPE,
            device=DEVICE,
        )
        weights = torch.tensor([0.1, 0.3, 0.6], dtype=DTYPE, device=DEVICE)
        out_scalar = torch.zeros((), dtype=DTYPE, device=DEVICE)

        entropy_penalty_(
            out_scalar,
            probabilities,
            COEF,
            torch.empty_like(probabilities),
            empty(),
            row_weights=weights,
        )

        expected = COEF * sum(
            float(weights[t]) * _ref_entropy(probabilities[t], 1.0)
            for t in range(probabilities.shape[0])
        )
        assert_close(out_scalar, torch.tensor(expected, dtype=DTYPE))

    def test_row_weighted_route_allocates_no_tensor_storage(self) -> None:
        probabilities = _make_entropy((7, 4), seed=SEEDS[0])
        weights = torch.full((7,), 1.0 / 7, dtype=DTYPE, device=DEVICE)
        assert_zero_alloc(
            entropy_penalty_,
            empty(),
            probabilities,
            COEF,
            torch.empty_like(probabilities),
            empty(),
            row_weights=weights,
        )


class TestExactEntropyObjective:
    """The affiliation update minimises the objective the loss reports, down to ``tiny``.

    One binary row with costs ``(0, c)`` at unit temperature has the softmax minimiser
    ``s = 1/(1 + e^c)``. Taking ``c = -log(tiny) - 1`` puts ``s`` just above ``tiny``. The
    exact objective ``cost·gamma - H(gamma)`` at ``s`` is no higher than the vertex ``s = 0``,
    whose objective is zero.
    """

    def test_the_softmax_beats_the_vertex_under_the_reported_entropy(self) -> None:
        tiny = _tiny(DTYPE)
        gap = -math.log(tiny) - 1.0
        cost = torch.tensor([0.0, gap], dtype=DTYPE, device=DEVICE)
        gamma = torch.empty_like(cost)
        softmax_with_temp_(gamma, cost, 1.0, 0, empty(1))
        vertex = torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE)

        def objective(probabilities: torch.Tensor) -> float:
            out = torch.dot(cost, probabilities).reshape(())
            entropy_penalty_(out, probabilities, 1.0, torch.empty_like(probabilities), empty())
            return float(out)

        s = float(gamma[1])
        # The rounding of the reported objective: its terms are of size cost·s
        tolerance = 4 * torch.finfo(DTYPE).eps * gap * s
        assert s > tiny
        assert objective(gamma) <= objective(vertex) + tolerance


class TestStatisticsBoundary:
    def test_float32_calculations_create_no_float64_tensor(self) -> None:
        probabilities = torch.tensor([0.25, 0.75], dtype=torch.float32)
        reference = torch.tensor([0.1, 0.3], dtype=torch.float32)
        query = torch.tensor([0.2], dtype=torch.float32)
        scores = torch.empty(1, dtype=torch.float32)

        # A float64 default makes a factory call that ignores the input dtype visible.
        prior = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            assert assert_no_float64(entropy, probabilities).dtype == torch.float32
            assert assert_no_float64(effective_dimension, probabilities).dtype == torch.float32
            assert_no_float64(inlier_scores_, scores, reference, query)
        finally:
            torch.set_default_dtype(prior)
