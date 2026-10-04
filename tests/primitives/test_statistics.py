"""Tests for state-free statistical primitives."""

import math

import pytest
import torch
from _alloc import assert_no_float64, assert_zero_alloc
from conftest import DEVICE, DTYPE

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
        probabilities = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=DTYPE, device=DEVICE)

        entropies = entropy(probabilities, dim=1)
        raw = effective_dimension(probabilities, dim=1, normalise=False)
        normalised = effective_dimension(probabilities, dim=1)

        assert_close(entropies, torch.tensor([0.0, math.log(2.0)], dtype=DTYPE))
        assert_close(raw, torch.tensor([1.0, 2.0], dtype=DTYPE))
        assert_close(normalised, torch.tensor([0.5, 1.0], dtype=DTYPE))

        # ``dim`` must reach the entropy itself, not only the normalising axis length.
        columns = probabilities.transpose(0, 1).contiguous()
        assert_close(effective_dimension(columns, dim=0, normalise=False), raw)


class TestInlierScores:
    def test_writes_the_right_continuous_ecdf(self) -> None:
        reference = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        query = torch.tensor([1.999, 2.0, 4.0], dtype=DTYPE, device=DEVICE)
        result = torch.full((3,), -1.0, dtype=DTYPE, device=DEVICE)

        returned = inlier_scores_(result, reference, query)

        assert returned is None
        assert_close(result, torch.tensor([0.25, 0.75, 1.0], dtype=DTYPE))

    def test_empty_reference_raises(self) -> None:
        result = empty(1)
        with pytest.raises(ValueError, match="empty"):
            inlier_scores_(result, empty(0), torch.tensor([1.0], dtype=DTYPE, device=DEVICE))


def _make_entropy(shape, *, seed):
    """Build a non-negative ``p`` of the given shape via a seeded generator.

    ``p`` need not be normalised: the penalty sums ``p·log p`` over every element
    regardless. For one shape a single entry is forced to ``0.0`` to exercise the
    epsilon floor (where ``0·log(eps) = 0``).
    """
    p = rand(*shape, g=generator(seed))  # non-negative
    if shape == (6,):  # force the eps-floor path on at least one case
        p[0] = 0.0
    return p


def _ref_entropy(p, coefficient):
    """Naive-loop oracle for the scalar contribution added to ``out_scalar``.

    Returns ``coefficient·Σ_elements p·log(max(p, eps)) = -coefficient·H``, i.e.
    the amount the penalty adds to ``out_scalar`` (decreasing it by ``coef·H``).
    """
    eps = torch.finfo(p.dtype).eps
    acc = sum(float(v) * math.log(max(float(v), eps)) for v in p.flatten())
    return coefficient * acc


class TestEntropyPenalty:
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


class TestStatisticsBoundary:
    def test_float32_calculations_create_no_float64_tensor(self) -> None:
        probabilities = torch.tensor([0.25, 0.75], dtype=torch.float32)
        reference = torch.tensor([0.1, 0.3], dtype=torch.float32)
        query = torch.tensor([0.2], dtype=torch.float32)
        scores = torch.empty(1, dtype=torch.float32)

        assert assert_no_float64(entropy, probabilities).dtype is torch.float32
        assert assert_no_float64(effective_dimension, probabilities).dtype is torch.float32
        assert_no_float64(inlier_scores_, scores, reference, query)
        assert scores.dtype is torch.float32
