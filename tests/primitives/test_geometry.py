"""Tests for grouped geometry primitives."""

import pytest
import torch
from _alloc import assert_no_float64, assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.geometry import grouped_means, mark_non_empty_clusters_
from primitives._fixtures import SEEDS, assert_close, empty, generator, stochastic

EMPTY_SHAPES = [(7, 3), (10, 4), (6, 2), (5, 5)]  # (T, K)


class TestGroupedMeans:
    def test_returns_weighted_means_and_empty_group_fallbacks(self) -> None:
        index = torch.tensor([0, 0, 1], device=DEVICE)
        weights = torch.tensor([1.0, 3.0, 2.0], dtype=DTYPE, device=DEVICE)
        dense = torch.tensor([[0.0, 2.0], [4.0, 6.0], [10.0, 14.0]], dtype=DTYPE, device=DEVICE)
        categorical = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]], dtype=DTYPE, device=DEVICE)
        ]

        dense_means, categorical_means, masses = grouped_means(
            index, weights, dense, categorical, 3
        )

        assert_close(masses, torch.tensor([4.0, 2.0, 0.0], dtype=DTYPE))
        assert_close(
            dense_means,
            torch.tensor([[3.0, 5.0], [10.0, 14.0], [16.0 / 3.0, 8.0]], dtype=DTYPE),
        )
        assert_close(
            categorical_means[0],
            torch.tensor([[0.25, 0.75], [0.25, 0.75], [0.5, 0.5]], dtype=DTYPE),
        )

    @pytest.mark.parametrize(("scale", "empty_group"), [(1.0, True), (2.0, False)])
    def test_mass_at_machine_precision_counts_as_empty(
        self, scale: float, empty_group: bool
    ) -> None:
        index = torch.tensor([0, 1], device=DEVICE)
        weights = torch.tensor([scale * torch.finfo(DTYPE).eps, 1.0], dtype=DTYPE, device=DEVICE)
        dense = torch.tensor([[4.0], [1.0]], dtype=DTYPE, device=DEVICE)
        categorical = [torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE, device=DEVICE)]

        dense_means, categorical_means, _masses = grouped_means(
            index, weights, dense, categorical, 2
        )

        if empty_group:
            fallback = (weights[:, None] * dense).sum(dim=0) / weights.sum()
            assert_close(dense_means[0], fallback)
            assert_close(categorical_means[0][0], torch.tensor([0.5, 0.5], dtype=DTYPE))
        else:
            assert_close(dense_means[0], dense[0])
            assert_close(categorical_means[0][0], categorical[0][0])

    def test_float32_inputs_stay_float32_and_read_only(self) -> None:
        index = torch.tensor([0, 1])
        weights = torch.tensor([0.25, 0.75], dtype=torch.float32)
        dense = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        categorical = [torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)]
        dense_before = dense.clone()
        categorical_before = categorical[0].clone()

        dense_means, categorical_means, masses = assert_no_float64(
            grouped_means,
            index,
            weights,
            dense,
            categorical,
            2,
        )

        assert dense_means.dtype is categorical_means[0].dtype is masses.dtype is torch.float32
        assert torch.equal(dense, dense_before)
        assert torch.equal(categorical[0], categorical_before)

    def test_zero_width_categorical_feature_with_an_empty_group(self) -> None:
        index = torch.tensor([0, 0], device=DEVICE)
        weights = torch.tensor([1.0, 1.0], dtype=DTYPE, device=DEVICE)
        dense = torch.tensor([[1.0], [3.0]], dtype=DTYPE, device=DEVICE)
        categorical = [torch.zeros(2, 0, dtype=DTYPE, device=DEVICE)]  # no categories

        _dense_means, categorical_means, masses = grouped_means(
            index, weights, dense, categorical, 2
        )

        assert categorical_means[0].shape == (2, 0)
        assert_close(masses, torch.tensor([2.0, 0.0], dtype=DTYPE))


def _make_empty(T, K, *, seed):
    """Build a ``(T, K)`` row-stochastic ``gamma`` with one forced-empty cluster.

    Returns ``gamma`` plus the caller-supplied ``(K,)`` output buffers
    ``not_empty`` (bool) and ``cluster_mass``. Column ``1`` is zeroed so that at
    least one cluster falls below the epsilon threshold.
    """
    g = generator(seed)
    gamma = stochastic(T, K, dim=1, g=g)
    gamma[:, 1] = 0.0  # force an empty cluster (sub-epsilon mass)
    gamma /= gamma.sum(1, keepdim=True)  # row-stochastic Γ
    not_empty = torch.empty(K, dtype=torch.bool, device=DEVICE)
    cluster_mass = empty(K)
    return gamma, not_empty, cluster_mass


def _ref_empty(gamma, threshold):
    """Naive-loop implementation for (not_empty, cluster_mass, count)."""
    T, K = gamma.shape
    mass = [sum(float(gamma[t, k]) for t in range(T)) for k in range(K)]
    not_empty = [mass[k] > threshold for k in range(K)]
    return not_empty, mass, sum(not_empty)


class TestMarkNonEmptyClusters:
    @pytest.mark.parametrize("shape", EMPTY_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, seed):
        T, K = shape
        gamma, not_empty, cluster_mass = _make_empty(T, K, seed=seed)
        count = assert_zero_alloc(mark_non_empty_clusters_, not_empty, cluster_mass, gamma)
        threshold = torch.finfo(gamma.dtype).eps
        ne_ref, mass_ref, count_ref = _ref_empty(gamma, threshold)
        assert count == count_ref
        assert not_empty.tolist() == ne_ref
        assert_close(cluster_mass, torch.tensor(mass_ref, dtype=DTYPE))
