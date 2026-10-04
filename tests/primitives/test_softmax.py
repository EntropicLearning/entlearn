"""Tests for the softmax / argmin / log-partition primitives (softmax.py)."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

import entlearn.primitives.softmax as softmax_primitives
from entlearn.primitives.softmax import (
    argmin_assign_,
    assign_simplex_,
    compute_log_partition_,
    softmax_with_temp_,
)
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand, randn

EPS = torch.finfo(DTYPE).eps


def _keepdim_shape(shape, dim):
    """Shape of ``shape`` with ``dim`` collapsed to 1 (the keepdim buffer shape)."""
    out = list(shape)
    out[dim] = 1
    return tuple(out)


def _ref_softmax(cost, temp, dim):
    """Naive per-slice stable softmax(-cost/temp) with the uniform-1/N fallback.

    Operates on a Python-list view of ``cost`` along ``dim``; a slice whose
    shifted exp-sum is non-finite or non-positive falls back to uniform ``1/N``.
    """
    n = cost.shape[dim]
    out = torch.empty_like(cost)
    moved = cost.movedim(dim, -1)  # (..., N) view; reduce over the last axis
    out_moved = out.movedim(dim, -1)
    flat = moved.reshape(-1, n)
    out_flat = out_moved.reshape(-1, n)
    for i in range(flat.shape[0]):
        shifted = [-float(flat[i, j]) / temp for j in range(n)]
        m = max(shifted)
        exps = [math.exp(s - m) for s in shifted]
        z = sum(exps)
        if math.isfinite(z) and z > 0.0:
            for j in range(n):
                out_flat[i, j] = exps[j] / z
        else:
            for j in range(n):
                out_flat[i, j] = 1.0 / n
    return out


def _ref_argmin(cost, dim):
    """Naive one-hot at the first argmin of ``cost`` along ``dim`` (ties -> first)."""
    n = cost.shape[dim]
    out = torch.zeros_like(cost)
    moved = cost.movedim(dim, -1)
    out_moved = out.movedim(dim, -1)
    flat = moved.reshape(-1, n)
    out_flat = out_moved.reshape(-1, n)
    for i in range(flat.shape[0]):
        best_j = 0
        best_v = float(flat[i, 0])
        for j in range(1, n):
            v = float(flat[i, j])
            if v < best_v:  # strict: ties keep the first index
                best_v = v
                best_j = j
        out_flat[i, best_j] = 1.0
    return out


def _ref_log_partition(b, temp):
    """Naive stable log-sum-exp over ``-b/temp``."""
    shifted = [-float(bt) / temp for bt in b]
    m = max(shifted)
    return m + math.log(sum(math.exp(s - m) for s in shifted))


class TestSoftmaxWithTemp:
    @pytest.mark.parametrize("temp", [0.5, 2.0])
    @pytest.mark.parametrize("shape,dim", [((7, 4), 1), ((6,), 0), ((5,), 0)])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, dim, temp, seed):
        cost = randn(*shape, g=generator(seed))
        ref = _ref_softmax(cost, temp, dim)
        out = torch.empty_like(cost)
        scratch = empty(*_keepdim_shape(shape, dim))
        assert_zero_alloc(softmax_with_temp_, out, cost, temp, dim, scratch)
        assert_close(out, ref)
        slice_sums = out.sum(dim)
        assert_close(slice_sums, torch.ones_like(slice_sums))  # stochastic along dim

    @pytest.mark.parametrize("temp", [0.5, 2.0])
    @pytest.mark.parametrize("shape,dim", [((7, 4), 1), ((6,), 0), ((5,), 0)])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_torch_softmax(self, shape, dim, temp, seed):
        # Cross-check against the allocating torch.softmax on finite data (no
        # degenerate slice, so the uniform fallback never fires and the two agree).
        cost = randn(*shape, g=generator(seed))
        ref = torch.softmax(-cost / temp, dim=dim)
        out = torch.empty_like(cost)
        scratch = empty(*_keepdim_shape(shape, dim))
        softmax_with_temp_(out, cost, temp, dim, scratch)
        assert_close(out, ref)

    def test_degenerate_slice_uniform(self):
        shape, dim, temp = (5, 4), 1, 1.5
        cost = randn(*shape, g=generator(7))
        cost[2, :] = float("inf")  # whole slice is +inf -> uniform fallback
        out = torch.empty_like(cost)
        scratch = empty(*_keepdim_shape(shape, dim))
        softmax_with_temp_(out, cost, temp, dim, scratch)
        n = shape[dim]
        assert_close(out[2, :], torch.full((n,), 1.0 / n, dtype=DTYPE))
        assert_close(out.sum(dim), torch.ones(shape[0], dtype=DTYPE))


class TestArgminAssign:
    @pytest.mark.parametrize("shape,dim", [((7, 4), 1), ((6,), 0)])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, dim, seed):
        cost = randn(*shape, g=generator(seed))
        ref = _ref_argmin(cost, dim)
        out = torch.empty_like(cost)
        scratch_idx = torch.empty(_keepdim_shape(shape, dim), dtype=torch.int64, device=DEVICE)
        assert_zero_alloc(argmin_assign_, out, cost, dim, scratch_idx)
        assert_close(out, ref)
        slice_sums = out.sum(dim)
        assert_close(slice_sums, torch.ones_like(slice_sums))  # exactly one 1 per slice

    def test_tie_picks_first(self):
        shape, dim = (2, 4), 1
        cost = torch.tensor(
            [[1.0, -3.0, -3.0, 2.0], [0.0, 5.0, -1.0, -1.0]], dtype=DTYPE, device=DEVICE
        )  # row 0: tie at indices 1,2; row 1: tie at indices 2,3
        out = torch.empty_like(cost)
        scratch_idx = torch.empty(_keepdim_shape(shape, dim), dtype=torch.int64, device=DEVICE)
        argmin_assign_(out, cost, dim, scratch_idx)
        expected = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=DTYPE, device=DEVICE
        )  # first of each tied pair wins
        assert torch.equal(out, expected)


class TestComputeLogPartition:
    @pytest.mark.parametrize("temp", [0.5, 1.0, 3.0])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, temp, seed):
        t = 9
        b = randn(t, g=generator(seed))
        ref = _ref_log_partition(b, temp)
        out_scalar = empty()
        out_scalar.fill_(-123.0)  # sentinel; must be overwritten
        scratch_T = empty(t)
        scratch_scalar = empty()
        assert_zero_alloc(compute_log_partition_, out_scalar, b, temp, scratch_T, scratch_scalar)
        assert_close(out_scalar, torch.tensor(ref, dtype=DTYPE))


_AS_SHAPES = [(7, 4), (1, 3), (64, 16), (5, 1)]


class TestAssignSimplex:
    def _buffers(self, shape):
        cost = rand(*shape, g=generator(SEEDS[0]))
        out = torch.empty_like(cost)
        scratch = empty(shape[0], 1)
        idx = torch.empty(shape[0], 1, dtype=torch.int64, device=DEVICE)
        return cost, out, scratch, idx

    @pytest.mark.parametrize("shape", _AS_SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_soft_route_bitwise_matches_softmax(self, shape, seed):
        cost = rand(*shape, g=generator(seed))
        out = torch.empty_like(cost)
        expected = torch.empty_like(cost)
        scratch = empty(shape[0], 1)
        idx = torch.empty(shape[0], 1, dtype=torch.int64, device=DEVICE)
        assign_simplex_(out, cost, 0.37, 1, scratch, idx)
        softmax_with_temp_(expected, cost, 0.37, 1, scratch)
        assert torch.equal(out, expected)  # bitwise: same route, same ops

    @pytest.mark.parametrize("shape", _AS_SHAPES)
    def test_hard_route_bitwise_matches_argmin(self, shape):
        cost, out, scratch, idx = self._buffers(shape)
        expected = torch.empty_like(cost)
        assign_simplex_(out, cost, EPS, 1, scratch, idx)  # gate needs strict >
        argmin_assign_(expected, cost, 1, idx)
        assert torch.equal(out, expected)

    def test_dim0_vector_site_shape(self):
        # The Wt/Wd/Wm sites: 1-D cost, dim=0, (1,) scratches.
        cost = rand(9, g=generator(SEEDS[0]))
        out = torch.empty_like(cost)
        expected = torch.empty_like(cost)
        scratch = empty(1)
        idx = torch.empty(1, dtype=torch.int64, device=DEVICE)
        assign_simplex_(out, cost, 0.05, 0, scratch, idx)
        softmax_with_temp_(expected, cost, 0.05, 0, scratch)
        assert torch.equal(out, expected)

    def test_cost_read_only(self):
        cost, out, scratch, idx = self._buffers((7, 4))
        cost_copy = cost.clone()
        assign_simplex_(out, cost, 0.37, 1, scratch, idx)
        assign_simplex_(out, cost, EPS, 1, scratch, idx)
        assert torch.equal(cost, cost_copy)

    def test_zero_alloc_both_routes(self):
        cost, out, scratch, idx = self._buffers((7, 4))
        assert_zero_alloc(assign_simplex_, out, cost, 0.37, 1, scratch, idx)
        assert_zero_alloc(assign_simplex_, out, cost, EPS, 1, scratch, idx)


def _weighted_assign(out, cost, temp, weights):
    """Run ``weighted_assign_simplex_`` into ``out`` with fresh scratch buffers."""
    T = cost.shape[0]
    softmax_primitives.weighted_assign_simplex_(
        out,
        cost,
        temp,
        weights,
        empty(T, 1),
        torch.empty(T, 1, dtype=torch.int64, device=DEVICE),
    )


class TestAssignWeightedRows:
    def test_equal_weights_match_the_flat_average_route(self) -> None:
        cost = rand(7, 4, g=generator(SEEDS[0]))
        weights = torch.full((7,), 1.0 / 7, dtype=DTYPE, device=DEVICE)
        weighted_cost = cost * weights[:, None]
        actual = torch.empty_like(cost)
        expected = torch.empty_like(cost)

        _weighted_assign(actual, weighted_cost, 0.2, weights)
        indices = torch.empty(7, 1, dtype=torch.int64, device=DEVICE)
        assign_simplex_(expected, weighted_cost, 0.2 / 7, 1, empty(7, 1), indices)

        assert_close(actual, expected)

    def test_sample_weights_cancel_from_the_soft_assignment(self) -> None:
        base_cost = torch.tensor(
            [[0.2, 0.7], [0.8, 0.1], [0.3, 0.4]],
            dtype=DTYPE,
            device=DEVICE,
        )
        weights = torch.tensor([0.1, 0.3, 0.6], dtype=DTYPE, device=DEVICE)
        actual = torch.empty_like(base_cost)
        expected = torch.empty_like(base_cost)

        _weighted_assign(actual, base_cost * weights[:, None], 0.2, weights)
        softmax_with_temp_(expected, base_cost, 0.2, 1, empty(3, 1))

        assert_close(actual, expected)

    def test_update_does_not_increase_the_weighted_coordinate_objective(self) -> None:
        base_cost = torch.tensor(
            [[0.2, 0.7], [0.8, 0.1], [0.3, 0.4]],
            dtype=DTYPE,
            device=DEVICE,
        )
        weights = torch.tensor([0.1, 0.3, 0.6], dtype=DTYPE, device=DEVICE)
        cost = base_cost * weights[:, None]
        initial = torch.tensor(
            [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]],
            dtype=DTYPE,
            device=DEVICE,
        )
        updated = torch.empty_like(initial)
        temperature = 0.2

        _weighted_assign(updated, cost, temperature, weights)

        def objective(gamma: torch.Tensor) -> torch.Tensor:
            return (gamma * cost).sum() + temperature * (
                weights[:, None] * gamma * gamma.log()
            ).sum()

        assert objective(updated) <= objective(initial)

    @pytest.mark.parametrize("temperature", [0.2, 0.0], ids=["soft", "hard"])
    def test_zero_alloc(self, temperature) -> None:
        cost = rand(7, 4, g=generator(SEEDS[0]))
        weights = rand(7, g=generator(SEEDS[0] + 1))
        assert_zero_alloc(
            softmax_primitives.weighted_assign_simplex_,
            torch.empty_like(cost),
            cost,
            temperature,
            weights,
            empty(7, 1),
            torch.empty(7, 1, dtype=torch.int64, device=DEVICE),
        )
