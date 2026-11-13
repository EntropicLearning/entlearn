"""Tests for the softmax / argmin / log-partition primitives (softmax.py)."""

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE

import entlearn.primitives.softmax as softmax_primitives
from entlearn.primitives.normalise import _tiny
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
    @pytest.mark.parametrize("shape,dim", [((7, 4), 1), ((6,), 0), ((4, 7), 0)])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, shape, dim, temp, seed):
        cost = randn(*shape, g=generator(seed))
        ref = _ref_softmax(cost, temp, dim)
        out = torch.empty_like(cost)
        assert_zero_alloc(
            softmax_with_temp_, out, cost, temp, dim, empty(*_keepdim_shape(shape, dim))
        )
        assert_close(out, ref)
        slice_sums = out.sum(dim)
        assert_close(slice_sums, torch.ones_like(slice_sums))  # stochastic along dim

    def test_degenerate_slice_uniform(self):
        shape, dim, temp = (5, 4), 1, 1.5
        cost = randn(*shape, g=generator(7))
        cost[2, :] = float("inf")  # whole slice is +inf -> uniform fallback
        out = torch.empty_like(cost)
        softmax_with_temp_(out, cost, temp, dim, empty(*_keepdim_shape(shape, dim)))
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
        scratch_min = empty(*_keepdim_shape(shape, dim))
        scratch_idx = torch.empty(_keepdim_shape(shape, dim), dtype=torch.int64, device=DEVICE)
        assert_zero_alloc(argmin_assign_, out, cost, dim, scratch_min, scratch_idx)
        assert torch.equal(out, ref)

    def test_tie_picks_first(self):
        shape, dim = (2, 4), 1
        cost = torch.tensor(
            [[1.0, -3.0, -3.0, 2.0], [0.0, 5.0, -1.0, -1.0]], dtype=DTYPE, device=DEVICE
        )  # row 0: tie at indices 1,2; row 1: tie at indices 2,3
        out = torch.empty_like(cost)
        scratch_min = empty(*_keepdim_shape(shape, dim))
        scratch_idx = torch.empty(_keepdim_shape(shape, dim), dtype=torch.int64, device=DEVICE)
        argmin_assign_(out, cost, dim, scratch_min, scratch_idx)
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
        out_log_sum, out_shift = empty(), empty()
        out_log_sum.fill_(-123.0)  # sentinels; must be overwritten
        out_shift.fill_(-123.0)
        scratch_T = empty(t)
        assert_zero_alloc(compute_log_partition_, out_log_sum, out_shift, b, temp, scratch_T)
        assert_close(out_shift, b.min())
        assert_close(out_log_sum - out_shift / temp, torch.tensor(ref, dtype=DTYPE))

    @pytest.mark.parametrize(
        "temp",
        [_tiny(DTYPE) * EPS, 2 * _tiny(DTYPE)],
        ids=["smallest-subnormal", "twice-tiny"],
    )
    def test_both_parts_stay_finite_where_the_log_partition_overflows(self, temp):
        # -min(b)/temp is far outside the dtype; the two tied minima still give log 2.
        b = torch.tensor([693.0, 693.0, 694.0], dtype=DTYPE, device=DEVICE)
        out_log_sum, out_shift, scratch_T = empty(), empty(), empty(3)
        compute_log_partition_(out_log_sum, out_shift, b, temp, scratch_T)
        assert_close(out_shift, torch.tensor(693.0, dtype=DTYPE, device=DEVICE))
        assert_close(out_log_sum, torch.tensor(math.log(2), dtype=DTYPE, device=DEVICE))


class TestAssignSimplex:
    @pytest.mark.parametrize("temperature", [EPS / 16, 2 * _tiny(DTYPE), _tiny(DTYPE) / 4])
    def test_positive_temperatures_resolve_costs_on_their_own_scale(self, temperature):
        cost = torch.tensor([[0.0, temperature]], dtype=DTYPE, device=DEVICE)
        out = torch.empty_like(cost)
        scratch = empty(1, 1)
        indices = torch.empty(1, 1, dtype=torch.int64, device=DEVICE)

        assert_zero_alloc(assign_simplex_, out, cost, temperature, 1, scratch, indices)

        p = 1.0 / (1.0 + math.exp(-1.0))
        expected = torch.tensor([[p, 1.0 - p]], dtype=DTYPE, device=DEVICE)
        assert_close(out, expected)

    @pytest.mark.parametrize("temperature", [2 * _tiny(DTYPE), _tiny(DTYPE) * EPS])
    def test_underflow_preserves_minima_and_ties_instead_of_a_uniform_fallback(self, temperature):
        cost = torch.tensor([[10.0, 20.0, 10.0], [30.0, 30.0, 30.0]], dtype=DTYPE, device=DEVICE)
        out = torch.empty_like(cost)
        scratch = empty(2, 1)
        indices = torch.empty(2, 1, dtype=torch.int64, device=DEVICE)

        assert_zero_alloc(assign_simplex_, out, cost, temperature, 1, scratch, indices)

        expected = torch.tensor(
            [[0.5, 0.0, 0.5], [1 / 3, 1 / 3, 1 / 3]], dtype=DTYPE, device=DEVICE
        )
        assert torch.isfinite(out).all()
        assert_close(out, expected)

    def _buffers(self, shape):
        cost = rand(*shape, g=generator(SEEDS[0]))
        out = torch.empty_like(cost)
        scratch = empty(shape[0], 1)
        idx = torch.empty(shape[0], 1, dtype=torch.int64, device=DEVICE)
        return cost, out, scratch, idx

    def test_soft_route_bitwise_matches_softmax(self):
        cost, out, scratch, idx = self._buffers((7, 4))
        expected = torch.empty_like(cost)
        assign_simplex_(out, cost, 0.37, 1, scratch, idx)
        softmax_with_temp_(expected, cost, 0.37, 1, scratch)
        assert torch.equal(out, expected)  # bitwise: same route, same ops

    def test_hard_route_bitwise_matches_argmin(self):
        cost, out, scratch, idx = self._buffers((7, 4))
        expected = torch.empty_like(cost)
        assign_simplex_(out, cost, 0.0, 1, scratch, idx)
        argmin_assign_(expected, cost, 1, scratch, idx)
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
    def test_a_hot_temperature_does_not_lose_a_finite_ratio_to_a_tiny_row_weight(self):
        info = torch.finfo(DTYPE)
        weight = _tiny(DTYPE) * info.eps  # smallest positive subnormal
        temperature = math.ldexp(1.0, 2 - math.frexp(_tiny(DTYPE))[1])
        gap = (temperature * weight) * 2
        weights = torch.tensor([weight, 1.0], dtype=DTYPE, device=DEVICE)
        cost = torch.tensor([[0.0, gap], [0.0, 0.0]], dtype=DTYPE, device=DEVICE)
        out = torch.empty_like(cost)
        indices = torch.empty(2, 1, dtype=torch.int64, device=DEVICE)

        assert_zero_alloc(
            softmax_primitives.weighted_assign_simplex_,
            out,
            cost,
            temperature,
            weights,
            empty(2, 1),
            indices,
        )

        p = 1 / (1 + math.exp(-2))
        expected = torch.tensor([[p, 1 - p], [0.5, 0.5]], dtype=DTYPE, device=DEVICE)
        assert_close(out, expected)

    def test_subnormal_row_temperatures_are_scaled_without_an_underflowed_reciprocal(self):
        temperature = 2 * _tiny(DTYPE)
        weights = torch.tensor([0.25, 0.5], dtype=DTYPE, device=DEVICE)
        cost = torch.tensor(
            [[0.0, temperature / 4], [0.0, temperature / 2]], dtype=DTYPE, device=DEVICE
        )
        before = cost.clone()
        out = torch.empty_like(cost)
        indices = torch.empty(2, 1, dtype=torch.int64, device=DEVICE)

        assert_zero_alloc(
            softmax_primitives.weighted_assign_simplex_,
            out,
            cost,
            temperature,
            weights,
            empty(2, 1),
            indices,
        )

        p = 1 / (1 + math.exp(-1))
        expected = torch.tensor([[p, 1 - p], [p, 1 - p]], dtype=DTYPE, device=DEVICE)
        assert_close(out, expected)
        assert torch.equal(cost, before)

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
        softmax_with_temp_(expected, base_cost, 0.2, 1, empty(base_cost.shape[0], 1))

        assert_close(actual, expected)

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
