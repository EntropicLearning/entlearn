"""Tests for the stochastic-matrix primitives (normalise.py)."""

import math

import pytest
import torch
from _alloc import assert_no_float64, assert_zero_alloc
from conftest import DEVICE, DTYPE

from entlearn.primitives.normalise import _is_soft, _tiny, floored_log_, normalise_
from primitives._fixtures import SEEDS, assert_close, empty, generator, rand

SHAPES = [(3, 4), (5, 2), (4, 4), (2, 5)]  # (K_target, K_source)


def _ref(theta):
    """Naive column normalisation: per-column divide, empty -> 1/Kt."""
    Kt, Ks = theta.shape
    out = theta.clone()
    for ks in range(Ks):
        s = sum(float(theta[kt, ks]) for kt in range(Kt))
        for kt in range(Kt):
            out[kt, ks] = (float(theta[kt, ks]) / s) if s > 0.0 else 1.0 / Kt
    return out


class TestNormalise:
    """``dim=0`` normalises the columns, ``dim=1`` the rows."""

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_reference(self, dim, shape, seed):
        theta = rand(*shape, g=generator(seed))  # non-negative
        ref = _ref(theta) if dim == 0 else _ref(theta.T).T
        slices = shape[1 - dim]
        assert_zero_alloc(normalise_, theta, dim, empty(slices))
        assert_close(theta, ref)
        assert_close(theta.sum(dim), torch.ones(slices, dtype=theta.dtype))  # stochastic

    @pytest.mark.parametrize("dim", [0, 1])
    def test_empty_slice_uniform(self, dim):
        theta = rand(3, 4, g=generator(11))
        theta.select(1 - dim, 1).zero_()  # slice 1 carries no mass
        normalise_(theta, dim, empty(theta.shape[1 - dim]))
        width = theta.shape[dim]
        assert_close(theta.select(1 - dim, 1), torch.full((width,), 1.0 / width, dtype=theta.dtype))
        assert_close(theta.sum(dim), torch.ones(theta.shape[1 - dim], dtype=theta.dtype))

    @pytest.mark.parametrize("dim", [0, 1])
    def test_a_slice_with_mass_at_most_tiny_is_empty(self, dim):
        # The same rule as an empty centroid: a subnormal slice sum counts as zero.
        tiny = _tiny(DTYPE)
        theta = rand(3, 4, g=generator(12))
        theta.select(1 - dim, 1).zero_().select(0, 0).fill_(tiny)
        theta.select(1 - dim, 2).zero_().select(0, 0).fill_(2 * tiny)
        normalise_(theta, dim, empty(theta.shape[1 - dim]))
        width = theta.shape[dim]
        uniform = torch.full((width,), 1.0 / width, dtype=theta.dtype, device=DEVICE)
        one_hot = torch.zeros(width, dtype=theta.dtype, device=DEVICE)
        one_hot[0] = 1.0
        assert_close(theta.select(1 - dim, 1), uniform)
        assert_close(theta.select(1 - dim, 2), one_hot)


class TestIsSoft:
    def test_zero_is_hard_and_every_positive_temperature_is_soft(self):
        # The regime is a property of the setting alone, the same in every dtype.
        assert not _is_soft(0.0)
        for dtype in (torch.float32, torch.float64):
            info = torch.finfo(dtype)
            assert _is_soft(info.eps)
            assert _is_soft(info.tiny)
            assert _is_soft(info.tiny * info.eps)  # the smallest subnormal
        assert _is_soft(math.ulp(0.0))
        assert _is_soft(float("inf"))


class TestTiny:
    @pytest.mark.parametrize(("dtype", "exponent"), [(torch.float32, -126), (torch.float64, -1022)])
    def test_matches_the_smallest_normal_binary_value(self, dtype, exponent):
        assert _tiny(dtype) == math.ldexp(1.0, exponent)


class TestFlooredLog:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_oracle_and_floors_zeros(self, seed):
        src = rand(6, 5, g=generator(seed))
        src[0, 0] = 0.0  # must hit the floor, not -inf
        out = torch.empty_like(src)
        floored_log_(out, src)
        tiny = _tiny(src.dtype)
        assert torch.equal(out, torch.log(torch.clamp_min(src, tiny)))

    def test_positive_probabilities_keep_their_exact_log(self):
        info = torch.finfo(DTYPE)
        # Zero, the smallest normal, a value below eps, eps itself and an ordinary value
        values = [0.0, _tiny(DTYPE), info.eps / 1024, info.eps, 0.25]
        src = torch.tensor(values, dtype=DTYPE, device=DEVICE)
        out = torch.empty_like(src)

        floored_log_(out, src)

        expected = [math.log(_tiny(DTYPE))] + [math.log(v) for v in values[1:]]
        assert torch.isfinite(out).all()
        assert_close(out, torch.tensor(expected, dtype=DTYPE))

    def test_a_subnormal_probability_takes_the_floor(self):
        tiny = _tiny(DTYPE)
        src = torch.tensor([tiny / 16], dtype=DTYPE, device=DEVICE)
        if not bool(src > 0):
            pytest.skip("this device flushes subnormal values to zero")
        out = torch.empty_like(src)

        floored_log_(out, src)

        assert float(out) == pytest.approx(math.log(tiny), rel=2 * torch.finfo(DTYPE).eps, abs=0)

    def test_source_read_only_and_zero_alloc(self):
        src = rand(6, 5, g=generator(SEEDS[0]))
        out = torch.empty_like(src)
        src_c = src.clone()
        assert_zero_alloc(floored_log_, out, src)
        assert torch.equal(src, src_c)


class TestPrimitiveBoundary:
    def test_float32_normalise_creates_no_float64_tensor(self):
        prior = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            theta = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
            scratch = torch.empty(2, dtype=torch.float32)
            assert_no_float64(normalise_, theta, 0, scratch)
        finally:
            torch.set_default_dtype(prior)
