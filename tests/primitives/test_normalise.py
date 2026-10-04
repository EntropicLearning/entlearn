"""Tests for the stochastic-matrix primitives (normalise.py)."""

import subprocess
import sys

import pytest
import torch
from _alloc import assert_no_float64, assert_zero_alloc

from entlearn.primitives.normalise import _is_soft, floored_log_, normalise_
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


class TestIsSoft:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_machine_precision_is_hard_and_the_next_float_is_soft(self, dtype):
        eps = torch.finfo(dtype).eps
        above = float(
            torch.nextafter(torch.tensor(eps, dtype=dtype), torch.tensor(1.0, dtype=dtype))
        )
        assert not _is_soft(0.0, dtype)
        assert not _is_soft(eps, dtype)
        assert _is_soft(above, dtype)
        assert _is_soft(float("inf"), dtype)


class TestFlooredLog:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_oracle_and_floors_zeros(self, seed):
        src = rand(6, 5, g=generator(seed))
        src[0, 0] = 0.0  # must hit the floor, not -inf
        out = torch.empty_like(src)
        floored_log_(out, src)
        eps = torch.finfo(src.dtype).eps
        assert torch.equal(out, torch.log(torch.clamp_min(src, eps)))

    def test_source_read_only_and_zero_alloc(self):
        src = rand(6, 5, g=generator(SEEDS[0]))
        out = torch.empty_like(src)
        src_c = src.clone()
        assert_zero_alloc(floored_log_, out, src)
        assert torch.equal(src, src_c)


class TestPrimitiveBoundary:
    def test_imports_do_not_load_optional_dependencies(self):
        code = """
import sys
import torch

# PyTorch can import NumPy itself. Record only modules added by entlearn.
before = set(sys.modules)
import entlearn
import entlearn.primitives.distance
import entlearn.primitives.encoding
import entlearn.primitives.manifold
import entlearn.primitives.normalise
import entlearn.primitives.softmax

forbidden = {"numpy", "scipy", "pandas", "sklearn"} & {
    name.partition(".")[0] for name in set(sys.modules) - before
}
raise SystemExit(f"optional dependencies imported: {sorted(forbidden)}" if forbidden else 0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code], check=False, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr or result.stdout

    def test_float32_normalise_creates_no_float64_tensor(self):
        prior = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            theta = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
            scratch = torch.empty(2, dtype=torch.float32)
            assert_no_float64(normalise_, theta, 0, scratch)
            assert theta.dtype is torch.float32
        finally:
            torch.set_default_dtype(prior)
