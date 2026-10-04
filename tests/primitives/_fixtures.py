"""Shared fixtures for the primitive test modules: seeds, tolerances, builders.

Every builder creates tensors on the run's device and dtype (``conftest.DEVICE`` /
``conftest.DTYPE``), so a module that builds its inputs here runs on every backend.
"""

from __future__ import annotations

import random

import torch
from conftest import DEVICE, DTYPE, by_depth

# The fixed RNG makes each depth's seeds a prefix of the next depth's. A test that needs
# one seed reads ``SEEDS[0]`` and takes a second stream from ``SEEDS[0] + 1``, so every
# depth supplies it.
N_REPS = by_depth(minimal=1, standard=2, exhaustive=8)
_SEED_RNG = random.Random(42)
SEEDS = [_SEED_RNG.randrange(2**31) for _ in range(N_REPS)]

# Reference-comparison tolerances for the run dtype.
RTOL = {torch.float64: 1e-12, torch.float32: 1e-5}[DTYPE]
ATOL = {torch.float64: 1e-14, torch.float32: 1e-6}[DTYPE]


def assert_close(actual, expected):
    """Compare against a reference at the run dtype's tolerance."""
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL, check_device=False)


def generator(seed):
    """Return a seeded generator on the test device."""
    return torch.Generator(device=DEVICE).manual_seed(seed)


def rand(*shape, g):
    return torch.rand(shape, generator=g, dtype=DTYPE, device=DEVICE)


def randn(*shape, g):
    return torch.randn(shape, generator=g, dtype=DTYPE, device=DEVICE)


def empty(*shape):
    return torch.empty(shape, dtype=DTYPE, device=DEVICE)


def stochastic(*shape, dim, g):
    """Random tensor whose slices along ``dim`` sum to one."""
    x = rand(*shape, g=g)
    x /= x.sum(dim, keepdim=True)
    return x
