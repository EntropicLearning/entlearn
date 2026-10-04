"""Smoke tests for the zero-allocation probes (``tests/_alloc.py``).

These prove the harness reliably distinguishes allocating code from in-place
code, so the per-primitive zero-alloc tests can rely on it.
"""

import gc

import pytest
import torch
from _alloc import assert_zero_alloc, count_allocations


class TestCountAllocations:
    def test_detects_factory_call(self):
        with count_allocations() as counts:
            _ = torch.zeros(4, 4)
        assert counts["zeros"] == 1
        assert sum(counts.values()) == 1

    def test_zero_for_inplace_and_out_kwarg(self):
        out = torch.zeros(4, 4)
        a = torch.ones(4, 4)
        with count_allocations() as counts:
            out.add_(a)
            out.mul_(2.0)
            torch.add(a, a, out=out)  # out= form: writes into existing storage
        assert sum(counts.values()) == 0

    def test_restores_originals(self):
        real_zeros = torch.zeros
        with count_allocations():
            assert torch.zeros is not real_zeros  # patched inside
        assert torch.zeros is real_zeros  # restored outside


class TestAssertZeroAlloc:
    def test_passes_for_inplace_kernel(self):
        out = torch.zeros(4, 4)
        a = torch.ones(4, 4)
        assert_zero_alloc(lambda: out.add_(a))

    def test_raises_on_factory_allocation(self):
        with pytest.raises(AssertionError):
            assert_zero_alloc(lambda: torch.zeros(4, 4))


class TestCyclicGcGuard:
    """The CUDA delta is only attributable if no unrelated collection runs inside the window.

    ``torch.cuda.memory_allocated()`` is process-global, so a cyclic collection reclaiming a
    CUDA tensor stranded by an earlier test lands in this call's delta as a spurious negative
    -- which made the probes order-dependent (a CPU-only kernel test could fail purely because
    a GPU test ran before it). The probe holds the collector off across the window instead.
    """

    def test_cyclic_gc_is_held_off_during_the_call(self):
        seen = {}
        assert_zero_alloc(lambda: seen.setdefault("enabled", gc.isenabled()))
        assert seen["enabled"] is False

    def test_enabled_collector_is_restored(self):
        gc.enable()
        assert_zero_alloc(lambda: None)
        assert gc.isenabled()

    def test_disabled_collector_stays_disabled(self):
        gc.disable()
        try:
            assert_zero_alloc(lambda: None)
            assert not gc.isenabled()
        finally:
            gc.enable()
