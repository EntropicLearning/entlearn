"""Zero-allocation test probes for the allocation-free hot-path kernels.

Two complementary probes, because Python/PyTorch has no single cheap hook that
catches every tensor allocation:

* ``count_allocations``: a device-independent tracer that monkeypatches the
  named ``torch`` factory functions and counts calls. It does not catch
  arithmetic temporaries (``a + b``, ``a @ b``, ``softmax(...)`` without
  ``out=``), which route through ``aten`` kernels rather than the named
  factories.
* the CUDA branch of ``assert_zero_alloc``, a CUDA ``memory_allocated`` delta
  probe. It sees allocations by any route, named factory or ``aten`` kernel,
  but only those that outlive the call: a temporary released before the call
  returns leaves the counter where it was, and only runs when CUDA is available.

On CPU the real guard is the coding discipline, every hot-path kernel takes
``out=`` / ``scratch_*`` buffers and writes in place, with these probes
catching regressions the moment they appear.
"""

from __future__ import annotations

import contextlib
import functools
import gc

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Storage-allocating ``torch`` factory functions that could plausibly appear in
# our hot-path code. Keep in sync with the in-place coding discipline.
_TRACED_FACTORIES = (
    "empty",
    "zeros",
    "ones",
    "full",
    "tensor",
    "rand",
    "randn",
    "randint",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand_like",
    "randn_like",
    "clone",
    "cat",
    "stack",
    "arange",
    "linspace",
    "eye",
)


@contextlib.contextmanager
def count_allocations():
    """Count calls to storage-allocating ``torch`` factory functions.

    Yields a ``dict`` mapping each traced factory name to its call count during
    the ``with`` block; restores the originals on exit.
    """
    counts = {name: 0 for name in _TRACED_FACTORIES}
    originals = {name: getattr(torch, name) for name in _TRACED_FACTORIES}

    def make_wrapper(name, orig):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return orig(*args, **kwargs)

        return wrapper

    try:
        for name in _TRACED_FACTORIES:
            setattr(torch, name, make_wrapper(name, originals[name]))
        yield counts
    finally:
        for name, orig in originals.items():
            setattr(torch, name, orig)


@contextlib.contextmanager
def _no_cyclic_gc():
    """Hold Python's cyclic collector off, restoring its prior state on exit.

    ``torch.cuda.memory_allocated()`` is a process-global counter, so anything that
    releases CUDA storage between the two readings is charged to the call under test.
    A cyclic collection is the usual culprit: it runs at an arbitrary allocation count,
    and reclaiming a CUDA tensor stranded in a reference cycle by an earlier test makes
    the delta negative, a failure with no connection to the call being measured (which
    made these probes order-dependent, and a CPU-only kernel test fail purely because a
    GPU test ran before it). Freezing the collector across the window is what keeps the
    delta attributable; ``gc.disable()`` suppresses the automatic collections, which are
    the ones that fire unpredictably.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@functools.cache
def _warm_cuda() -> None:
    """Pay PyTorch's one-time cuBLAS workspace before any delta is measured.

    The first matrix multiplication in a CUDA process allocates a multi-megabyte
    cuBLAS workspace through the caching allocator, and
    ``torch.cuda.memory_allocated()`` counts it. Without this, whichever probed
    kernel a selection happens to reach first is charged the whole workspace and
    fails however allocation-free it is, so the failure follows collection order
    rather than the kernel. Issuing the multiplication once per process, outside
    any measured window, is what keeps the delta attributable, for the same reason
    :func:`_no_cyclic_gc` exists.
    """
    if torch.cuda.is_available():
        warm = torch.ones(8, 8, device="cuda")
        warm @ warm
        torch.cuda.synchronize()


def assert_zero_alloc(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` once and assert it makes no new tensor storage.

    Wraps a single call in both probes: the factory tracer (device-independent)
    and, when CUDA is available, the ``memory_allocated`` delta, taken under
    :func:`_no_cyclic_gc` so an unrelated collection cannot be charged to this
    call. Returns ``fn``'s result. ``fn`` is called exactly once, so it is safe
    for accumulating kernels (``cost += ...``).
    """
    cuda = torch.cuda.is_available()
    if cuda:
        _warm_cuda()
    before = after = 0
    with _no_cyclic_gc():
        if cuda:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
        with count_allocations() as counts:
            result = fn(*args, **kwargs)
        if cuda:
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated()
    offending = {k: v for k, v in counts.items() if v}
    assert not offending, f"factory-function allocations: {offending}"
    assert after == before, f"allocated {after - before} CUDA bytes"
    return result


def _tensors_in(value):
    """Yield tensors contained in a dispatcher result."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors_in(item)


class _RejectFloat64(TorchDispatchMode):
    """Reject an operation that produces a float64 tensor."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        offenders = [
            tensor.shape for tensor in _tensors_in(result) if tensor.dtype is torch.float64
        ]
        assert not offenders, f"{func} produced float64 tensors: {offenders}"
        return result


def assert_no_float64(fn, *args, **kwargs):
    """Run ``fn`` and reject every float64 tensor produced by its operations."""
    with _RejectFloat64():
        return fn(*args, **kwargs)
