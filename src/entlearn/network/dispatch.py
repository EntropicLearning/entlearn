"""Seeded candidates run serially, on threads or on processes, with results in seed order."""

from __future__ import annotations

import multiprocessing
import threading
import warnings
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from typing import Literal, TextIO

from entlearn._warnings import _warn

_Backend = Literal["serial", "threads", "processes"]
type _HeldWarning = tuple[type[Warning], str]
type _HeldWarnings = list[_HeldWarning]

# The warnings raised by the candidate running in this thread, while one is running.
_candidate_warnings: ContextVar[_HeldWarnings | None] = ContextVar(
    "_candidate_warnings", default=None
)
_window_lock = threading.Lock()


def _hold(
    fallback: _HeldWarnings,
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Keep a warning with the candidate running on this thread, else with ``fallback``."""
    held = _candidate_warnings.get()
    (fallback if held is None else held).append((category, str(message)))


@contextmanager
def _holding_warnings(fallback: _HeldWarnings) -> Iterator[None]:
    """Hold back every warning in this process, from any thread, until the outermost block exits."""
    catcher = warnings.catch_warnings()
    with _window_lock:
        hook = warnings.showwarning
        opened = not (isinstance(hook, partial) and hook.func is _hold)
        if opened:
            catcher.__enter__()
            warnings.simplefilter("always")
            warnings.showwarning = partial(_hold, fallback)
    try:
        yield
    finally:
        if opened:
            with _window_lock:
                catcher.__exit__(None, None, None)


def _run_candidate[R](
    candidate: Callable[[int], R], seed: int, *, own_process: bool
) -> tuple[R, tuple[_HeldWarning, ...]]:
    """Run one candidate in a worker and return its result with the warnings it raised."""
    held: _HeldWarnings = []
    token = _candidate_warnings.set(held)
    try:
        with _holding_warnings(held) if own_process else nullcontext():
            result = candidate(seed)
    finally:
        _candidate_warnings.reset(token)
    return result, tuple(held)


def _map_seeds[R](
    candidate: Callable[[int], R],
    seeds: tuple[int, ...],
    *,
    n_jobs: int | None,
    backend: _Backend,
) -> tuple[_Backend, Generator[R]]:
    """Return the backend used and a lazy generator of ``candidate(seed)`` in seed order.

    Processes requested inside a thread or worker fall back to threads with a warning.
    """
    process = multiprocessing.current_process()
    if backend == "processes" and (
        process.daemon
        or process.name != "MainProcess"
        or threading.current_thread() is not threading.main_thread()
    ):
        _warn(
            "parallel_backend='processes' was requested inside a worker; using "
            "threads instead - processes must sit outside threads, never nested",
            UserWarning,
        )
        backend = "threads"
    return backend, _results(candidate, seeds, n_jobs=n_jobs, backend=backend)


def _results[R](
    candidate: Callable[[int], R],
    seeds: tuple[int, ...],
    *,
    n_jobs: int | None,
    backend: _Backend,
) -> Generator[R]:
    """Yield results in seed order; parallel warnings are raised when the generator finishes."""
    if backend == "serial":
        yield from map(candidate, seeds)
        return
    from joblib import Parallel, delayed

    held: _HeldWarnings = []
    try:
        with _holding_warnings(held) if backend == "threads" else nullcontext():
            for result, raised in Parallel(
                n_jobs=n_jobs,
                backend="threading" if backend == "threads" else "loky",
                return_as="generator",
            )(
                delayed(_run_candidate)(candidate, seed, own_process=backend == "processes")
                for seed in seeds
            ):
                held.extend(raised)
                yield result
    finally:
        # Workers hold their warnings back; raise them here, on the caller's line.
        for category, message in held:
            _warn(message, category)
