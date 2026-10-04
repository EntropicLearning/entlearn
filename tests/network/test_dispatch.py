"""Seeded candidates run serially, on threads or on processes."""

import multiprocessing
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

from entlearn.network.dispatch import _map_seeds


def run(candidate, seeds, *, n_jobs=None, backend="serial"):
    """Return the backend used and every result, in seed order."""
    used, results = _map_seeds(candidate, tuple(seeds), n_jobs=n_jobs, backend=backend)
    return used, list(results)


class TestMapSeeds:
    @pytest.mark.parametrize(
        "backend,n_jobs", [("serial", None), ("threads", 2), ("threads", -1), ("processes", 2)]
    )
    def test_results_follow_seed_order_on_every_backend(self, backend, n_jobs):
        seeds = (17, 23, 41, 59)
        assert run(lambda seed: seed % 7, seeds, n_jobs=n_jobs, backend=backend) == (
            backend,
            [seed % 7 for seed in seeds],
        )

    def test_serial_results_are_computed_as_they_are_consumed(self):
        calls = []
        _, results = _map_seeds(
            lambda seed: calls.append(seed) or seed, (0, 1, 2), n_jobs=None, backend="serial"
        )
        assert calls == []
        assert next(results) == 0
        assert calls == [0]

    def test_completion_order_does_not_change_result_order(self):
        finished_second = threading.Event()
        completion_order = []

        def candidate(seed):
            if seed == 0:
                assert finished_second.wait(timeout=10)
            completion_order.append(seed)
            if seed == 1:
                finished_second.set()
            return seed

        assert run(candidate, (0, 1), n_jobs=2, backend="threads") == ("threads", [0, 1])
        assert completion_order == [1, 0]

    def test_serial_execution_does_not_import_joblib(self, without_joblib):
        assert run(lambda seed: seed, (3,)) == ("serial", [3])
        with pytest.raises(ImportError, match="joblib is unavailable"):
            run(lambda seed: seed, (3,), n_jobs=2, backend="threads")

    @pytest.mark.parametrize("backend,n_jobs", [("serial", None), ("threads", 2)])
    def test_a_candidate_failure_propagates(self, backend, n_jobs):
        def candidate(seed):
            if seed == 1:
                raise RuntimeError("candidate failed")
            return seed

        with pytest.raises(RuntimeError, match="candidate failed"):
            run(candidate, range(3), n_jobs=n_jobs, backend=backend)

    def test_process_request_inside_thread_uses_threads_with_a_warning(self):
        def outer():
            return run(lambda seed: seed, range(2), n_jobs=2, backend="processes")

        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            pytest.warns(UserWarning, match="processes must sit outside threads") as seen,
        ):
            result = pool.submit(outer).result(timeout=20)
        assert [record.filename for record in seen] == [__file__]
        assert result == ("threads", [0, 1])

    def test_process_request_inside_process_worker_warns_and_uses_threads(self):
        def outer(seed):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                used, results = _map_seeds(
                    lambda inner: multiprocessing.current_process().pid,
                    (0, 1, 2),
                    n_jobs=2,
                    backend="processes",
                )
                pids = set(results)
            messages = [str(warning.message) for warning in caught]
            return used, pids, multiprocessing.current_process().pid, messages

        _, results = run(outer, range(2), n_jobs=2, backend="processes")
        used, inner_pids, outer_pid, messages = results[0]
        assert outer_pid != multiprocessing.current_process().pid
        assert inner_pids == {outer_pid}
        assert used == "threads"
        assert any("processes must sit outside threads" in message for message in messages)

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_worker_warnings_reach_the_caller_in_seed_order(self, backend):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message="A worker stopped")
            run(_late_first_warning, range(4), n_jobs=4, backend=backend)
        assert [str(record.message) for record in seen] == [f"seed {seed}" for seed in range(4)]
        assert {record.category for record in seen} == {RuntimeWarning}
        assert {record.filename for record in seen} == {__file__}

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_a_worker_warning_as_error_raises_after_every_result(self, backend):
        consumed = []
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", message="A worker stopped")
            _, results = _map_seeds(_late_first_warning, (0, 1, 2, 3), n_jobs=4, backend=backend)
            with pytest.raises(RuntimeWarning, match="seed 0"):
                consumed.extend(results)
        assert consumed == [0, 1, 2, 3]

    def test_closing_early_ends_the_warning_window(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            hook = warnings.showwarning
            _, results = _map_seeds(_late_first_warning, (0, 1, 2, 3), n_jobs=4, backend="threads")
            assert next(results) == 0
            assert warnings.showwarning is not hook
            results.close()
            assert warnings.showwarning is hook
        assert str(seen[0].message) == "seed 0"

    def test_a_nested_threads_window_keeps_the_outermost_hold(self):
        hooks = []

        def inner(seed):
            hooks.append(warnings.showwarning)
            warnings.warn(f"inner {seed}", RuntimeWarning, stacklevel=1)
            return seed

        def outer(seed):
            hooks.append(warnings.showwarning)
            return run(inner, (seed,), n_jobs=2, backend="threads")[1]

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            hook = warnings.showwarning
            assert run(outer, (0, 1), n_jobs=2, backend="threads") == ("threads", [[0], [1]])
            assert warnings.showwarning is hook
        assert len(hooks) == 4
        assert len({id(held) for held in hooks}) == 1
        assert hooks[0] is not hook
        assert [str(record.message) for record in seen] == ["inner 0", "inner 1"]

    def test_a_threads_run_yields_each_result_before_later_seeds_finish(self):
        first_consumed = threading.Event()

        def candidate(seed):
            if seed == 1:
                assert first_consumed.wait(timeout=10)
            return seed

        _, results = _map_seeds(candidate, (0, 1), n_jobs=2, backend="threads")
        assert next(results) == 0
        first_consumed.set()
        assert list(results) == [1]

    def test_a_caller_warning_during_a_threads_run_waits_for_the_last_result(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            _, results = _map_seeds(lambda seed: seed, (0, 1), n_jobs=2, backend="threads")
            assert next(results) == 0
            warnings.warn("caller", RuntimeWarning, stacklevel=1)
            assert seen == []
            assert list(results) == [1]
        assert [str(record.message) for record in seen] == ["caller"]

    def test_a_caller_warning_during_a_processes_run_surfaces_at_once(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            _, results = _map_seeds(lambda seed: seed, (0, 1), n_jobs=2, backend="processes")
            assert next(results) == 0
            warnings.warn("caller", RuntimeWarning, stacklevel=1)
            assert "caller" in [str(record.message) for record in seen]
            assert list(results) == [1]


def _late_first_warning(seed):
    """Warn after a delay that makes later seeds finish first."""
    time.sleep(0.05 * (3 - seed))
    warnings.warn(f"seed {seed}", RuntimeWarning, stacklevel=1)
    return seed
