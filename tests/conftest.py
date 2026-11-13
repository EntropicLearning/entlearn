"""Pytest configuration for the tensor primitive suites."""

from __future__ import annotations

import os
import re
import warnings

import pytest
import torch

pytest_plugins = ("_fixtures",)

pytest.register_assert_rewrite("_alloc")

# A match= that spells out at least WHOLE_SHARE of a message of MIN_WORDS or more words
# fails: it should name the offending value and the reason, so the wording can change.
WHOLE_SHARE = 0.9
MIN_WORDS = 6

try:
    from _pytest.raises import AbstractRaises, stringify_exception
    from _pytest.recwarn import WarningsChecker

    _check_match, _matches = AbstractRaises._check_match, WarningsChecker.matches
except (ImportError, AttributeError) as error:
    raise ImportError(
        f"pytest {pytest.__version__} lacks the match hooks the whole-message guard in "
        "tests/conftest.py wraps (written for pytest 9.1.1); update the guard"
    ) from error


def _literal_length(pattern: str) -> int:
    """Count the characters a pattern spells out, ignoring regex syntax and wildcards."""
    literal = re.sub(r"\\([^A-Za-z0-9])", r"\1", pattern)
    return len(re.sub(r"\.\*\??|\.\+\??|\\S\+|\\s\+|\\d\+|\[[^]]*\][*+?]?|[\^$]", "", literal))


def _require_partial_match(pattern: str | re.Pattern[str], message: str) -> None:
    """Fail when the pattern spells out at least 90 % of a message of six or more words."""
    pattern = getattr(pattern, "pattern", pattern)
    words = len(re.findall(r"[A-Za-z]{2,}", message))
    if words >= MIN_WORDS and _literal_length(pattern) / len(message) >= WHOLE_SHARE:
        pytest.fail(
            "match= spells out the whole message; match the offending value and the reason\n"
            f"  pattern: {pattern!r}\n  message: {message!r}",
            pytrace=False,
        )


def _checked_raise(self: AbstractRaises, e: BaseException) -> bool:
    matched = _check_match(self, e)
    if matched and self.match is not None:
        _require_partial_match(self.match, stringify_exception(e, include_subexception_msg=False))
    return matched


def _checked_warning(self: WarningsChecker, warning: warnings.WarningMessage) -> bool:
    matched = _matches(self, warning)
    if matched and self.match_expr is not None:
        _require_partial_match(self.match_expr, str(warning.message))
    return matched


_guard = pytest.MonkeyPatch()
_guard.setattr(AbstractRaises, "_check_match", _checked_raise)
_guard.setattr(WarningsChecker, "matches", _checked_warning)

_DTYPE_FOR_DEVICE: dict[str, torch.dtype] = {
    "cpu": torch.float64,
    "cuda": torch.float32,
}


def _resolve_device_dtype() -> tuple[torch.device, torch.dtype, str | None]:
    """Resolve the requested test device and dtype."""
    requested = os.environ.get("EON_TEST_DEVICE", "cpu").strip().lower()
    note: str | None = None
    if requested == "cuda" and not torch.cuda.is_available():
        note = (
            "EON_TEST_DEVICE=cuda requested but CUDA is unavailable on this machine — refusing "
            "to substitute cpu/float64 for it; rerun on a CUDA-enabled machine, or unset "
            "EON_TEST_DEVICE (or set it to cpu) to run the cpu lane on purpose"
        )
        requested = "cpu"
    elif requested not in _DTYPE_FOR_DEVICE:
        note = (
            f"EON_TEST_DEVICE={requested!r} is not a recognised device — refusing to "
            f"substitute cpu/float64 for it; use one of {sorted(_DTYPE_FOR_DEVICE)}"
        )
        requested = "cpu"

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if requested == "cuda"
        else torch.device(requested)
    )
    dtype = _DTYPE_FOR_DEVICE[requested]
    dtype_req = os.environ.get("EON_TEST_DTYPE", "").strip().lower()
    if dtype_req:
        override = {"float32": torch.float32, "float64": torch.float64}.get(dtype_req)
        if override is None:
            extra = (
                f"EON_TEST_DTYPE={dtype_req!r} is not a recognised dtype — refusing to ignore "
                "it; use float32 or float64, or unset EON_TEST_DTYPE"
            )
            note = f"{note}; {extra}" if note else extra
        else:
            dtype = override
    return device, dtype, note


_DEPTHS = ("minimal", "standard", "exhaustive")


def _resolve_depth() -> tuple[str, str | None]:
    """Resolve the requested test depth, which scales the seed replicates."""
    requested = os.environ.get("EON_TEST_DEPTH", "").strip().lower() or "standard"
    if requested in _DEPTHS:
        return requested, None
    note = (
        f"EON_TEST_DEPTH={requested!r} is not a recognised depth — refusing to substitute "
        f"standard for it; use one of {list(_DEPTHS)}, or unset EON_TEST_DEPTH"
    )
    return "standard", note


DEVICE, DTYPE, _BACKEND_NOTE = _resolve_device_dtype()
DEPTH, _DEPTH_NOTE = _resolve_depth()
_FALLBACK_NOTE = "; ".join(note for note in (_BACKEND_NOTE, _DEPTH_NOTE) if note) or None

# A fit with a fixed iteration budget (tol=0 or a short max_iter) is not meant to converge.
FIXED_BUDGET = pytest.mark.filterwarnings("ignore:the solver:entlearn.ConvergenceWarning")
# A small manifold fixture can start a cluster below its subspace rank; its basis is completed.
RANK_DEFICIENT = pytest.mark.filterwarnings("ignore:manifold initialisation:UserWarning")


def by_depth[T](minimal: T, standard: T, exhaustive: T) -> T:
    """Return the value for the active ``EON_TEST_DEPTH``; CI runs at ``standard``.

    A seed axis keeps each smaller depth's seeds among the larger depths' seeds, so a
    test ID names the same case at every depth and an exhaustive failure reproduces by ID.
    """
    return {"minimal": minimal, "standard": standard, "exhaustive": exhaustive}[DEPTH]


# Fits repeat bitwise run to run on every device without PyTorch's deterministic
# algorithms, so the trajectory-identity contracts (resume, persistence, transport) run in
# the default mode; the environment flag runs them under deterministic algorithms instead.
_DETERMINISTIC = os.environ.get("EON_TEST_DETERMINISTIC", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@pytest.fixture(autouse=True, scope="session")
def _enforce_determinism():
    """Enable deterministic PyTorch operations when requested."""
    if not _DETERMINISTIC:
        yield
        return

    prior = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prior)


def pytest_report_header() -> list[str]:
    """Report the active test backend and depth."""
    lines = [f"EON test backend: device={DEVICE}, dtype={DTYPE}, depth={DEPTH}"]
    if _FALLBACK_NOTE is not None:
        lines.append(f"EON test backend: {_FALLBACK_NOTE}")
    if _DETERMINISTIC:
        lines.append("EON test backend: deterministic algorithms ENFORCED (tripwire)")
    return lines


def pytest_configure(config) -> None:
    """Reject a test backend or depth that cannot be honoured."""
    if _FALLBACK_NOTE is not None:
        raise pytest.UsageError(_FALLBACK_NOTE)
