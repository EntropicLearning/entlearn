"""``tests/conftest.py``'s own device/dtype/depth resolution refuses an unhonourable
``EON_TEST_DEVICE``, ``EON_TEST_DTYPE`` or ``EON_TEST_DEPTH`` rather than silently falling
back to cpu/float64 at standard depth. The resolution runs once, at conftest import time, before any in-process
fixture could observe or intervene in it, so this is exercised out-of-process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND_VARIABLES = ("EON_TEST_DEVICE", "EON_TEST_DTYPE", "EON_TEST_DEPTH")


class TestConftestBackend:
    @pytest.mark.parametrize(
        ("request_", "fragments"),
        [
            pytest.param(
                {"EON_TEST_DEVICE": "not-a-device"},
                ("refusing to substitute",),
                id="device",
            ),
            pytest.param(
                {"EON_TEST_DTYPE": "float16"},
                ("refusing to ignore",),
                id="dtype",
            ),
            pytest.param(
                {"EON_TEST_DEVICE": "not-a-device", "EON_TEST_DTYPE": "float16"},
                ("refusing to substitute", "refusing to ignore"),
                id="device-and-dtype",
            ),
            pytest.param(
                {"EON_TEST_DEPTH": "deep"},
                ("is not a recognised depth",),
                id="depth",
            ),
            pytest.param(
                {"EON_TEST_DTYPE": "float16", "EON_TEST_DEPTH": "deep"},
                ("refusing to ignore", "is not a recognised depth"),
                id="dtype-and-depth",
            ),
        ],
    )
    def test_an_unhonourable_backend_request_aborts_the_session(self, request_, fragments):
        # "not-a-device" and "float16" are unrecognised on every machine. Unlike "cuda" a
        # device name never becomes honourable just because a future machine happens to
        # have a GPU (see the module docstring of tests/conftest.py), so these assertions
        # hold everywhere this test runs. Targeting tests/conftest.py itself is enough:
        # pytest loads it as a conftest before collecting anything, so the abort fires
        # without needing a throwaway test file. The lane this suite is itself running in
        # is cleared from the child's environment, so each case sees only its own request.
        repo_root = Path(__file__).resolve().parent.parent
        inherited = {
            key: value for key, value in os.environ.items() if key not in _BACKEND_VARIABLES
        }
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/conftest.py"],
            cwd=repo_root,
            env={**inherited, **request_},
            capture_output=True,
            text=True,
            check=False,  # a non-zero exit is exactly what this test asserts on
        )
        # Exit 4 is pytest's usage error, which is what a refusal raises. A bare "not zero"
        # would pass against the old silent-fallback conftest too: collecting conftest.py
        # yields no tests, so pytest exits 5 whatever the resolution decided. The message
        # fragments have to be the refusals themselves for the same reason, the fallback
        # warning also printed the device name. The device, dtype and depth refusals word
        # themselves differently, so each case names the wording it expects, and the
        # combined cases pin that both reach the report rather than one displacing the
        # other.
        output = result.stdout + result.stderr
        assert result.returncode == 4, output
        for fragment in fragments:
            assert fragment in output, output
