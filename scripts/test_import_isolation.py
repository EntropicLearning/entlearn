"""Ordinary imports do not load call-time optional dependencies."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

OPTIONAL_MODULES = {"joblib", "matplotlib", "numpy", "pandas", "plotly", "scipy", "sklearn"}


def run_python_probe(source: str) -> object:
    """Run a fresh Python process and decode its JSON result."""
    result = subprocess.run(
        [sys.executable, "-c", source], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


class TestImportIsolation:
    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param("import entlearn", id="package"),
            pytest.param("from entlearn import Network", id="network"),
            pytest.param("import entlearn.primitives", id="primitives"),
            pytest.param("import entlearn.helpers", id="helper-namespace"),
        ],
    )
    def test_package_imports_do_not_load_additional_optional_dependencies(
        self, statement: str
    ) -> None:
        loaded = run_python_probe(
            f"""
import builtins
import json
import torch
optional = {OPTIONAL_MODULES!r}
requested = set()
original_import = builtins.__import__
def tracking_import(name, *args, **kwargs):
    root = name.partition('.')[0]
    if root in optional:
        requested.add(root)
    return original_import(name, *args, **kwargs)
builtins.__import__ = tracking_import
{statement}
print(json.dumps(sorted(requested)))
"""
        )

        assert loaded == []

    def test_network_import_does_not_load_optional_helpers(self):
        loaded = run_python_probe(
            """
import json
import sys
from entlearn import Network
print(json.dumps(sorted(
    name for name in sys.modules
    if name == "entlearn.helpers" or name.startswith("entlearn.helpers.")
)))
"""
        )

        assert loaded == []

    @pytest.mark.parametrize(
        ("helper_call", "required"),
        [
            pytest.param(
                """
feature_weights(
    [[0.0], [1.0], [2.0], [3.0]],
    ([0, 0, 1, 1],),
    [0, 0, 1, 1],
    task="classification",
    method="correlation",
)
""",
                {"scipy"},
                id="correlation",
            ),
            pytest.param(
                """
feature_weights(
    [[0.0], [1.0], [2.0], [3.0]],
    ([0, 0, 1, 1],),
    [0, 0, 1, 1],
    task="classification",
    method="mutual_info",
    random_state=0,
)
""",
                {"sklearn"},
                id="mutual-information",
            ),
        ],
    )
    def test_optional_dependencies_load_only_when_a_helper_is_called(
        self,
        helper_call: str,
        required: set[str],
    ) -> None:
        result = run_python_probe(
            f"""
import json
import sys
import torch
from entlearn.helpers import feature_weights
optional = {OPTIONAL_MODULES!r}
before = set(sys.modules)
{helper_call}
loaded = {{name.partition('.')[0] for name in set(sys.modules) - before}} & optional
print(json.dumps(sorted(loaded)))
"""
        )

        assert required <= set(result)
