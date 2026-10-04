"""Run the Python examples on the guide pages against the installed package."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs_site"
# Pages may show backends that the development environment does not install.
OPTIONAL_BACKENDS = {"optuna", "optuna_integration"}
# Data a page's examples take from the reader instead of defining it.
PAGE_SETUP = {
    "guide/package.md": (
        "import torch\n"
        "X = torch.rand(40, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64)\n"
        "y = (X[:, 0] > 0.5).long()\n"
    ),
    "guide/estimators.md": (
        "import polars as pl\n"
        "query = pl.DataFrame({\n"
        '    "x": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],\n'
        '    "zone": ["west", "west", "east", "east", "west", "east"],\n'
        '    "label": ["low", "low", "low", "high", "high", "high"],\n'
        "}).lazy()\n"
    ),
    "guide/manifold.md": (
        "import numpy as np\n"
        "t = np.linspace(0, 1, 12)\n"
        "X = np.column_stack((t, 0.2 + 0.5 * t))\n"
        "y = t**2\n"
    ),
}


class TestScaffoldCode:
    @pytest.mark.parametrize(
        "page",
        [
            "guide/package.md",
            "guide/networks.md",
            "guide/estimators.md",
            "guide/hyperparameters.md",
            "guide/model_selection.md",
            "guide/manifold.md",
            "guide/installation.md",
            "guide/troubleshooting.md",
        ],
    )
    def test_python_examples_execute_in_page_order(
        self, page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contents = (DOCS_ROOT / page).read_text(encoding="utf-8")
        # Blocks nested in admonitions are indented; they run in page order too.
        fenced = re.findall(
            r"^( *)```python\n(.*?)^\1```", contents, flags=re.MULTILINE | re.DOTALL
        )
        blocks = [textwrap.dedent(code) for _, code in fenced]
        assert blocks, f"{page}: expected a runnable API example"
        monkeypatch.chdir(tmp_path)
        namespace = {"__name__": "__main__"}
        exec(PAGE_SETUP.get(page, ""), namespace)  # noqa: S102
        for index, code in enumerate(blocks, start=1):
            # Execute trusted repository examples.
            try:
                exec(compile(code, f"{page}:block-{index}", "exec"), namespace)  # noqa: S102
            except ModuleNotFoundError as error:
                if error.name not in OPTIONAL_BACKENDS:
                    raise
                break  # later blocks may use the missing backend
