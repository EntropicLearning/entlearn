"""The plotting extra installs Plotly and nothing the core needs."""

import subprocess
import sys
from importlib.metadata import metadata, requires

import pytest
from packaging.requirements import Requirement


class TestPlottingDependencies:
    def test_plotting_extra_installs_plotly_without_changing_core(self):
        extras = metadata("entlearn").get_all("Provides-Extra")
        assert "plotting" in extras
        assert "visuals" not in extras
        requirements = [Requirement(value) for value in requires("entlearn") or ()]

        def active(extra):
            return {
                value.name
                for value in requirements
                if value.marker is None or value.marker.evaluate({"extra": extra})
            }

        frontends = {"plotly"}
        assert not active("") & frontends
        assert active("plotting") - active("") == frontends

    @pytest.mark.parametrize(
        "missing,operation",
        [
            ("plotly", "network.plot()"),
            ("plotly", "from entlearn.plotting import plot_theta"),
        ],
    )
    def test_missing_backends_recommend_the_same_extra(self, missing, operation):
        program = f"""
import importlib.abc
import sys
import torch
from entlearn import Input, Network, Recipe, RegressionHead

class MissingBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {missing!r} or fullname.startswith({missing!r} + "."):
            raise ModuleNotFoundError(name={missing!r})

sys.meta_path.insert(0, MissingBackend())
X = torch.tensor([[0.0], [0.1], [0.9], [1.0]], dtype=torch.float64)
network = Network.fit(Recipe.chain(Input(K=2), RegressionHead()), X, X, seed=7)
try:
    {operation}
except ModuleNotFoundError as error:
    assert "entlearn[plotting]" in str(error), str(error)
else:
    raise AssertionError("Missing backend did not produce an installation hint")
"""
        subprocess.run([sys.executable, "-c", program], check=True)
