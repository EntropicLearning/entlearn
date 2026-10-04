"""Rendering smoke test for the optional Rich display of a fitted Network."""

import io

import pytest
from conftest import DEVICE, DTYPE
from network._fixtures import (
    BLOB_SPREADS,
    blobs,
    classification_recipe,
    regression_blobs,
    regression_recipe,
)

from entlearn import Network, PredictConfig

pytest.importorskip("rich")
from rich.console import Console

# Each width selects one table layout, which its column heading identifies.
_LAYOUTS = ((72, "Configuration"), (120, "Hyperparameters"))


def _render(network: Network, width: int) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=width, color_system=None).print(network)
    return buffer.getvalue()


class TestRichDisplay:
    # One fit stops after a single iteration, so both fit statuses are rendered.
    @pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
    @pytest.mark.parametrize(
        ("task", "predict_config", "n_inits", "retain", "max_iter", "multi_init", "calibration"),
        [
            pytest.param(
                "classification",
                None,
                1,
                "winner",
                5,
                "1 candidate · members not retained",
                "in-sample",
                id="derived",
            ),
            pytest.param(
                "classification",
                PredictConfig(epsilon_P=0.5),
                1,
                "winner",
                5,
                "1 candidate · members not retained",
                "not performed (supplied ε_P)",
                id="supplied",
            ),
            pytest.param(
                "classification",
                None,
                3,
                "members",
                5,
                "3 candidates · 3 members retained",
                "in-sample",
                id="multi-init",
            ),
            pytest.param(
                "classification",
                None,
                1,
                "members",
                5,
                "1 candidate · 1 member retained",
                "in-sample",
                id="single-member-retained",
            ),
            pytest.param(
                "regression",
                None,
                1,
                "winner",
                1,
                "1 candidate · members not retained",
                "not applicable",
                id="regression",
            ),
        ],
    )
    def test_renders_the_fit_summary_in_both_layouts(
        self, task, predict_config, n_inits, retain, max_iter, multi_init, calibration
    ):
        if task == "classification":
            X, y = blobs(3, BLOB_SPREADS["separated"])
            recipe, data = classification_recipe(K=3), "3 continuous · 0 categorical · 3 classes"
        else:
            X, y, _ = regression_blobs(2)
            recipe, data = regression_recipe(), "1 continuous · 0 categorical · 2 outputs"
        network = Network.fit(
            recipe,
            X,
            y,
            n_inits=n_inits,
            retain=retain,
            predict_config=predict_config,
            max_iter=max_iter,
            seed=0,
        )
        status = "Converged" if network.diagnostics.converged else "Not converged"
        compute = f"{str(DTYPE).removeprefix('torch.')} · {DEVICE}"
        for width, heading in _LAYOUTS:
            text = _render(network, width)
            for expected in (heading, status, data, compute, multi_init, calibration):
                assert expected in text, f"{expected!r} missing at width {width}:\n{text}"
