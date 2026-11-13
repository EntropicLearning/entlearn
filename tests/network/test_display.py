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

from entlearn import ClassificationHead, Hidden, Input, Network, PredictConfig, Recipe

pytest.importorskip("rich")
from rich.console import Console


def _render(network: Network, width: int) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=width, color_system=None).print(network)
    return buffer.getvalue()


class TestRichDisplay:
    # The regression fit stops after one iteration, so a recorded warning is rendered too.
    @pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
    @pytest.mark.parametrize(
        ("task", "predict_config", "n_inits", "retain", "max_iter"),
        [
            pytest.param(
                "classification", PredictConfig(epsilon_P=0.5), 1, "winner", 5, id="supplied"
            ),
            pytest.param("classification", None, 3, "members", 5, id="multi-init"),
            pytest.param("regression", None, 1, "winner", 1, id="regression"),
        ],
    )
    def test_renders_the_fitted_facts_at_any_width(
        self, task, predict_config, n_inits, retain, max_iter
    ):
        if task == "classification":
            X, y = blobs(3, BLOB_SPREADS["separated"])
            recipe = classification_recipe(K=3)
        else:
            X, y, _ = regression_blobs(2)
            recipe = regression_recipe()
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
        facts = [name for name, _ in network.schema.K_active]
        facts += [network.schema.task, str(DTYPE).removeprefix("torch."), str(DEVICE)]
        if predict_config is not None:
            facts.append(str(predict_config.epsilon_P))
        for width in (60, 120):
            text = _render(network, width)
            for fact in facts:
                assert fact in text, f"{fact!r} missing at width {width}:\n{text}"

    @pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
    def test_a_ratio_shows_the_value_it_gives(self):
        X, y = blobs(3, BLOB_SPREADS["separated"])
        recipe = Recipe.chain(
            Input(K=4, epsilon=0.05),
            Hidden(K=3, epsilon_ratio=1.5),
            ClassificationHead(),
            delta=0.5,
            delta_ratio=4,
        )
        network = Network.fit(recipe, X, y, max_iter=2, seed=0)
        for width in (60, 120):
            text = _render(network, width)
            assert "ε=0.75 (1.5·δ)" in text
            assert "δ=2 (4·previous δ)" in text
