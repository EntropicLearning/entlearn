"""Finite probability-log behaviour across the public fitted-model lifecycle."""

import math

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET

from entlearn import ClassificationHead, Coupling, Input, Network, PredictConfig, Recipe
from entlearn.primitives.normalise import _tiny


@FIXED_BUDGET
class TestProbabilityLogs:
    @pytest.mark.parametrize(
        "epsilon_T", [2 * _tiny(DTYPE), _tiny(DTYPE) / 2], ids=["tiny", "subnormal"]
    )
    def test_a_tiny_instance_weight_temperature_keeps_an_exact_normaliser(self, epsilon_T):
        X = torch.zeros(2, 1, dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=1, epsilon_T=epsilon_T, delta_cat=1e3),
            ClassificationHead(coupling=Coupling.M),
        )
        # A mixed categorical centroid gives both rows the same large finite cost, so
        # -cost/epsilon_T overflows while the training softmax is exactly uniform.
        model = Network.fit(recipe, X, y, X_cat=(y,), max_iter=1)
        result = model.predict_with_details(X, X_cat=(y,), details=("instance_weights",))
        half = torch.full((2,), 0.5, dtype=DTYPE, device=DEVICE)
        assert result.instance_weights is not None
        torch.testing.assert_close(result.instance_weights, half)

    @pytest.mark.parametrize("categorical", [False, True])
    def test_sparse_soft_models_predict_and_round_trip_without_non_finite_values(
        self, tmp_path, categorical
    ):
        X = torch.tensor([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]], dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=DEVICE)
        categories = (torch.nn.functional.one_hot(y, 2).to(DTYPE),) if categorical else ()
        recipe = Recipe.chain(Input(K=2, epsilon=0.05), ClassificationHead(coupling=Coupling.M))
        model = Network.fit(recipe, X, y, X_cat=categories, max_iter=8, tol=0, seed=7)
        path = tmp_path / "sparse.safetensors"
        model.save(path, resumable=True)
        restored = Network.load(path, device=DEVICE)
        continued = restored.resume(X, y, X_cat=categories, max_iter=9, tol=0)

        for policy in (
            model.predict_config,
            PredictConfig(predict_mode="iterative", max_iter=3, tol=0),
        ):
            expected = model.predict(X, X_cat=categories, predict_config=policy)
            assert torch.isfinite(expected).all()
            assert torch.equal(expected.argmax(dim=1), y)
            for fitted in (restored, continued):
                actual = fitted.predict(X, X_cat=categories, predict_config=policy)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected)
        assert all(math.isfinite(value) for value in continued.diagnostics.loss_history)

    @pytest.mark.parametrize("weight_temperature", [0.0, math.inf])
    def test_zero_weight_channels_and_sparse_levels_remain_inactive(self, weight_temperature):
        X = torch.tensor([[0.0], [0.1], [0.9], [1.0]], dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=DEVICE)
        categories = (torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=DEVICE),)
        recipe = Recipe.chain(
            Input(
                K=2,
                epsilon_D=weight_temperature,
                epsilon_T=weight_temperature,
                delta_cat=0.0,
            ),
            ClassificationHead(coupling=Coupling.M),
        )

        model = Network.fit(recipe, X, y, X_cat=categories, max_iter=3, tol=0, seed=7)
        result = model.predict_with_details(X, X_cat=categories, details=("affiliations",))

        assert torch.isfinite(result.prediction).all()
        assert all(math.isfinite(value) for value in model.diagnostics.loss_history)
        assert result.affiliations is not None
        assert all(torch.isfinite(value).all() for value in result.affiliations.values())
