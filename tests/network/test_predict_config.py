"""Prediction policy through the public Network lifecycle."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from network._fixtures import blobs, classification_recipe, regression_recipe

import entlearn
from entlearn import ClassificationHead, Coupling, Input, Network, PredictConfig, Recipe


class TestPredictConfig:
    def test_supplied_temperature_and_complete_override(self):
        X, y = blobs(2, 0.28, seed=3)
        config = entlearn.PredictConfig(epsilon_P=0.5)
        model = Network.fit(
            classification_recipe(Input(K=3, epsilon=0.2)),
            X,
            y,
            predict_config=config,
            max_iter=3,
            tol=0,
        )
        assert model.predict_config.epsilon_P == 0.5
        assert model.predict_config.output_mode == "geometric"
        assert model.predict_config.predict_mode == "single"
        assert model.predict_config.max_iter == 100
        assert model.predict_config.tol == 1e-4
        with pytest.raises(FrozenInstanceError):
            model.predict_config.epsilon_P = 1.0
        with pytest.raises(AttributeError):
            model.predict_config = config

        uncalibrated = model.predict(X, predict_config=entlearn.PredictConfig())
        explicit = model.predict(X, predict_config=replace(config, epsilon_P=1.0))
        torch.testing.assert_close(uncalibrated, explicit)
        torch.testing.assert_close(
            model.predict(X), uncalibrated.square() / uncalibrated.square().sum(1, keepdim=True)
        )

    @pytest.mark.parametrize("epsilon", [-1, float("nan"), float("inf"), True, "1", 10**1000])
    def test_invalid_temperature_is_rejected(self, epsilon):
        with pytest.raises(ValueError, match="epsilon_P"):
            PredictConfig(epsilon_P=epsilon)

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"output_mode": "arithmetic", "epsilon_P": 0}, "epsilon_P"),
            ({"predict_mode": "single_pass"}, "predict_mode"),
            ({"output_mode": "unknown"}, "output_mode"),
            ({"tol": float("nan")}, "tol"),
            ({"tol": -1}, "tol"),
            ({"max_iter": 0}, "max_iter"),
            ({"max_iter": True}, "max_iter"),
            ({"max_iter": 1.5}, "max_iter"),
        ],
    )
    def test_invalid_controls_are_rejected(self, fields, message):
        with pytest.raises(ValueError, match=message):
            PredictConfig(**fields)

    @pytest.mark.parametrize(
        ("recipe", "config", "message"),
        [
            (regression_recipe(), PredictConfig(epsilon_P=1), "regression"),
            (regression_recipe(), PredictConfig(output_mode="geometric"), "regression"),
            (regression_recipe(), PredictConfig(output_mode="arithmetic"), "regression"),
            (
                Recipe.chain(Input(K=2), ClassificationHead(coupling=Coupling.S)),
                PredictConfig(output_mode="arithmetic"),
                "S classification",
            ),
            (classification_recipe(), {}, "PredictConfig"),
        ],
    )
    def test_task_policy_is_rejected_before_fit_or_prediction_staging(
        self, recipe, config, message
    ):
        with pytest.raises(ValueError, match=message):
            Network.fit(recipe, object(), object(), predict_config=config)
        X, codes = blobs(2, 0.04)
        y = codes.to(X.dtype) if recipe.blocks[-1].__class__ is entlearn.RegressionHead else codes
        model = Network.fit(recipe, X, y, max_iter=2)
        with pytest.raises(ValueError, match=message):
            model.predict(object(), predict_config=config)

    def test_arithmetic_has_no_temperature_and_is_not_merged_into_override(self):
        X, y = blobs(2, 0.28, seed=3)
        model = Network.fit(
            classification_recipe(Input(K=3, epsilon=0.2)),
            X,
            y,
            predict_config=PredictConfig(output_mode="arithmetic"),
            max_iter=3,
        )
        assert model.predict_config.epsilon_P is None
        arithmetic = model.predict(X)
        geometric = model.predict(X, predict_config=PredictConfig())
        assert not torch.allclose(arithmetic, geometric)
        torch.testing.assert_close(
            geometric, model.predict(X, predict_config=PredictConfig(epsilon_P=1))
        )
        assert model.predict_config.output_mode == "arithmetic"

    def test_regression_default_has_no_classification_policy(self):
        X, codes = blobs(2, 0.04)
        model = Network.fit(regression_recipe(), X, codes.to(X.dtype), max_iter=2)
        assert model.predict_config.output_mode is None
        assert model.predict_config.epsilon_P is None
        torch.testing.assert_close(
            model.predict(X), model.predict(X, predict_config=PredictConfig(tol=0, max_iter=1))
        )
