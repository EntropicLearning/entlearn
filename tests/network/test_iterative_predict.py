"""Iterative prediction through the public Network interface."""

from dataclasses import replace
from itertools import pairwise

import pytest
import torch
from network._fixtures import (
    blobs,
    deep_classification_recipe,
    deep_regression_recipe,
    prediction_model,
)

from entlearn import Network, PredictConfig


class TestIterativePrediction:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_refinement_descends_and_details_match_primary_prediction(self, task):
        X, codes = blobs(3, 0.28, seed=3)
        classification = task == "classification"
        recipe = (
            deep_classification_recipe((5, 4), epsilon=0.1)
            if classification
            else deep_regression_recipe((5, 4), epsilon=0.1)
        )
        y = codes if classification else X[:, :2].square()
        config = PredictConfig(
            predict_mode="iterative",
            epsilon_P=0.2 if classification else None,
            tol=1e-6,
            max_iter=40,
        )
        model = Network.fit(recipe, X, y, predict_config=config, max_iter=20)
        query = X * 0.7 + 0.15
        single = model.predict(query, predict_config=replace(config, predict_mode="single"))
        result = model.predict_with_details(query, details=("affiliations", "diagnostics"))
        assert not torch.equal(single, result.prediction)
        torch.testing.assert_close(model.predict(query), result.prediction, rtol=0, atol=0)
        assert len(result.loss_history) == result.n_iter
        assert result.n_iter >= 2
        for previous, current in pairwise(result.loss_history):
            assert current <= previous + 256 * torch.finfo(X.dtype).eps * max(1, abs(previous))
        for gamma in result.affiliations.values():
            torch.testing.assert_close(gamma.sum(1), torch.ones_like(X[:, 0]))
        if classification:
            torch.testing.assert_close(result.prediction.sum(1), torch.ones_like(X[:, 0]))


class TestIterativeReconstruction:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["continuous", "manifold"])
    def test_reconstruct_uses_the_final_input_affiliations(self, task, kind):
        model, X, _ = prediction_model(task, input_kind=kind)
        policy = replace(model.predict_config, predict_mode="iterative", max_iter=3, tol=0)
        result = model.predict_with_details(X, predict_config=policy, details=("affiliations",))
        # The numerical oracle reads fitted geometry independently of reconstruction.
        geometry = model._graph.input
        gamma = result.affiliations[model.recipe.blocks[0].name]
        if kind == "continuous":
            expected = gamma @ geometry.continuous_centroids
        else:
            centres = geometry.continuous_centroids
            basis = geometry.manifold_projectors
            residual = X[:, None, :] - centres
            projected = torch.einsum("tkd,kdj->tkj", residual, basis)
            charts = centres + torch.einsum("tkj,kdj->tkd", projected, basis)
            expected = (gamma[:, :, None] * charts).sum(1)
        torch.testing.assert_close(model.reconstruct(X, predict_config=policy).continuous, expected)


class TestCalibrationReuse:
    def test_iterative_default_reuses_single_pass_calibration(self):
        X, y = blobs(3, 0.28, seed=3)
        recipe = deep_classification_recipe((5, 4), epsilon=0.1)
        single = Network.fit(recipe, X, y, max_iter=10)
        iterative = Network.fit(
            recipe, X, y, max_iter=10, predict_config=PredictConfig(predict_mode="iterative")
        )
        assert iterative.predict_config.epsilon_P == single.predict_config.epsilon_P
        expected = single.predict(
            X, predict_config=replace(single.predict_config, predict_mode="iterative")
        )
        torch.testing.assert_close(iterative.predict(X), expected, rtol=0, atol=0)
