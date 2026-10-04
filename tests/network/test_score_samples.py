"""Inlier scores rank recovered weights against the fitted prediction reference."""

from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    classification_recipe,
    collapsed_reference_model,
    connection_initialisation_data,
    prediction_model,
    regression_recipe,
    squared_error,
    validation_pairs,
)

from entlearn import Input, Network, PredictConfig


class TestScoreSamples:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["continuous", "mixed", "manifold"])
    @pytest.mark.parametrize("mode", ["single", "iterative"])
    def test_scores_rank_recovered_training_values_not_fitted_weights(self, task, kind, mode):
        policy = PredictConfig(predict_mode=mode, max_iter=3, tol=0)
        model, X, cats = prediction_model(task, input_kind=kind, predict_config=policy)
        reference = model.predict_with_details(
            X, X_cat=cats, details=("instance_weights",)
        ).instance_weights
        query = X * 0.8 + 0.1
        for override in (None, replace(model.predict_config, predict_mode="single")):
            recovered = model.predict_with_details(
                query, X_cat=cats, predict_config=override, details=("instance_weights",)
            ).instance_weights
            expected = (reference[None, :] <= recovered[:, None]).to(DTYPE).mean(dim=1)
            actual = model.score_samples(query, X_cat=cats, predict_config=override)
            torch.testing.assert_close(actual, expected)
            assert actual.dtype == DTYPE and actual.device == DEVICE
            assert not actual.requires_grad
        ranked = model.score_samples(X, X_cat=cats)
        assert ranked.min() > 0 and ranked.max() == 1

    @pytest.mark.parametrize("epsilon_T", [0.0, float("inf")])
    def test_missing_recovery_fails_before_query_staging(self, epsilon_T):
        model, _, _ = prediction_model(epsilon_T=epsilon_T)
        with pytest.raises(ValueError, match="recovery"):
            model.score_samples(None)

    def test_degenerate_reference_warns_and_retains_right_continuous_step(self):
        model = collapsed_reference_model()
        query = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        with pytest.warns(UserWarning, match="collapsed"):
            scores = model.score_samples(query)
        torch.testing.assert_close(scores, torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE))

    def test_every_retained_member_has_its_own_scoring_reference(self):
        model, X, cats = prediction_model(members=True)
        assert model.members is not None
        for member in model.members:
            reference = member.predict_with_details(
                X, X_cat=cats, details=("instance_weights",)
            ).instance_weights
            expected = (reference[None, :] <= reference[:, None]).to(DTYPE).mean(dim=1)
            torch.testing.assert_close(member.score_samples(X, X_cat=cats), expected)

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_validation_refit_builds_the_full_data_reference_with_final_policy(self, task):
        X, y = connection_initialisation_data()
        first = Input(K=2, epsilon=0.1, epsilon_T=0.4)
        recipe = (
            classification_recipe(first) if task == "classification" else regression_recipe(first)
        )
        target = y if task == "classification" else X[:, 0]
        policy = PredictConfig(predict_mode="iterative", max_iter=3, tol=0)
        model = Network.fit(
            recipe,
            X,
            target,
            validation_pairs=validation_pairs(),
            n_inits=2,
            retain="members",
            max_iter=3,
            selection_loss=squared_error,
            predict_config=policy,
        )
        assert model.members is not None
        for member in model.members:
            reference = member.predict_with_details(
                X, details=("instance_weights",)
            ).instance_weights
            expected = (reference[None, :] <= reference[:, None]).to(DTYPE).mean(dim=1)
            torch.testing.assert_close(member.score_samples(X), expected)
            replay = Network.fit(
                recipe,
                X,
                target,
                initial_state=member.initial_state,
                predict_config=policy,
                max_iter=3,
            )
            assert member.predict_config == replay.predict_config
            torch.testing.assert_close(member.predict(X), replay.predict(X))
            replay_weights = replay.predict_with_details(
                X, details=("instance_weights",)
            ).instance_weights
            torch.testing.assert_close(reference, replay_weights)
            torch.testing.assert_close(member.score_samples(X), replay.score_samples(X))


class TestScoringStateValidation:
    def test_absent_reference_rejects_before_prediction(self, monkeypatch):
        # Missing persisted state is not constructible through the public fit operation.
        model, _, _ = prediction_model()
        monkeypatch.setattr(model, "_fitted", replace(model._fitted, Wt_ref=None))
        with pytest.raises(ValueError, match="reference"):
            model.score_samples(None)

    def test_finite_normaliser_does_not_enable_a_frozen_channel(self):
        model, _, _ = prediction_model(epsilon_T=float("inf"))
        with torch.inference_mode():
            model._graph.input.log_partition.zero_()
        with pytest.raises(ValueError, match="recovery"):
            model.score_samples(None)
        with pytest.raises(ValueError, match="recovery"):
            model.predict_with_details(None, details=("instance_weights",))
