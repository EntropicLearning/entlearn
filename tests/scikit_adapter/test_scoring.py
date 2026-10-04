"""Scikit-learn scorer semantics for Network candidate selection."""

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from entlearn import Network
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import classifier_recipe, regression_data, regression_recipe, tabular_data


class TestClassificationScoring:
    def test_callable_receives_real_labelled_rows_and_sample_weights(self):
        X, labels = tabular_data()
        y = np.unique(labels, return_inverse=True)[1]
        y[::7] = -1
        weights = np.linspace(0.5, 2.0, len(y))
        calls = []

        def scorer(estimator, X_part, y_part, sample_weight=None):
            calls.append((X_part.copy(), y_part.copy(), sample_weight.copy()))
            return float(np.average(estimator.predict(X_part) == y_part, weights=sample_weight))

        EONClassifier(
            classifier_recipe(),
            n_inits=2,
            scoring=scorer,
            random_state=9,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y, sample_weight=weights)

        assert len(calls) == 2
        expected_rows = y != -1
        expected_weights = weights.astype(np.float32 if DTYPE is torch.float32 else np.float64)
        for X_part, y_part, scored_weights in calls:
            np.testing.assert_array_equal(X_part, X[expected_rows])
            np.testing.assert_array_equal(y_part, y[expected_rows])
            np.testing.assert_array_equal(scored_weights, expected_weights[expected_rows])

    def test_probability_scorer_reads_cached_probabilities(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(),
            n_inits=2,
            scoring="neg_log_loss",
            retain="members",
            random_state=3,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        assert estimator.network_.members is not None
        expected = [
            log_loss(
                y, member.predict(torch.as_tensor(X, dtype=DTYPE, device=DEVICE)).cpu().numpy()
            )
            for member in estimator.network_.members
        ]
        assert [
            outcome.score for outcome in estimator.network_.diagnostics.initialisation_outcomes
        ] == (pytest.approx(expected))


class TestRegressionScoring:
    @pytest.mark.parametrize("target_shape", ["vector", "multi_output"])
    def test_default_single_candidate_fit_accepts_one_labelled_row(self, target_shape):
        X = np.arange(12, dtype=float).reshape(6, 2) / 12
        if target_shape == "vector":
            y = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        else:
            y = np.full((6, 2), np.nan)
            y[0] = [1.0, 2.0]

        estimator = EONRegressor(
            regression_recipe(), random_state=3, dtype=DTYPE, device=DEVICE
        ).fit(X, y)

        assert np.isfinite(estimator.predict(X)).all()
        outcome = estimator.network_.diagnostics.initialisation_outcomes[0]
        assert outcome.score == estimator.network_.diagnostics.loss_history[-1]

    @pytest.mark.parametrize("columns", [0, slice(None)], ids=["vector", "multi-output"])
    def test_explicit_scorer_still_runs_for_one_candidate_without_validation(self, columns):
        X, outputs = regression_data()
        y = outputs[:, columns]
        calls = []

        def scorer(estimator, X_part, target):
            calls.append((estimator.predict(X_part), target.copy()))
            return 0.25

        estimator = EONRegressor(
            regression_recipe(),
            scoring=scorer,
            random_state=3,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        assert len(calls) == 1
        assert calls[0][0].shape == calls[0][1].shape
        assert estimator.network_.diagnostics.initialisation_outcomes[0].score == -0.25

    def test_default_scorer_runs_for_one_candidate_with_explicit_validation(self):
        X, y = regression_data()
        estimator = EONRegressor(
            regression_recipe(),
            cv=KFold(2),
            random_state=3,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        outcome = estimator.network_.diagnostics.initialisation_outcomes[0]
        assert np.isfinite(outcome.score)

    def test_default_scorer_is_r2_and_preserves_multi_output_shape(self):
        X, y = regression_data()
        estimator = EONRegressor(
            regression_recipe(),
            n_inits=3,
            retain="states",
            random_state=13,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        X_t = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        y_t = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
        assert estimator.network_.initial_states is not None
        expected = []
        for state in estimator.network_.initial_states:
            candidate = Network.fit(
                estimator.recipe,
                X_t,
                y_t,
                initial_state=state,
                computation_dtype=DTYPE,
            )
            expected.append(-r2_score(y, candidate.predict(X_t).cpu().numpy()))

        assert [
            outcome.score for outcome in estimator.network_.diagnostics.initialisation_outcomes
        ] == (pytest.approx(expected))

    def test_validation_scorer_sign_mean_winner_and_refit_match_public_network(self):
        X, outputs = regression_data()
        y = outputs[:, 0]
        folds = tuple(KFold(3).split(X))
        estimator = EONRegressor(
            regression_recipe(),
            n_inits=2,
            cv=folds,
            scoring="neg_mean_squared_error",
            retain="states",
            random_state=19,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        X_t = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        y_t = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
        states = estimator.network_.initial_states
        assert states is not None
        expected = []
        for state in states:
            fold_scores = []
            for training, validation in folds:
                candidate = Network.fit(
                    estimator.recipe,
                    X_t[training],
                    y_t[training],
                    initial_state=state,
                    computation_dtype=DTYPE,
                )
                prediction = candidate.predict(X_t[validation]).cpu().numpy()[:, 0]
                fold_scores.append(mean_squared_error(y[validation], prediction))
            expected.append(float(np.mean(fold_scores)))

        outcomes = estimator.network_.diagnostics.initialisation_outcomes
        assert [outcome.score for outcome in outcomes] == pytest.approx(expected)
        winner = int(np.argmin(expected))
        assert estimator.network_.initial_state is states[winner]
        replay = Network.fit(
            estimator.recipe,
            X_t,
            y_t,
            initial_state=states[winner],
            computation_dtype=DTYPE,
        )
        np.testing.assert_array_equal(estimator.predict(X), replay.predict(X_t).cpu().numpy()[:, 0])
