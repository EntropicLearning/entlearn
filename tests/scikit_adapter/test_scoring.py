"""Scikit-learn scorer semantics for Network candidate selection."""

from functools import partial

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE, by_depth
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import get_scorer, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from entlearn import Network
from entlearn.scikit_adapter import EONClassifier, EONRegressor
from entlearn.scikit_adapter.selection import (
    _classification_selection_loss,
    _regression_selection_loss,
)

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
        ] == expected


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
        # The scorer sees the target staged in the computation precision.
        staged = y_t.cpu().numpy()
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
            expected.append(-r2_score(staged, candidate.predict(X_t).cpu().numpy()))

        assert [
            outcome.score for outcome in estimator.network_.diagnostics.initialisation_outcomes
        ] == expected

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
        staged = y_t.cpu().numpy()
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
                fold_scores.append(mean_squared_error(staged[validation], prediction))
            expected.append(float(np.mean(fold_scores)))

        outcomes = estimator.network_.diagnostics.initialisation_outcomes
        # The fold mean may sum in another order: a few units of float64 rounding.
        rounding = 4 * np.finfo(np.float64).eps
        assert [outcome.score for outcome in outcomes] == pytest.approx(
            expected, rel=rounding, abs=0
        )
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


def _scored_rows(
    rng: np.random.Generator, labelled: np.ndarray, seed: int
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, np.ndarray]:
    """Score every labelled row for an even seed, and a random validation partition otherwise."""
    if seed % 2 == 0:
        return None, np.flatnonzero(labelled)
    validation = np.sort(rng.choice(labelled.shape[0], labelled.shape[0] // 2, replace=False))
    training = np.setdiff1d(np.arange(labelled.shape[0]), validation)
    fold = (torch.as_tensor(training, device=DEVICE), torch.as_tensor(validation, device=DEVICE))
    return fold, validation[labelled[validation]]


def _selection_loss(build, scoring, prediction, weights=None, fold=None):
    """Return the loss that the callback ``build`` makes for ``scoring`` gives ``prediction``."""
    # The callback reads the target from the adapter's own copy, not from this argument.
    return build(scoring, n_inits=2, cv=None)(
        prediction,
        torch.empty(0, device=DEVICE),
        sample_weights=weights,
        class_weights=None,
        task_weights=None,
        fold=fold,
        partition="validation",
    )


class TestNamedScorerValues:
    """Selection computes accuracy, log loss and R² itself, with the scorer's exact value."""

    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    @pytest.mark.parametrize("n_classes", [2, 3])
    @pytest.mark.parametrize("name", ["neg_log_loss", "accuracy"])
    @pytest.mark.parametrize("seed", by_depth((0, 1), range(8), range(32)))
    def test_classification_scores_equal_the_scorer(self, seed, name, n_classes, weighted):
        rng = np.random.default_rng(seed)
        codes = rng.integers(-1, n_classes, 60)
        classes = np.array(["east", "north", "west"])[:n_classes]
        fold, rows = _scored_rows(rng, codes != -1, seed)
        codes[rows[:n_classes]] = np.arange(n_classes)
        logits = rng.normal(scale=rng.choice([1.0, 20.0]), size=(rows.shape[0], n_classes))
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[::5] = np.eye(n_classes)[rng.integers(n_classes, size=len(rows[::5]))]
        prediction = torch.as_tensor(probabilities, dtype=DTYPE, device=DEVICE)
        weights = torch.as_tensor(rng.uniform(0.1, 2.0, rows.shape[0]), dtype=DTYPE, device=DEVICE)
        build = partial(_classification_selection_loss, codes, classes, np.zeros((60, 2)))

        named, scorer = (
            _selection_loss(build, scoring, prediction, weights if weighted else None, fold)
            for scoring in (name, get_scorer(name))
        )

        assert named == scorer

    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    @pytest.mark.parametrize("was_1d", [True, False], ids=["vector", "multi-output"])
    @pytest.mark.parametrize("seed", by_depth((0, 1), range(8), range(32)))
    def test_r2_equals_the_scorer(self, seed, was_1d, weighted):
        rng = np.random.default_rng(seed)
        target = rng.uniform(size=60 if was_1d else (60, 3))
        target[rng.random(60) < 0.25] = np.nan
        target = torch.as_tensor(target, dtype=DTYPE).numpy()
        labelled = ~np.isnan(target) if was_1d else ~np.isnan(target).all(axis=1)
        fold, rows = _scored_rows(rng, labelled, seed)
        truth = target[rows].reshape(rows.shape[0], -1)
        prediction = torch.as_tensor(
            truth + rng.normal(0, 0.1, truth.shape), dtype=DTYPE, device=DEVICE
        )
        weights = torch.as_tensor(rng.uniform(0.1, 2.0, rows.shape[0]), dtype=DTYPE, device=DEVICE)
        build = partial(_regression_selection_loss, target, was_1d, np.zeros((60, 2)))

        named, scorer = (
            _selection_loss(build, scoring, prediction, weights if weighted else None, fold)
            for scoring in ("r2", get_scorer("r2"))
        )

        assert named == scorer

    def test_a_class_missing_from_the_scored_rows_reaches_the_scorer_refusal(self):
        codes = np.array([0, 1, 2, 0, 1])
        build = partial(
            _classification_selection_loss, codes, np.array(["a", "b", "c"]), np.zeros((5, 2))
        )
        prediction = torch.full((4, 3), 1 / 3, dtype=DTYPE, device=DEVICE)
        fold = (torch.tensor([2], device=DEVICE), torch.tensor([0, 1, 3, 4], device=DEVICE))

        with pytest.raises(ValueError, match="different number of classes"):
            _selection_loss(build, "neg_log_loss", prediction, fold=fold)

    def test_a_constant_target_takes_the_scorer_r2(self):
        target = torch.full((4,), 0.5, dtype=DTYPE).numpy()
        build = partial(_regression_selection_loss, target, True, np.zeros((4, 2)))
        prediction = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=DTYPE, device=DEVICE)

        assert _selection_loss(build, "r2", prediction) == 0.0

    def test_one_scored_row_takes_the_scorer_warning(self):
        # One weighted row's mean differs from its value by rounding, so the formula would
        # return a finite R² where the scorer warns.
        target = torch.tensor([0.1, torch.nan], dtype=DTYPE).numpy()
        build = partial(_regression_selection_loss, target, True, np.zeros((2, 2)))
        prediction = torch.tensor([[0.2]], dtype=DTYPE, device=DEVICE)
        weights = torch.tensor([0.1], dtype=DTYPE, device=DEVICE)

        with pytest.warns(UndefinedMetricWarning, match="less than two samples"):
            _selection_loss(build, "r2", prediction, weights)
