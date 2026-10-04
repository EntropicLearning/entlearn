"""Initialisation selection through the public tabular estimators."""

import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn import config_context
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate

from entlearn import (
    ClassificationHead,
    ConvergenceWarning,
    Coupling,
    InitialState,
    Input,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.scikit_adapter import EONClassifier, EONRegressor, common_train_rows
from entlearn.scikit_adapter.selection import _materialise_splits, _resolve_init_rows

from ._fixtures import (
    classifier_recipe,
    regression_data,
    regression_recipe,
    tabular_data,
    tensor_data,
)


class _RecordingSplitter:
    def __init__(self, folds):
        self.folds = folds
        self.calls = []

    def split(self, X, y, groups=None):
        self.calls.append((np.asarray(X).copy(), np.asarray(y).copy(), np.asarray(groups).copy()))
        yield from self.folds


class _GrouplessSplitter:
    """A duck-typed splitter whose split signature omits scikit-learn's groups."""

    def __init__(self, folds):
        self.folds = folds

    def split(self, X, y=None):
        yield from self.folds


class _ReusedBufferSplitter:
    """Yield every fold through the same two index arrays, overwritten in place."""

    def __init__(self, folds):
        self.folds = folds

    def split(self, X, y=None, groups=None):
        training, validation = (np.empty_like(rows) for rows in self.folds[0])
        for fold in self.folds:
            training[:], validation[:] = fold
            yield training, validation


class _UnusedSplitter:
    def split(self, X, y=None, groups=None):
        raise AssertionError("supplied state must bypass candidate-selection splitting")

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2


def problem(classification=True):
    X = np.random.default_rng(13).uniform(size=(30, 2))
    y = (X[:, 0] > 0.5).astype(int) if classification else X[:, 0]
    head = ClassificationHead(Coupling.M) if classification else RegressionHead()
    recipe = Recipe.chain(Input(K=3), head)
    return X, y, recipe


class TestFullDataSelection:
    @pytest.mark.parametrize("scoring", [None, "accuracy"])
    def test_accuracy_scorer_selects_public_network_candidates(self, scoring):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        estimator = EONClassifier(
            classifier_recipe(),
            n_inits=3,
            scoring=scoring,
            retain="states",
            random_state=11,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        outcomes = estimator.network_.diagnostics.initialisation_outcomes
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
            predicted = np.unique(y)[candidate.predict(X_t).argmax(dim=1).cpu().numpy()]
            expected.append(-float(np.mean(predicted == y)))

        assert [outcome.score for outcome in outcomes] == pytest.approx(expected)
        assert (
            estimator.network_.initial_state
            is estimator.network_.initial_states[np.argmin(expected)]
        )
        for legacy_name in ("inits_", "init_scores_", "sub_seeds_", "best_init_index_"):
            assert not hasattr(estimator, legacy_name)

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_dispatch_and_member_retention_delegate_to_network(self, backend):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(),
            n_inits=2,
            n_jobs=2,
            parallel_backend=backend,
            retain="members",
            random_state=7,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        assert len(estimator.network_.members) == 2
        assert len(estimator.network_.initial_states) == 2
        for member, state in zip(
            estimator.network_.members, estimator.network_.initial_states, strict=True
        ):
            assert member.initial_state is state
        outcomes = estimator.network_.diagnostics.initialisation_outcomes
        assert {outcome.requested_backend for outcome in outcomes} == {backend}
        # n_jobs reached Network: one worker per candidate, never serial execution.
        assert "serial" not in {outcome.effective_backend for outcome in outcomes}

    def test_a_single_unscored_candidate_records_the_training_objective(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), random_state=4, dtype=DTYPE, device=DEVICE
        ).fit(X, y)

        outcomes = estimator.network_.diagnostics.initialisation_outcomes
        assert [outcome.score for outcome in outcomes] == [estimator.loss_curve_[-1]]

    def test_explicit_initialisation_rows_restrict_the_fitted_geometry(self):
        X, labels = tabular_data()
        X_t, y_t = tensor_data(X, labels)
        rows = [0, 1, 2, 16, 17, 32]
        estimator = EONClassifier(
            classifier_recipe(), init_rows=rows, random_state=7, dtype=DTYPE, device=DEVICE
        ).fit(X, labels)

        seed = estimator.network_.diagnostics.initialisation_outcomes[0].seed
        expected = Network.initialise(estimator.recipe, X_t[rows], y_t[rows], seed=seed)
        whole = Network.initialise(estimator.recipe, X_t, y_t, seed=seed)
        torch.testing.assert_close(
            estimator.network_.initial_state.input_geometry.continuous_centroids,
            expected.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
        assert not torch.equal(
            expected.input_geometry.continuous_centroids, whole.input_geometry.continuous_centroids
        )

    def test_scored_regression_targets_reach_the_scorer_in_computation_precision(self):
        X, Y = regression_data()
        seen = []

        def scorer(estimator, X_part, y_part):
            seen.append(y_part.dtype)
            return -float(np.mean((estimator.predict(X_part) - y_part) ** 2))

        EONRegressor(
            regression_recipe(),
            scoring=scorer,
            dtype=torch.float32,
            device=DEVICE,
            random_state=2,
        ).fit(X, Y)

        assert set(seen) == {np.dtype(np.float32)}


class TestValidationSelection:
    def test_splitter_is_materialised_once_with_groups_and_scorer_weights(self):
        X, labels = tabular_data()
        y = np.unique(labels, return_inverse=True)[1]
        even = np.arange(0, len(y), 2)
        odd = np.arange(1, len(y), 2)
        splitter = _RecordingSplitter(((odd, even), (even, odd)))
        groups = np.repeat(np.arange(12), 4)
        weights = np.linspace(0.5, 2.0, len(y))
        scored = []

        def scorer(estimator, X_part, y_part, sample_weight=None):
            scored.append(
                (
                    np.asarray(X_part).copy(),
                    np.asarray(y_part).copy(),
                    np.asarray(sample_weight).copy(),
                )
            )
            return float(np.average(estimator.predict(X_part) == y_part, weights=sample_weight))

        estimator = EONClassifier(
            classifier_recipe(),
            n_inits=2,
            cv=splitter,
            scoring=scorer,
            return_train_score=True,
            random_state=5,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y, sample_weight=weights, groups=groups)

        assert len(splitter.calls) == 1
        split_X, split_y, split_groups = splitter.calls[0]
        np.testing.assert_array_equal(split_X, X)
        np.testing.assert_array_equal(split_y, y)
        np.testing.assert_array_equal(split_groups, groups)
        assert len(scored) == 2 * 2 * 2
        expected_weights = weights.astype(np.float32 if DTYPE is torch.float32 else np.float64)
        weight_by_row = {
            tuple(row): weight for row, weight in zip(X, expected_weights, strict=True)
        }
        for scored_X, scored_y, scored_weights in scored:
            np.testing.assert_array_equal(
                scored_weights, [weight_by_row[tuple(row)] for row in scored_X]
            )
            assert len(scored_X) == len(scored_y) == len(scored_weights)
        assert len(estimator.network_.diagnostics.initialisation_outcomes) == 2

    def test_regression_scorer_receives_the_fold_rows_and_their_weights(self):
        X, y = regression_data()
        weights = np.linspace(0.5, 2.0, len(y))
        scored = []

        def scorer(estimator, X_part, y_part, sample_weight=None):
            scored.append((np.asarray(X_part).copy(), np.asarray(sample_weight).copy()))
            return -float(np.average((estimator.predict(X_part) - y_part) ** 2))

        EONRegressor(
            regression_recipe(),
            cv=TimeSeriesSplit(2),
            scoring=scorer,
            return_train_score=True,
            random_state=3,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y, sample_weight=weights)

        expected_weights = weights.astype(np.float32 if DTYPE is torch.float32 else np.float64)
        weight_by_row = {
            tuple(row): weight for row, weight in zip(X, expected_weights, strict=True)
        }
        assert len(scored) == 2 * 2
        for scored_X, scored_weights in scored:
            assert 0 < len(scored_X) < len(X)
            np.testing.assert_array_equal(
                scored_weights, [weight_by_row[tuple(row)] for row in scored_X]
            )

    def test_integer_cv_stratifies_the_classification_folds(self):
        X, y = tabular_data()
        seen = []

        def scorer(estimator, X_part, y_part):
            seen.append(np.unique(y_part))
            return float(np.mean(estimator.predict(X_part) == y_part))

        EONClassifier(
            classifier_recipe(),
            cv=3,
            scoring=scorer,
            random_state=9,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        # Only validation partitions are scored without return_train_score.
        assert len(seen) == 3
        for classes in seen:
            np.testing.assert_array_equal(classes, np.unique(y))

    def test_regression_splitter_receives_the_target_and_fit_groups(self):
        X, Y = regression_data()
        y = Y[:, 0]
        even, odd = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
        splitter = _RecordingSplitter(((odd, even), (even, odd)))
        groups = np.repeat(np.arange(12), 4)

        EONRegressor(
            regression_recipe(),
            n_inits=2,
            cv=splitter,
            random_state=5,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y, groups=groups)

        assert len(splitter.calls) == 1
        split_X, split_y, split_groups = splitter.calls[0]
        np.testing.assert_array_equal(split_X, X)
        staged = y.astype(np.float32 if DTYPE is torch.float32 else np.float64)
        np.testing.assert_array_equal(split_y, staged)
        np.testing.assert_array_equal(split_groups, groups)

    def test_integral_regression_targets_are_never_stratified(self):
        X, _ = regression_data()
        # Whole-numbered targets look like class labels to scikit-learn's target
        # inspection; only a regression splitter tolerates one row per value.
        y = np.arange(len(X), dtype=float)
        estimator = EONRegressor(
            regression_recipe(), cv=2, random_state=5, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        assert estimator.predict(X).shape == y.shape

    def test_classifier_folds_are_stratified_and_reach_a_callable_init_rows(self):
        X, y = tabular_data()
        received = []

        def first_training_fold(folds):
            received.append(folds)
            return folds[0][0]

        estimator = EONClassifier(
            classifier_recipe(),
            cv=3,
            init_rows=first_training_fold,
            random_state=5,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)

        assert len(received) == 1
        classes = set(np.unique(y))
        for training, validation in received[0]:
            assert set(y[training]) == classes
            assert set(y[validation]) == classes
        assert len(estimator.network_.diagnostics.initialisation_outcomes) == 1

    def test_callable_initialisation_rows_receive_materialised_folds(self):
        X, y = regression_data()
        weights = np.linspace(0.5, 2.0, len(y))
        received = []

        def common_training_rows(folds):
            received.append(folds)
            return sorted(set.intersection(*(set(training) for training, _ in folds)))

        estimator = EONRegressor(
            regression_recipe(),
            n_inits=2,
            cv=TimeSeriesSplit(3),
            init_rows=common_training_rows,
            retain="states",
            random_state=17,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y, sample_weight=weights)

        assert len(received) == 1
        rows = np.asarray(sorted(set.intersection(*(set(training) for training, _ in received[0]))))
        X_t = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        y_t = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
        weights_t = torch.as_tensor(weights, dtype=DTYPE, device=DEVICE)
        for outcome, state in zip(
            estimator.network_.diagnostics.initialisation_outcomes,
            estimator.network_.initial_states,
            strict=True,
        ):
            expected = Network.initialise(
                estimator.recipe,
                X_t[rows],
                y_t[rows],
                sample_weights=weights_t[rows],
                seed=outcome.seed,
            )
            torch.testing.assert_close(
                state.input_geometry.continuous_centroids,
                expected.input_geometry.continuous_centroids,
                rtol=0,
                atol=0,
            )


class TestSelectionContracts:
    def test_fit_metadata_declares_sample_weights_and_groups(self):
        estimator = EONClassifier(classifier_recipe())
        with config_context(enable_metadata_routing=True):
            routed = estimator.set_fit_request(sample_weight=True, groups=True)
            assert routed.get_metadata_routing().consumes("fit", ["sample_weight", "groups"]) == {
                "sample_weight",
                "groups",
            }

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"scoring": "not_a_scorer"}, "not_a_scorer"),
            ({"init_rows": lambda folds: [0, 1, 2]}, "requires cv"),
            ({"cv": [(np.array([0.0, 1.0]), np.array([2, 3]))]}, "integer array"),
            ({"init_rows": np.array([0.0, 1.0, 2.0])}, "integer array"),
        ],
    )
    def test_invalid_selection_controls_raise_before_publication(self, kwargs, message):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), n_inits=2, dtype=DTYPE, device=DEVICE, **kwargs
        )
        with pytest.raises((TypeError, ValueError), match=message):
            estimator.fit(X, y)
        assert not hasattr(estimator, "network_")

    def test_group_length_mismatch_raises_before_splitter_traversal(self):
        X, y = tabular_data()
        splitter = _RecordingSplitter(())
        estimator = EONClassifier(
            classifier_recipe(),
            n_inits=2,
            cv=splitter,
            dtype=DTYPE,
            device=DEVICE,
        )
        with pytest.raises(ValueError, match="inconsistent numbers"):
            estimator.fit(X, y, groups=np.ones(len(y) - 1))
        assert splitter.calls == []
        assert not hasattr(estimator, "network_")

    def test_selection_loss_is_built_only_when_a_control_requests_it(self):
        X, y = tabular_data()
        scored = []

        def scorer(estimator, X_part, y_part):
            scored.append(len(y_part))
            return float(np.mean(estimator.predict(X_part) == y_part))

        controls = {"dtype": DTYPE, "device": DEVICE, "random_state": 6}
        plain = EONClassifier(classifier_recipe(), **controls).fit(X, y)
        outcome = plain.network_.diagnostics.initialisation_outcomes[0]
        assert outcome.score == pytest.approx(plain.network_.diagnostics.loss_history[-1])
        assert scored == []

        EONClassifier(classifier_recipe(), scoring=scorer, **controls).fit(X, y)
        assert len(scored) == 1

        validated = EONClassifier(classifier_recipe(), cv=2, **controls).fit(X, y)
        assert len(validated.network_.diagnostics.initialisation_outcomes) == 1

    def test_splitter_without_a_groups_parameter_is_traversed(self):
        X, y = tabular_data()
        even, odd = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
        estimator = EONClassifier(
            classifier_recipe(),
            cv=_GrouplessSplitter(((even, odd), (odd, even))),
            random_state=3,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)
        assert len(estimator.network_.diagnostics.initialisation_outcomes) == 1

    def test_row_index_tensors_are_placed_on_the_requested_device(self):
        X, y = tabular_data()
        codes = np.unique(y, return_inverse=True)[1]
        device = torch.device("meta")
        pairs = _materialise_splits(
            [(np.arange(0, len(y) // 2), np.arange(len(y) // 2, len(y)))],
            X,
            codes,
            None,
            classifier=True,
            device=device,
        )
        assert pairs is not None
        assert [tensor.device for pair in pairs for tensor in pair] == [device, device]
        rows = _resolve_init_rows([0, 1, 2], pairs, device=device)
        assert rows is not None
        assert rows.device == device

    def test_folds_are_snapshotted_when_a_splitter_reuses_its_buffers(self):
        X, y = tabular_data()
        codes = np.unique(y, return_inverse=True)[1]
        even, odd = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
        folds = ((even, odd), (odd, even))
        pairs = _materialise_splits(
            _ReusedBufferSplitter(folds), X, codes, None, classifier=True, device=DEVICE
        )
        assert pairs is not None
        for fold, tensors in zip(folds, pairs, strict=True):
            for expected, tensor in zip(fold, tensors, strict=True):
                np.testing.assert_array_equal(tensor.cpu().numpy(), expected)

    def test_failed_scorer_preserves_a_previous_complete_fit(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4
        ).fit(X, y)
        previous = estimator.network_
        prediction = estimator.predict_proba(X)

        def fail(*args, **kwargs):
            raise RuntimeError("scorer failed")

        estimator.set_params(n_inits=2, scoring=fail)
        with pytest.raises(RuntimeError, match="scorer failed"):
            estimator.fit(X, y)
        assert estimator.network_ is previous
        np.testing.assert_array_equal(estimator.predict_proba(X), prediction)

    def test_adapter_passes_through_one_core_convergence_warning(self):
        X, y = tabular_data()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            EONClassifier(
                classifier_recipe(),
                n_inits=2,
                max_iter=1,
                tol=0,
                dtype=DTYPE,
                device=DEVICE,
            ).fit(X, y)
        convergence = [
            warning
            for warning in seen
            if issubclass(warning.category, (ConvergenceWarning, SklearnConvergenceWarning))
        ]
        assert len(convergence) == 1
        assert convergence[0].category is ConvergenceWarning


class TestCommonTrainRows:
    def test_common_rows_are_the_sorted_intersection_of_the_training_partitions(self):
        folds = [(np.arange(20)[::-1], np.arange(20, 30)), (np.arange(10, 30), np.arange(10))]
        np.testing.assert_array_equal(common_train_rows(folds), np.arange(10, 20))

    @pytest.mark.parametrize(
        "training",
        [np.zeros((2, 3), dtype=np.int64), np.arange(3.0)],
        ids=("two-dimensional", "floating"),
    )
    def test_common_rows_reject_malformed_training_indices(self, training):
        with pytest.raises(ValueError, match="one-dimensional integer"):
            common_train_rows([(training, np.arange(3, 6))])

    def test_common_rows_need_at_least_one_fold(self):
        with pytest.raises(ValueError, match="needs at least one fold"):
            common_train_rows([])

    def test_common_rows_are_int64_whatever_the_fold_index_dtype(self):
        folds = [(np.arange(5, dtype=np.int32), np.arange(5, 8))] * 2
        assert common_train_rows(folds).dtype == np.int64


class TestEnclosingSearch:
    def test_an_outer_scorer_reads_the_inner_winners_gap(self):
        X, y, recipe = problem()

        def inner_gap(estimator, X, y):
            # Both scores are negated accuracies, so this is the inner winner's mean
            # training accuracy minus its mean validation accuracy.
            outcome = estimator.network_.diagnostics.selected_outcome
            return outcome.score - outcome.train_score

        inner = EONClassifier(
            recipe,
            dtype=DTYPE,
            device=DEVICE,
            max_iter=2,
            n_inits=3,
            cv=2,
            return_train_score=True,
        )
        result = cross_validate(
            inner,
            X,
            y,
            cv=2,
            scoring={"accuracy": "accuracy", "inner_gap": inner_gap},
            return_estimator=True,
        )
        for gap, fitted in zip(result["test_inner_gap"], result["estimator"], strict=True):
            outcomes = fitted.network_.diagnostics.initialisation_outcomes
            winner = min(outcomes, key=lambda item: (item.score, item.score - item.train_score))
            assert fitted.network_.diagnostics.selected_outcome is winner
            assert gap == winner.score - winner.train_score

    @pytest.mark.parametrize("classification", [False, True])
    def test_supplied_state_bypasses_selection_splitter_and_scorer(self, classification):
        X, y, recipe = problem(classification)
        state = Network.initialise(
            recipe,
            torch.as_tensor(X, dtype=DTYPE, device=DEVICE),
            torch.as_tensor(y, device=DEVICE),
            computation_dtype=DTYPE,
        )

        def unused(*args, **kwargs):
            pytest.fail("selection scorer ran with a supplied state")

        estimator = (EONClassifier if classification else EONRegressor)(
            recipe,
            initial_state=state,
            cv=_UnusedSplitter(),
            scoring=unused,
            dtype=DTYPE,
            device=DEVICE,
            max_iter=2,
        ).fit(X, y)
        assert estimator.predict(X).shape == y.shape

    def test_core_supplied_state_ignores_selection_controls(self):
        X, y, recipe = problem()
        tensor = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        target = torch.as_tensor(y, device=DEVICE)
        state = Network.initialise(recipe, tensor, target)

        def unused(*args, **kwargs):
            pytest.fail("selection loss ran with a supplied state")

        result = Network.fit(
            recipe,
            tensor,
            target,
            initial_state=state,
            validation_pairs=object(),
            selection_loss=unused,
            max_iter=2,
        )
        replay = Network.fit(recipe, tensor, target, initial_state=state, max_iter=2)
        torch.testing.assert_close(result.predict(tensor), replay.predict(tensor))

    def test_a_native_grid_fits_its_winner_from_the_selected_state(self):
        X, y, recipe = problem()
        tensor = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        target = torch.as_tensor(y, device=DEVICE)
        states = [Network.initialise(recipe, tensor, target, seed=seed) for seed in (1, 2)]
        search = GridSearchCV(
            EONClassifier(recipe, dtype=DTYPE, device=DEVICE, max_iter=2, random_state=0),
            {"initial_state": states, "recipe__blocks__input__epsilon": [0.1, 0.2]},
            cv=2,
            error_score="raise",
        ).fit(X, y)
        torch.testing.assert_close(
            search.best_estimator_.network_.initial_state.input_geometry.continuous_centroids,
            search.best_params_["initial_state"].input_geometry.continuous_centroids,
        )

    def test_a_native_grid_fits_a_smaller_K_from_the_leading_centroids_of_each_state(
        self, monkeypatch
    ):
        X, y, _ = problem()
        recipe = Recipe.chain(Input(K=5), ClassificationHead(Coupling.M))
        tensor = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        target = torch.as_tensor(y, device=DEVICE)
        states = [Network.initialise(recipe, tensor, target, seed=seed) for seed in (1, 2)]
        fit = Network.fit
        fits = []

        def record(*args, **kwargs):
            network = fit(*args, **kwargs)
            fits.append((args, kwargs, network))
            return network

        monkeypatch.setattr(Network, "fit", record)
        search = GridSearchCV(
            EONClassifier(recipe, dtype=DTYPE, device=DEVICE, max_iter=2, random_state=0),
            {"initial_state": states, "recipe__blocks__input__K": [3, 5]},
            cv=2,
            error_score="raise",
        ).fit(X, y)

        # Four candidates on two folds, then one refit of the winner on every row.
        assert len(fits) == 4 * 2 + 1
        for args, kwargs, network in fits:
            geometry = kwargs["initial_state"].input_geometry
            K = args[0].blocks[0].K
            leading = InitialState(
                replace(
                    geometry,
                    input=replace(geometry.input, K=K),
                    continuous_centroids=geometry.continuous_centroids[:K],
                ),
                connection_sub_seeds=kwargs["initial_state"].connection_sub_seeds,
            )
            reference = fit(*args, **{**kwargs, "initial_state": leading})
            torch.testing.assert_close(
                network.predict(args[1]), reference.predict(args[1]), rtol=0, atol=0
            )
        refit_args, refit_kwargs, _ = fits[-1]
        assert len(refit_args[1]) == len(X)
        torch.testing.assert_close(
            refit_kwargs["initial_state"].input_geometry.continuous_centroids,
            search.best_params_["initial_state"].input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
