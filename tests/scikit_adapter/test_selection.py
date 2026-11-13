"""Initialisation selection through the public tabular estimators."""

import warnings

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from entlearn import (
    ClassificationHead,
    ConvergenceWarning,
    Coupling,
    Input,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.scikit_adapter import EONClassifier, EONRegressor, common_train_rows

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
        self.calls = 0

    def split(self, X, y=None):
        self.calls += 1
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


def categorical_problem(classification=True):
    """Two continuous columns in [0, 1] and a categorical column cycling through p, q, r."""
    x = np.random.default_rng(1).uniform(size=(60, 2))
    levels = np.array(["p", "q", "r"], dtype=object)[np.arange(60) % 3]
    if classification:
        y = np.where(levels == "r", "a", np.where(x[:, 0] > 0.5, "a", "b")).astype(object)
    else:
        y = x[:, 0] + (levels == "r")
    X = np.column_stack([x, levels]).astype(object)
    return X, y, levels, categorical_recipe(classification)


def categorical_recipe(classification=True, K=4):
    head = ClassificationHead() if classification else RegressionHead()
    return Recipe.chain(Input(K=K, epsilon=0.01), head)


def categorical_estimator(classification=True, K=4, **params):
    estimator_type = EONClassifier if classification else EONRegressor
    return estimator_type(
        categorical_recipe(classification, K),
        categorical_features=[2],
        random_state=0,
        dtype=DTYPE,
        device=DEVICE,
        max_iter=2,
        **params,
    )


def class_problem():
    """Two continuous columns in [0, 1], with classes a, b and c split along the first."""
    x = np.random.default_rng(0).uniform(size=(90, 2))
    y = np.array(["a", "b", "c"], dtype=object)[np.minimum((3 * x[:, 0]).astype(int), 2)]
    return x, y


def class_estimator(K=6, **params):
    return EONClassifier(
        Recipe.chain(Input(K=K, epsilon=0.01), ClassificationHead()),
        random_state=0,
        dtype=DTYPE,
        device=DEVICE,
        max_iter=2,
        **params,
    )


def captured_class_state(x, y):
    """A state captured from a classifier fitted on ``x`` and ``y``, and its cluster count."""
    state = class_estimator().fit(x, y).network_.capture_current_state()
    assert state.input_geometry is not None
    return state, state.input_geometry.K_active


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

    def test_dispatch_controls_reach_candidate_selection(self):
        X, y = tabular_data()
        controls = dict(dtype=DTYPE, device=DEVICE, random_state=7, max_iter=3, n_inits=2)
        threaded = EONClassifier(classifier_recipe(), n_jobs=2, **controls).fit(X, y)
        assert {
            outcome.effective_backend
            for outcome in threaded.network_.diagnostics.initialisation_outcomes
        } == {"threads"}
        # A single worker needs no pool, but still records the requested policy.
        serial = EONClassifier(classifier_recipe(), parallel_backend="processes", **controls).fit(
            X, y
        )
        assert {
            (outcome.requested_backend, outcome.effective_backend)
            for outcome in serial.network_.diagnostics.initialisation_outcomes
        } == {("processes", "serial")}

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

    def test_a_regression_splitter_receives_the_staged_target(self):
        X, Y = regression_data()
        y = Y[:, 0]
        even, odd = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
        splitter = _RecordingSplitter(((odd, even), (even, odd)))

        EONRegressor(
            regression_recipe(),
            n_inits=2,
            cv=splitter,
            random_state=5,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y, groups=np.repeat(np.arange(12), 4))

        _, split_y, _ = splitter.calls[0]
        staged = y.astype(np.float32 if DTYPE is torch.float32 else np.float64)
        np.testing.assert_array_equal(split_y, staged)
        assert split_y.dtype == staged.dtype

    def test_integral_regression_targets_are_never_stratified(self):
        X, _ = regression_data()
        # Whole-numbered targets look like class labels to scikit-learn's target
        # inspection; only a regression splitter tolerates one row per value.
        y = np.arange(len(X), dtype=float)
        estimator = EONRegressor(
            regression_recipe(), cv=2, random_state=5, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        assert estimator.predict(X).shape == y.shape

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
    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"init_rows": lambda folds: [0, 1, 2]}, "requires cv"),
            ({"cv": [(np.array([0.0, 1.0]), np.array([2, 3]))]}, "cv fold 0 training"),
            ({"init_rows": np.array([0.0, 1.0, 2.0])}, "init_rows must resolve"),
        ],
        ids=["callable-init-rows-without-cv", "floating-fold", "floating-init-rows"],
    )
    def test_invalid_selection_controls_raise_before_publication(self, kwargs, message):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), n_inits=2, dtype=DTYPE, device=DEVICE, **kwargs
        )
        with pytest.raises(ValueError, match=message):
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

    def test_a_scorer_or_a_splitter_alone_requests_scored_selection(self):
        X, y = tabular_data()
        scored = []

        def scorer(estimator, X_part, y_part):
            scored.append(len(y_part))
            return float(np.mean(estimator.predict(X_part) == y_part))

        controls = {"dtype": DTYPE, "device": DEVICE, "random_state": 6}
        EONClassifier(classifier_recipe(), scoring=scorer, **controls).fit(X, y)
        assert len(scored) == 1

        # Unscored, the one outcome records the training objective instead.
        validated = EONClassifier(classifier_recipe(), cv=2, **controls).fit(X, y)
        outcome = validated.network_.diagnostics.initialisation_outcomes[0]
        assert outcome.score != validated.loss_curve_[-1]

    def test_splitter_without_a_groups_parameter_is_traversed(self):
        X, y = tabular_data()
        even, odd = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
        splitter = _GrouplessSplitter(((even, odd), (odd, even)))
        EONClassifier(
            classifier_recipe(), cv=splitter, random_state=3, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        assert splitter.calls == 1

    def test_folds_are_snapshotted_when_a_splitter_reuses_its_buffers(self):
        X, y = tabular_data()
        even, odd = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
        folds = ((even, odd), (odd, even))
        received = []

        def first_training_fold(materialised):
            received.append(materialised)
            return materialised[0][0]

        EONClassifier(
            classifier_recipe(),
            cv=_ReusedBufferSplitter(folds),
            init_rows=first_training_fold,
            random_state=3,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)
        for fold, materialised in zip(folds, received[0], strict=True):
            for expected, rows in zip(fold, materialised, strict=True):
                np.testing.assert_array_equal(rows, expected)

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

    def test_disjoint_training_partitions_share_no_rows(self):
        folds = [(np.arange(5), np.arange(5, 10)), (np.arange(5, 10), np.arange(5))]
        common = common_train_rows(folds)
        assert common.shape == (0,)
        assert common.dtype == np.int64

    def test_common_rows_are_int64_whatever_the_fold_index_dtype(self):
        folds = [(np.arange(5, dtype=np.int32), np.arange(5, 8))] * 2
        assert common_train_rows(folds).dtype == np.int64


@FIXED_BUDGET
class TestEnclosingSearch:
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
        assert torch.equal(result.predict(tensor), replay.predict(tensor))

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


@FIXED_BUDGET
class TestStateCategoricalLevels:
    """A supplied state's level counts must equal those the fit learns for each column."""

    @pytest.mark.parametrize("classification", [False, True])
    def test_rows_lacking_a_level_of_the_state_raise(self, classification):
        X, y, levels, _ = categorical_problem(classification)
        state = categorical_estimator(classification).fit(X, y).network_.initial_state
        keep = levels != "q"
        estimator = categorical_estimator(classification, initial_state=state)
        with pytest.raises(ValueError, match=r"positions \(2,\) hold \(2,\) level.* holds \(3,\)"):
            estimator.fit(X[keep], y[keep])

    def test_rows_with_a_level_the_state_lacks_raise(self):
        X, y, levels, _ = categorical_problem()
        keep = levels != "q"
        state = categorical_estimator().fit(X[keep], y[keep]).network_.initial_state
        estimator = categorical_estimator(initial_state=state)
        with pytest.raises(ValueError, match=r"positions \(2,\) hold \(3,\) level.* holds \(2,\)"):
            estimator.fit(X, y)

    def test_a_wider_input_than_the_state_still_raises_on_rows_lacking_a_level(self):
        X, y, levels, _ = categorical_problem()
        state = categorical_estimator().fit(X, y).network_.initial_state
        keep = levels != "q"
        estimator = categorical_estimator(K=6, initial_state=state)
        with pytest.raises(ValueError, match=r"positions \(2,\) hold \(2,\) level.* holds \(3,\)"):
            estimator.fit(X[keep], y[keep])

    def test_a_warm_start_without_a_fitted_model_checks_the_state(self):
        X, y, levels, _ = categorical_problem()
        state = categorical_estimator().fit(X, y).network_.initial_state
        keep = levels != "q"
        estimator = categorical_estimator(initial_state=state, warm_start="resume")
        with (
            pytest.warns(UserWarning, match="has no fitted model"),
            pytest.raises(ValueError, match=r"positions \(2,\) hold \(2,\) level.* holds \(3,\)"),
        ):
            estimator.fit(X[keep], y[keep])

    def test_equal_level_counts_fit_as_network_fit_from_the_state(self):
        X, y, levels, recipe = categorical_problem()
        state = categorical_estimator().fit(X, y).network_.initial_state
        rows = slice(None, None, 2)
        estimator = categorical_estimator(initial_state=state).fit(X[rows], y[rows])
        X_cont = torch.as_tensor(X[rows, :2].astype(float), dtype=DTYPE, device=DEVICE)
        codes = {"p": 0, "q": 1, "r": 2}
        X_cat = (torch.tensor([codes[v] for v in levels[rows]], device=DEVICE),)
        target = torch.as_tensor((y[rows] == "b").astype(np.int64), device=DEVICE)
        direct = Network.fit(
            recipe,
            X_cont,
            target,
            X_cat=X_cat,
            initial_state=state,
            computation_dtype=DTYPE,
            max_iter=2,
            seed=0,
        )
        np.testing.assert_array_equal(
            estimator.predict_proba(X[rows]), direct.predict(X_cont, X_cat=X_cat).cpu().numpy()
        )

    def test_a_search_fold_lacking_a_level_of_a_shared_state_raises(self):
        X, y, levels, _ = categorical_problem()
        state = categorical_estimator().fit(X, y).network_.initial_state
        rows = np.arange(len(X))
        # The validation rows hold only levels the training rows hold, so scoring succeeds.
        training = rows[(levels != "q") & (rows < 40)]
        validation = rows[(levels != "q") & (rows >= 40)]
        search = GridSearchCV(
            categorical_estimator(),
            {"initial_state": [state]},
            cv=[(training, validation)],
            error_score="raise",
        )
        with pytest.raises(ValueError, match=r"positions \(2,\) hold \(2,\) level.* holds \(3,\)"):
            search.fit(X, y)


@FIXED_BUDGET
class TestStateClassCount:
    """A supplied state's classification head must hold as many classes as the fit's labels."""

    def test_rows_lacking_a_class_of_the_state_raise(self):
        x, y = class_problem()
        state, K = captured_class_state(x, y)
        keep = y != "b"
        estimator = class_estimator(K, initial_state=state)
        with pytest.raises(ValueError, match=r"labels hold 2 class.* holds 3"):
            estimator.fit(x[keep], y[keep])

    def test_rows_with_a_class_the_state_lacks_raise(self):
        x, y = class_problem()
        keep = y != "b"
        state, K = captured_class_state(x[keep], y[keep])
        estimator = class_estimator(K, initial_state=state)
        with pytest.raises(ValueError, match=r"labels hold 3 class.* holds 2"):
            estimator.fit(x, y)

    def test_equal_class_counts_fit_as_network_fit_from_the_state(self):
        x, y = class_problem()
        state, K = captured_class_state(x, y)
        rows = slice(None, None, 2)
        estimator = class_estimator(K, initial_state=state).fit(x[rows], y[rows])
        X_cont = torch.as_tensor(x[rows], dtype=DTYPE, device=DEVICE)
        codes = {"a": 0, "b": 1, "c": 2}
        target = torch.tensor([codes[label] for label in y[rows]], device=DEVICE)
        direct = Network.fit(
            Recipe.chain(Input(K=K, epsilon=0.01), ClassificationHead()),
            X_cont,
            target,
            initial_state=state,
            computation_dtype=DTYPE,
            max_iter=2,
            seed=0,
        )
        np.testing.assert_array_equal(
            estimator.predict_proba(x[rows]), direct.predict(X_cont).cpu().numpy()
        )

    def test_a_search_fold_lacking_a_class_of_a_shared_state_raises(self):
        x, y = class_problem()
        state, K = captured_class_state(x, y)
        rows = np.arange(len(x))
        # The validation rows hold only classes the training rows hold, so scoring succeeds.
        training = rows[(y != "b") & (rows < 60)]
        validation = rows[(y != "b") & (rows >= 60)]
        search = GridSearchCV(
            class_estimator(K),
            {"initial_state": [state]},
            cv=[(training, validation)],
            error_score="raise",
        )
        with pytest.raises(ValueError, match=r"labels hold 2 class.* holds 3"):
            search.fit(x, y)
