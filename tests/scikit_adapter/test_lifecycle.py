"""Successful and failed public lifecycles, without private numerical access."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from sklearn.base import clone

from entlearn import ConvergenceWarning, Input, Network, PredictConfig, Recipe, RegressionHead
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import (
    classifier_recipe,
    continuation_case,
    continuation_frame,
    regression_data,
    regression_recipe,
    tabular_data,
    tensor_data,
)


class TestLifecycle:
    @pytest.mark.parametrize("supplied", [False, True])
    def test_a_reused_fitted_state_on_other_rows_matches_network_fit(self, supplied):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        recipe = classifier_recipe()
        original = Network.initialise(recipe, X_t, y_t, seed=8)
        estimator = EONClassifier(
            recipe,
            initial_state=original if supplied else None,
            random_state=8,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)
        state = estimator.network_.initial_state
        torch.testing.assert_close(
            state.input_geometry.continuous_centroids,
            original.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
        fresh = clone(estimator).set_params(initial_state=state).fit(X[::2], y[::2])
        direct = Network.fit(recipe, X_t[::2], y_t[::2], initial_state=state, seed=8)
        np.testing.assert_array_equal(fresh.predict_proba(X), direct.predict(X_t).cpu().numpy())

    def test_required_recipe_checks(self):
        X, y = tabular_data()
        with pytest.raises(TypeError, match="recipe"):
            EONClassifier()  # ty: ignore[missing-argument]
        for recipe, message in (
            (None, "must be a valid Recipe"),
            ("default", "must be a valid Recipe"),
            (
                Recipe.chain(Input(), RegressionHead()),
                "EONClassifier requires a classification Recipe",
            ),
        ):
            with pytest.raises(ValueError, match=message):
                EONClassifier(recipe).fit(X, y)

    @FIXED_BUDGET
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_fit_and_resume_stage_every_tensor_on_the_requested_device(
        self, monkeypatch, estimator_type
    ):
        classifier = estimator_type is EONClassifier
        weights = {"class_weights": [1.0, 2.0, 3.0]} if classifier else {}
        model, X, y = continuation_case(
            estimator_type, categorical_features="from_dtype", **weights
        )
        frame = continuation_frame(X)
        model.fit(frame, y)
        staged = {}

        def record(*args, **kwargs):
            staged.update(X_cont=(args[1],), y=(args[2],), X_cat=tuple(kwargs["X_cat"]))
            if kwargs.get("validation_pairs") is not None:
                staged["validation_pairs"] = tuple(
                    rows for pair in kwargs["validation_pairs"] for rows in pair
                )
            for name in ("class_weights", "init_rows"):
                if kwargs.get(name) is not None:
                    staged[name] = (kwargs[name],)
            raise RuntimeError("delegated")

        monkeypatch.setattr(Network, "fit", record)
        monkeypatch.setattr(Network, "resume", record)
        # The meta device needs no accelerator, so it discriminates on the cpu lane.
        model.set_params(device="meta", cv=2, n_inits=2, init_rows=np.arange(6))
        continued = {"X_cont", "y", "X_cat"} | ({"class_weights"} if classifier else set())
        for mode, expected in (
            (False, continued | {"validation_pairs", "init_rows"}),
            ("resume", continued),
        ):
            staged.clear()
            model.set_params(warm_start=mode, n_inits=2 if mode is False else 1)
            with pytest.raises(RuntimeError, match="delegated"):
                model.fit(frame, y)
            assert staged.keys() == expected
            assert {tensor.device.type for group in staged.values() for tensor in group} == {"meta"}

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_a_failed_fresh_fit_publishes_nothing(self, estimator_type):
        # The refit would change every published attribute: the feature names and
        # layout, and the label vocabulary or the target rank.
        X, labels = tabular_data()
        frame = pd.DataFrame(X, columns=["first", "second"]).assign(zone=labels)
        classifier = estimator_type is EONClassifier
        estimator = estimator_type(
            classifier_recipe() if classifier else regression_recipe(),
            categorical_features="from_dtype",
            dtype=DTYPE,
            device=DEVICE,
            random_state=6,
        ).fit(frame, labels if classifier else X[:, 0])

        def fail(*args, **kwargs):
            raise RuntimeError("scorer failed")

        estimator.set_params(scoring=fail)
        before = vars(estimator).copy()
        prediction = estimator.predict(frame)
        with pytest.raises(RuntimeError, match="scorer failed"):
            estimator.fit(X, np.repeat([10, 20, 40], 16) if classifier else X)
        assert vars(estimator).keys() == before.keys()
        assert all(vars(estimator)[name] is value for name, value in before.items())
        np.testing.assert_array_equal(estimator.predict(frame), prediction)

    def test_a_fresh_unnamed_fit_forgets_the_previous_feature_names(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=6
        ).fit(pd.DataFrame(X, columns=["first", "second"]), y)
        estimator.fit(X, y)
        assert not hasattr(estimator, "feature_names_in_")

    @pytest.mark.parametrize(
        ("seed", "supplied_geometry", "message"),
        [
            (-1, False, "random_state must be an integer"),
            (2**63, False, "random_state must be an integer"),
            (True, False, "random_state must not"),
            (np.bool_(True), False, "random_state must not"),
            (2**63, True, "random_state must be an integer"),
        ],
        ids=["negative", "too-large", "bool", "numpy-bool", "too-large-with-geometry"],
    )
    def test_invalid_random_state_is_rejected_even_with_supplied_geometry(
        self, seed, supplied_geometry, message
    ):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        recipe = classifier_recipe()
        initial = Network.initialise(recipe, X_t, y_t) if supplied_geometry else None
        estimator = EONClassifier(
            recipe, random_state=seed, initial_state=initial, dtype=DTYPE, device=DEVICE
        )
        # Network's own guard reports the same range under the name ``seed``.
        with pytest.raises(ValueError, match=message):
            estimator.fit(X, y)

    def test_a_random_state_generator_draws_a_reproducible_seed(self):
        X, y = tabular_data()

        def loss_curve(random_state):
            return (
                EONClassifier(
                    classifier_recipe(), random_state=random_state, dtype=DTYPE, device=DEVICE
                )
                .fit(X, y)
                .loss_curve_
            )

        first = loss_curve(np.random.RandomState(0))
        assert loss_curve(np.random.RandomState(0)) == first
        assert loss_curve(np.random.RandomState(1)) != first

    @pytest.mark.parametrize("refit", [False, True])
    def test_warning_as_error_does_not_publish_a_partial_fit(self, refit):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), max_iter=2, tol=0, dtype=DTYPE, device=DEVICE, random_state=1
        )
        if refit:
            estimator.set_params(max_iter=100, tol=1e-4).fit(X, y)
            previous = estimator.network_
            predictions = estimator.predict_proba(X)
            estimator.set_params(max_iter=2, tol=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(ConvergenceWarning):
                estimator.fit(X, y)
        if refit:
            assert estimator.network_ is previous
            np.testing.assert_array_equal(estimator.predict_proba(X), predictions)
            np.testing.assert_array_equal(estimator.classes_, np.unique(y))
        else:
            assert not hasattr(estimator, "network_")
            assert not hasattr(estimator, "classes_")

    def test_constructor_changes_do_not_relocate_queries(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=2
        ).fit(X, y)
        before = estimator.predict_proba(X)
        estimator.set_params(dtype="not a dtype", device="not a device", recipe=None)
        np.testing.assert_array_equal(estimator.predict_proba(X), before)

    def test_initial_state_and_complete_prediction_policy_delegate_to_network(self):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        recipe = classifier_recipe()
        state = Network.initialise(recipe, X_t, y_t, seed=5)
        policy = PredictConfig(epsilon_P=0.4)
        # A supplied state bypasses the selection folds, but its scoring is still resolved.
        with pytest.raises(ValueError, match="not_a_scorer"):
            EONClassifier(
                recipe, initial_state=state, scoring="not_a_scorer", dtype=DTYPE, device=DEVICE
            ).fit(X, y)
        estimator = EONClassifier(
            recipe,
            initial_state=state,
            predict_config=policy,
            cv=5,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(X, y)
        direct = Network.fit(recipe, X_t, y_t, initial_state=state, predict_config=policy)
        estimator.set_params(predict_config=PredictConfig(epsilon_P=0.8))
        np.testing.assert_array_equal(estimator.predict_proba(X), direct.predict(X_t).cpu().numpy())
        override = PredictConfig(output_mode="arithmetic")
        np.testing.assert_array_equal(
            estimator.predict_proba(X, predict_config=override),
            direct.predict(X_t, predict_config=override).cpu().numpy(),
        )
        with pytest.raises(ValueError, match="exact PredictConfig"):
            estimator.predict_proba(X, predict_config="nonsense")


class TestWeights:
    @pytest.mark.parametrize("mapping", [True, False])
    def test_row_and_class_weights_reach_the_complete_tensor_lifecycle(self, mapping):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        weights = np.linspace(0.2, 1, len(y))
        class_weights = {"west": 0.5, "east": 0.2, "north": 0.3} if mapping else [0.2, 0.3, 0.5]
        estimator = EONClassifier(
            classifier_recipe(),
            class_weights=class_weights,
            dtype=DTYPE,
            device=DEVICE,
            random_state=9,
        ).fit(X, y, sample_weight=weights)
        direct = Network.fit(
            estimator.recipe,
            X_t,
            y_t,
            seed=9,
            sample_weights=torch.as_tensor(weights, dtype=DTYPE, device=DEVICE),
            class_weights=torch.tensor([0.2, 0.3, 0.5], dtype=DTYPE, device=DEVICE),
        )
        np.testing.assert_array_equal(estimator.predict_proba(X), direct.predict(X_t).cpu().numpy())
        assert estimator.network_.predict_config == direct.predict_config

    def test_a_zero_row_weight_reaches_the_network_check(self):
        X, y = tabular_data()
        weights = np.ones(len(y))
        weights[0] = 0.0
        estimator = EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE)
        with pytest.raises(ValueError, match="must be positive"):
            estimator.fit(X, y, sample_weight=weights)

    @pytest.mark.parametrize(
        ("controls", "weight", "message"),
        [
            ({"dtype": torch.float32}, 1e39, "too large for dtype"),
            ({"dtype": DTYPE, "device": "meta"}, 1.0, "must be on a device that contains data"),
        ],
    )
    def test_row_weights_are_staged_in_the_requested_precision_and_device(
        self, controls, weight, message
    ):
        X, y = regression_data()
        weights = np.ones(len(y))
        weights[0] = weight
        estimator = EONRegressor(regression_recipe(), **({"device": DEVICE} | controls))
        with pytest.raises(ValueError, match=message):
            estimator.fit(X, y, sample_weight=weights)

    def test_label_keyed_integer_weights_order_and_stage_to_the_computation_dtype(self):
        X, y = tabular_data()
        ordered = EONClassifier(
            classifier_recipe(),
            class_weights=[1.0, 2.0, 3.0],
            dtype=DTYPE,
            device=DEVICE,
            random_state=9,
        ).fit(X, y)
        keyed = EONClassifier(
            classifier_recipe(),
            class_weights={"west": 3, "east": 1, "north": 2},
            dtype=DTYPE,
            device=DEVICE,
            random_state=9,
        ).fit(X, y)
        np.testing.assert_array_equal(keyed.predict_proba(X), ordered.predict_proba(X))

    @pytest.mark.parametrize(
        "weights,message",
        [
            ([-1, 1, 1], "non-negative"),
            ({"east": 1}, "missing a weight for class"),
            ({"east": 1, "north": 1, "west": 1, "south": 1}, "unknown class"),
        ],
    )
    def test_invalid_class_weights_raise(self, weights, message):
        X, y = tabular_data()
        with pytest.raises(ValueError, match=message):
            EONClassifier(classifier_recipe(), class_weights=weights).fit(X, y)
