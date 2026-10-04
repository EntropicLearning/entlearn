"""Successful and failed public lifecycles, without private numerical access."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.base import clone

from entlearn import ConvergenceWarning, Input, Network, PredictConfig, Recipe, RegressionHead
from entlearn.scikit_adapter import EONClassifier, EONRegressor
from entlearn.scikit_adapter.base import _RANDOM_STATE_DRAW_HIGH

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
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_excess_input_width_warns_and_warning_as_error_preserves_the_previous_fit(
        self, estimator_type
    ):
        X, labels = tabular_data()
        X = X[[0, 16, 32]]
        y = labels[[0, 16, 32]] if estimator_type is EONClassifier else X[:, 0]
        recipe = classifier_recipe() if estimator_type is EONClassifier else regression_recipe()
        estimator = estimator_type(recipe, random_state=2, dtype=DTYPE, device=DEVICE).fit(X, y)
        previous = estimator.network_
        prediction = estimator.predict(X)
        source_name = recipe.blocks[0].name
        requested = recipe.replace_block(source_name, K=8)
        estimator.set_params(recipe=requested)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="input K=.*reducing", category=UserWarning)
            with pytest.raises(UserWarning, match="reducing"):
                estimator.fit(X, y)
        assert estimator.network_ is previous
        np.testing.assert_array_equal(estimator.predict(X), prediction)
        with pytest.warns(UserWarning, match=r"K=8.*3.*reducing"):
            assert estimator.fit(X, y) is estimator
        assert estimator.recipe == requested
        assert estimator.network_.initial_state.input_geometry.K_active == 3
        assert estimator.get_params()[f"recipe__blocks__{source_name}__K"] == 8
        assert estimator.predict(X).shape == y.shape

    @pytest.mark.parametrize("supplied", [False, True])
    def test_original_state_reuses_public_geometry_on_changed_rows_without_stateful_cloning(
        self, supplied
    ):
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
        before = estimator.predict_proba(X)
        state = estimator.network_.initial_state
        torch.testing.assert_close(
            state.input_geometry.continuous_centroids,
            original.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
        fresh = clone(estimator)
        assert not hasattr(fresh, "network_")
        if not supplied:
            assert fresh.initial_state is None
        fresh.set_params(initial_state=state).fit(X[::2], y[::2])
        direct = Network.fit(recipe, X_t[::2], y_t[::2], initial_state=state, seed=8)
        np.testing.assert_array_equal(fresh.predict_proba(X), direct.predict(X_t).cpu().numpy())
        np.testing.assert_array_equal(estimator.predict_proba(X), before)

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
        staged = []

        def record(*args, **kwargs):
            pairs = kwargs.get("validation_pairs") or ()
            staged.extend((*args[1:], *kwargs["X_cat"], *(rows for pair in pairs for rows in pair)))
            for name in ("class_weights", "init_rows"):
                if kwargs.get(name) is not None:
                    staged.append(kwargs[name])
            raise RuntimeError("delegated")

        monkeypatch.setattr(Network, "fit", record)
        monkeypatch.setattr(Network, "resume", record)
        # The meta device needs no accelerator, so it discriminates on the cpu lane.
        model.set_params(device="meta", cv=2, n_inits=2, init_rows=np.arange(6))
        with pytest.raises(RuntimeError, match="delegated"):
            model.fit(frame, y)
        model.set_params(warm_start="resume", n_inits=1)
        with pytest.raises(RuntimeError, match="delegated"):
            model.fit(frame, y)
        assert len(staged) == (13 if classifier else 11)
        assert {tensor.device.type for tensor in staged} == {"meta"}

    def test_failed_refit_preserves_network_labels_layout_and_names(self, monkeypatch):
        X, y = tabular_data()
        frame = pd.DataFrame(X, columns=["first", "second"]).assign(zone=y)
        estimator = EONClassifier(
            classifier_recipe(),
            categorical_features="from_dtype",
            dtype=DTYPE,
            device=DEVICE,
            random_state=6,
        ).fit(frame, y)
        fitted_network = estimator.network_
        prediction = estimator.predict_proba(frame)

        def fail(*args, **kwargs):
            raise RuntimeError("numerical lifecycle failed")

        with monkeypatch.context() as patch:
            patch.setattr(Network, "fit", fail)
            with pytest.raises(RuntimeError, match="lifecycle"):
                estimator.fit(X, np.repeat([10, 20, 40], 16))
        assert estimator.network_ is fitted_network
        np.testing.assert_array_equal(estimator.classes_, np.unique(y))
        np.testing.assert_array_equal(estimator.feature_names_in_, frame.columns)
        np.testing.assert_array_equal(estimator.feature_layout_.categories[0], np.unique(y))
        assert estimator.n_features_in_ == frame.shape[1]
        np.testing.assert_array_equal(estimator.predict_proba(frame), prediction)
        np.testing.assert_array_equal(estimator.predict(frame), y)
        with pytest.raises(ValueError, match="unseen"):
            estimator.predict(frame.assign(zone="new"))
        with pytest.raises(ValueError, match="feature names"):
            estimator.predict(frame.iloc[:, ::-1])
        estimator.fit(X, y)
        assert not hasattr(estimator, "feature_names_in_")

    @pytest.mark.parametrize("supplied_geometry", [False, True])
    @pytest.mark.parametrize("seed", [-1, 2**63, 2**64, True, np.bool_(True)])
    def test_invalid_random_state_is_rejected_even_with_supplied_geometry(
        self, supplied_geometry, seed
    ):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        recipe = classifier_recipe()
        initial = Network.initialise(recipe, X_t, y_t) if supplied_geometry else None
        estimator = EONClassifier(
            recipe, random_state=seed, initial_state=initial, dtype=DTYPE, device=DEVICE
        )
        # The adapter rejects its own parameter before delegating; Network's own
        # guard would report the same range under the name ``seed``.
        with pytest.raises(ValueError, match=r"^random_state must "):
            estimator.fit(X, y)

    def test_a_generator_random_state_draws_the_fitted_seed(self):
        X, y = tabular_data()
        drawn, fixed = (
            EONClassifier(classifier_recipe(), random_state=state, dtype=DTYPE, device=DEVICE).fit(
                X, y
            )
            for state in (
                np.random.RandomState(0),
                np.random.RandomState(0).randint(0, _RANDOM_STATE_DRAW_HIGH),
            )
        )
        assert drawn.loss_curve_ == fixed.loss_curve_
        np.testing.assert_array_equal(drawn.predict_proba(X), fixed.predict_proba(X))

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
            with pytest.raises(ConvergenceWarning, match="did not converge"):
                estimator.fit(X, y)
        if refit:
            assert estimator.network_ is previous
            np.testing.assert_array_equal(estimator.predict_proba(X), predictions)
            np.testing.assert_array_equal(estimator.classes_, np.unique(y))
        else:
            assert not hasattr(estimator, "network_")
            assert not hasattr(estimator, "classes_")

    def test_adapter_does_not_repeat_the_core_convergence_warning(self):
        X, y = tabular_data()
        with pytest.warns(ConvergenceWarning, match="did not converge") as seen:
            estimator = EONClassifier(
                classifier_recipe(), max_iter=1, tol=0, dtype=DTYPE, device=DEVICE, random_state=1
            ).fit(X, y)
        assert len(seen) == 1
        assert estimator.network_.diagnostics.warnings == (str(seen[0].message),)

    def test_fitted_views_are_read_only_and_constructor_changes_do_not_relocate_queries(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=2
        ).fit(X, y)
        before = estimator.predict_proba(X)
        estimator.set_params(dtype="not a dtype", device="not a device", recipe=None)
        np.testing.assert_array_equal(estimator.predict_proba(X), before)
        for attr in ("loss_curve_", "n_iter_"):
            with pytest.raises(AttributeError):
                setattr(estimator, attr, None)

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
        for query in (estimator.predict_proba, estimator.predict):
            with pytest.raises(ValueError, match="predict_config must be an exact PredictConfig"):
                query(X, predict_config="nonsense")


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
        assert estimator.score(X, y, sample_weight=weights) == pytest.approx(
            np.average(estimator.predict(X) == y, weights=weights)
        )

    @pytest.mark.parametrize(("weight", "rows"), [(-1.0, 0), (0.0, 0), (0.0, slice(None))])
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_non_positive_row_weights_are_rejected_before_delegation(
        self, monkeypatch, estimator_type, weight, rows
    ):
        if estimator_type is EONClassifier:
            (X, y), recipe = tabular_data(), classifier_recipe()
        else:
            (X, y), recipe = regression_data(), regression_recipe()

        def unexpected_fit(*args, **kwargs):
            raise AssertionError("Network.fit must not run on rejected sample weights")

        monkeypatch.setattr(Network, "fit", unexpected_fit)
        estimator = estimator_type(recipe, dtype=DTYPE, device=DEVICE, random_state=1)
        weights = np.ones(len(y))
        weights[rows] = weight
        with pytest.raises(ValueError, match="must be positive; drop zero-weight rows instead"):
            estimator.fit(X, y, sample_weight=weights)
        assert not hasattr(estimator, "network_")

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
            ([1], "length-3"),
            ([-1, 1, 1], "non-negative"),
            ([1, np.nan, 1], "finite"),
            ({"east": 1}, "missing a weight for class"),
            ({"east": 1, "north": 1, "west": 1, "south": 1}, "unknown class"),
        ],
    )
    def test_invalid_class_weights_raise(self, weights, message):
        X, y = tabular_data()
        with pytest.raises(ValueError, match=message):
            EONClassifier(classifier_recipe(), class_weights=weights).fit(X, y)
