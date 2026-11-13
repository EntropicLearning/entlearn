"""Observable classification behaviour at the tabular adapter."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import Coupling, Network, PredictConfig
from entlearn.scikit_adapter import EONClassifier

from ._fixtures import classifier_recipe, tabular_data, tensor_data


class TestClassifier:
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_tabular_fit_and_decoded_predictions_agree_with_tensor_fit(self, coupling):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        recipe = classifier_recipe(coupling=coupling)
        direct = Network.fit(recipe, X_t, y_t, seed=7, max_iter=100)

        estimator = EONClassifier(recipe, random_state=7, max_iter=100, dtype=DTYPE, device=DEVICE)
        estimator.fit(X, y)

        expected = direct.predict(X_t).cpu().numpy()
        np.testing.assert_array_equal(estimator.predict_proba(X), expected)
        np.testing.assert_array_equal(estimator.predict(X), np.unique(y)[expected.argmax(axis=1)])
        assert estimator.score(X, y) > 0.9
        assert estimator.network_.recipe is recipe
        assert estimator.loss_curve_ == direct.diagnostics.loss_history
        assert estimator.n_iter_ == direct.diagnostics.n_iter

    @pytest.mark.parametrize("predict_mode", ["single", "iterative"])
    def test_call_time_geometric_policy_keeps_the_fitted_temperature(self, predict_mode):
        rng = np.random.default_rng(5)
        X = rng.uniform(0.2, 0.8, (48, 2))
        y = np.where(X[:, 0] + rng.normal(0, 0.15, 48) > 0.5, "high", "low")
        estimator = EONClassifier(
            classifier_recipe(), random_state=7, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        fitted = estimator.network_.predict_config
        request = PredictConfig(predict_mode=predict_mode, output_mode="geometric")
        served = estimator.predict_proba(X, predict_config=request)

        expected = (
            estimator.predict_proba(X)
            if predict_mode == "single"
            else estimator.predict_proba(
                X, predict_config=replace(fitted, predict_mode="iterative")
            )
        )
        np.testing.assert_array_equal(served, expected)
        # An explicit epsilon_P = delta = 1, the uncalibrated temperature, still overrides.
        assert fitted.epsilon_P != 1.0
        explicit = estimator.predict_proba(X, predict_config=replace(request, epsilon_P=1.0))
        assert np.abs(explicit - served).max() > 1e3 * np.finfo(served.dtype).eps

    def test_mixed_features_replay_categories_in_original_column_order(self):
        X, y = tabular_data()
        mixed = np.column_stack([y, X[:, 0], np.repeat("constant", len(y)), X[:, 1]])
        estimator = EONClassifier(
            classifier_recipe(), categorical_features=[2, 0], dtype=DTYPE, device=DEVICE
        )
        with pytest.warns(UserWarning, match="single level"):
            estimator.fit(mixed, y)

        assert estimator.n_features_in_ == 4
        assert estimator.feature_layout_.continuous_indices == (1, 3)
        assert estimator.feature_layout_.categorical_indices == (0,)
        assert estimator.feature_layout_.dropped_indices == (2,)
        np.testing.assert_array_equal(estimator.feature_layout_.categories[0], np.unique(y))
        assert estimator.score(mixed, y) > 0.9
        changed = mixed.copy()
        changed[0, 0] = "unknown"
        with pytest.raises(ValueError, match="unseen"):
            estimator.predict(changed)


class TestClassifierControls:
    def test_constructor_parameters_are_stored_verbatim(self):
        recipe = classifier_recipe()
        controls = {
            "categorical_features": [0],
            "class_weights": {"east": 1.0},
            "warm_start": "resume",
            "n_inits": 3,
            "n_jobs": 2,
            "parallel_backend": "processes",
            "retain": "members",
            "scoring": "accuracy",
            "cv": 4,
            "init_rows": [0, 1],
            "predict_config": PredictConfig(epsilon_P=0.4),
            "max_iter": 7,
            "tol": 0.5,
            "random_state": 11,
            "device": "cpu",
            "dtype": torch.float32,
            "verbose": 2,
        }
        stored = EONClassifier(recipe, **controls).get_params(deep=False)
        assert {name: stored[name] for name in controls} == controls
        assert stored["recipe"] is recipe

    def test_requested_precision_and_tolerance_reach_the_network(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(),
            dtype=torch.float32,
            device=DEVICE,
            tol=1.0,
            max_iter=50,
            random_state=4,
        ).fit(X, y)

        assert estimator.network_.schema.computation_dtype == torch.float32
        assert estimator.predict_proba(X).dtype == np.float32
        # A tolerance that large stops the trajectory after its first update.
        assert estimator.n_iter_ == 1

    def test_an_invalid_verbosity_reaches_the_network_check(self):
        X, y = tabular_data()
        estimator = EONClassifier(classifier_recipe(), verbose=-1, dtype=DTYPE, device=DEVICE)
        with pytest.raises(ValueError, match="verbose must be"):
            estimator.fit(X, y)

    def test_target_rows_must_match_the_feature_rows(self):
        X, y = tabular_data()
        estimator = EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE)
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            estimator.fit(X, y[:-1])
