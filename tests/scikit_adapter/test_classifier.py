"""Observable classification behaviour at the tabular adapter."""

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
        assert estimator.fit(X, y) is estimator

        expected = direct.predict(X_t).cpu().numpy()
        np.testing.assert_array_equal(estimator.predict_proba(X), expected)
        np.testing.assert_array_equal(estimator.predict(X), np.unique(y)[expected.argmax(axis=1)])
        assert estimator.score(X, y) > 0.9
        assert estimator.network_.recipe is recipe
        assert estimator.loss_curve_ == direct.diagnostics.loss_history
        assert estimator.n_iter_ == direct.diagnostics.n_iter

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

    def test_reused_geometry_converts_to_the_requested_computation_precision(self):
        X, y = tabular_data()
        X_t, y_t = tensor_data(X, y)
        other = torch.float32 if torch.float64 == DTYPE else torch.float64
        state = Network.initialise(classifier_recipe(), X_t.to(other), y_t, seed=5)
        assert state.input_geometry.continuous_centroids.dtype == other

        estimator = EONClassifier(
            classifier_recipe(), initial_state=state, dtype=DTYPE, device=DEVICE
        ).fit(X, y)

        assert estimator.network_.initial_state.input_geometry.continuous_centroids.dtype == DTYPE

    @pytest.mark.parametrize(
        "controls,error,message",
        [
            ({"device": "bogus"}, RuntimeError, "device type"),
            ({"verbose": -1}, ValueError, "verbose must be a non-negative integer"),
        ],
    )
    def test_invalid_execution_controls_reach_their_validation(self, controls, error, message):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), **({"dtype": DTYPE, "device": DEVICE} | controls)
        )
        with pytest.raises(error, match=message):
            estimator.fit(X, y)

    def test_target_rows_must_match_the_feature_rows(self):
        X, y = tabular_data()
        estimator = EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE)
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            estimator.fit(X, y[:-1])
