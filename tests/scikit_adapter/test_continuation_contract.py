"""Continuation preserves fitted behaviour, diagnostics and failure atomicity."""

import warnings

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from entlearn import ConvergenceWarning, Coupling, Hidden, Network, PredictConfig
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import continuation_case, fitted_query_model, tensor_data

pytestmark = FIXED_BUDGET


class TestLifecycle:
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("column", [False, True])
    def test_continuation_preserves_the_original_regression_response_rank(self, mode, column):
        model, X, y = continuation_case(EONRegressor)
        model.fit(X, y[:, None] if column else y)
        expected_shape = model.predict(X).shape
        model.set_params(warm_start=mode, max_iter=6).fit(X, y if column else y[:, None])
        assert model.predict(X).shape == expected_shape
        assert model.n_outputs_ == 1
        model.set_params(warm_start=False).fit(X, y if column else y[:, None])
        assert model.predict(X).shape == (y.shape if column else y[:, None].shape)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_resume_reproduces_uninterrupted_history_and_obeys_total_ceiling(self, estimator_type):
        model, X, y = continuation_case(estimator_type)
        full = clone(model).set_params(max_iter=8).fit(X, y)
        model.fit(X, y)
        history = model.loss_curve_
        model.set_params(warm_start="resume").fit(X, y)
        assert model.loss_curve_ == history
        model.set_params(max_iter=8).fit(X, y)
        assert model.loss_curve_ == full.loss_curve_
        assert model.n_iter_ == full.n_iter_
        np.testing.assert_array_equal(model.predict(X), full.predict(X))

    def test_a_continuation_raising_a_warning_as_error_publishes_nothing(self):
        model, X, y = continuation_case(EONClassifier, max_iter=1)
        model.fit(X, y)
        model.set_params(warm_start="resume", max_iter=2)
        before = vars(model).copy()
        prediction = model.predict_proba(X)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(ConvergenceWarning):
                model.fit(X, y)
        assert vars(model).keys() == before.keys()
        assert all(vars(model)[name] is value for name, value in before.items())
        np.testing.assert_array_equal(model.predict_proba(X), prediction)

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_continuation_does_not_accept_a_different_prediction_policy(self, mode):
        model, X, y = continuation_case(EONClassifier)
        model.fit(X, y)
        source = model.network_
        model.set_params(warm_start=mode, predict_config=PredictConfig(output_mode="arithmetic"))
        with pytest.raises(ValueError, match="cannot change the fitted prediction policy"):
            model.fit(X, y)
        assert model.network_ is source

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_continuation_reads_the_geometric_classification_default(self, mode):
        model, X, y = continuation_case(EONClassifier)
        model.fit(X, y)
        source = model.network_
        # An explicit geometric read-out is the policy the fit already derived.
        model.set_params(
            warm_start=mode, predict_config=PredictConfig(output_mode="geometric"), max_iter=6
        )
        model.fit(X, y)
        assert model.network_ is not source

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_class_weights_reach_continuation_unchanged(self, mode):
        # Sharply unequal weights: on this balanced fixture near-uniform ones move
        # the loss by less than its last retained digit.
        weights = [0.98, 0.01, 0.01]
        model, X, y = continuation_case(EONClassifier, class_weights=weights, max_iter=1)
        model.fit(X, y)
        features, codes = tensor_data(X, y)
        expected = getattr(model.network_, mode)(
            features,
            codes,
            class_weights=torch.tensor(weights, dtype=DTYPE, device=DEVICE),
            max_iter=20,
            tol=0,
        )
        model.set_params(warm_start=mode, max_iter=20).fit(X, y)
        assert model.loss_curve_ == expected.diagnostics.loss_history
        np.testing.assert_array_equal(
            model.predict_proba(X), expected.predict(features).cpu().numpy()
        )

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("epsilon_P", [None, 0.4])
    def test_unchanged_explicit_policy_retains_derived_or_supplied_temperature(
        self, mode, epsilon_P
    ):
        model, X, y = continuation_case(
            EONClassifier,
            predict_config=PredictConfig(predict_mode="iterative", epsilon_P=epsilon_P, max_iter=3),
        )
        model.fit(X, y)
        source = model.network_
        features, codes = tensor_data(X, y)
        expected = getattr(source, mode)(features, codes, max_iter=6, tol=0)
        model.set_params(warm_start=mode, max_iter=6).fit(X, y)
        assert model.network_.predict_config == expected.predict_config
        np.testing.assert_array_equal(
            model.predict_proba(X), expected.predict(features).cpu().numpy()
        )
        if epsilon_P is not None:
            assert model.network_.predict_config.epsilon_P == epsilon_P

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_regression_continuation_keeps_its_fitted_prediction_policy(self, mode):
        policy = PredictConfig(predict_mode="iterative", max_iter=3)
        model, X, y = continuation_case(EONRegressor, predict_config=policy)
        model.fit(X, y)
        source = model.network_
        model.set_params(warm_start=mode, max_iter=6).fit(X, y)
        assert model.network_ is not source
        assert model.network_.predict_config == source.predict_config

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("supplied", [False, True])
    def test_explicit_configuration_cannot_change_temperature_provenance(self, mode, supplied):
        model, X, y = continuation_case(
            EONClassifier, predict_config=PredictConfig(epsilon_P=0.4) if supplied else None
        )
        model.fit(X, y)
        source = model.network_
        replacement = PredictConfig(epsilon_P=None if supplied else source.predict_config.epsilon_P)
        model.set_params(warm_start=mode, predict_config=replacement)
        with pytest.raises(ValueError, match="cannot change the fitted prediction policy"):
            model.fit(X, y)
        assert model.network_ is source
        # Omission, unlike an explicit configuration, requests no policy replacement.
        model.set_params(predict_config=None).fit(X, y)
        assert model.network_.predict_config.predict_mode == source.predict_config.predict_mode
        if supplied:
            assert model.network_.predict_config.epsilon_P == 0.4

    def test_an_unfitted_clone_with_a_captured_state_fits_fresh_from_that_state(self):
        model, X, y = continuation_case(EONClassifier)
        model.fit(X, y)
        state = model.network_.capture_current_state()
        copied = clone(model.set_params(initial_state=state, warm_start="resume"))
        with pytest.warns(UserWarning, match="no fitted model"):
            copied.fit(X, y)
        X_t, y_t = tensor_data(X, y)
        direct = Network.fit(model.recipe, X_t, y_t, initial_state=state, seed=7, max_iter=3, tol=0)
        # The initial loss tells the starting geometries apart; this fixture's
        # well-separated clusters converge to the same predictions from either.
        assert copied.loss_curve_ == direct.diagnostics.loss_history
        np.testing.assert_array_equal(copied.predict_proba(X), direct.predict(X_t).cpu().numpy())
        # An enclosing search fits warm-started clones without raising.
        with pytest.warns(UserWarning, match="warm_start='resume'.*no fitted model"):
            GridSearchCV(copied, {"max_iter": [2, 3]}, cv=2, error_score="raise").fit(X, y)

    @pytest.mark.filterwarnings("ignore:resume checkpoint loss differs:UserWarning")
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_regression_weighting_reaches_continuation_unchanged(self, mode):
        def weighting(target):
            return 0.1 + 10 * (target - target.min())

        model, X, y = continuation_case(EONRegressor, max_iter=1)
        model.fit(X, y)
        features = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        target = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
        continue_source = getattr(model.network_, mode)
        expected = continue_source(features, target, task_weights=weighting, max_iter=20, tol=0)
        unweighted = continue_source(features, target, max_iter=20, tol=0)
        assert expected.diagnostics.loss_history != unweighted.diagnostics.loss_history
        model.set_params(warm_start=mode, regression_weighting=weighting, max_iter=20).fit(X, y)
        assert model.loss_curve_ == expected.diagnostics.loss_history
        np.testing.assert_array_equal(
            model.predict(X), expected.predict(features).cpu().numpy()[:, 0]
        )


class TestContinuedQueries:
    @pytest.mark.filterwarnings("ignore:resume checkpoint loss differs:UserWarning")
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("manifold", [False, True])
    def test_queries_follow_weighted_continuation_through_the_public_network(
        self, estimator_type, mode, manifold
    ):
        model, X = fitted_query_model(estimator_type, manifold, sample_weight=True)
        source = model.network_
        rows = np.arange(len(X)) if mode == "resume" else np.arange(0, len(X), 2)
        features = torch.as_tensor(X[rows], dtype=DTYPE, device=DEVICE)
        labels = np.repeat(["east", "north", "west"], 16)[rows]
        target = (
            torch.as_tensor(np.searchsorted(model.classes_, labels), device=DEVICE)
            if estimator_type is EONClassifier
            else features[:, 0]
        )
        weights = np.linspace(0.2, 1.5, len(rows))
        expected = getattr(source, mode)(
            features,
            target,
            sample_weights=torch.as_tensor(weights, dtype=DTYPE, device=DEVICE),
            max_iter=8,
            tol=0,
        )
        model.set_params(warm_start=mode, max_iter=8, tol=0).fit(
            X[rows],
            labels if estimator_type is EONClassifier else X[rows, 0],
            sample_weight=weights,
        )
        query = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        np.testing.assert_array_equal(
            model.reconstruct(X).continuous, expected.reconstruct(query).continuous.cpu().numpy()
        )
        np.testing.assert_array_equal(
            model.score_samples(X), expected.score_samples(query).cpu().numpy()
        )
        np.testing.assert_array_equal(
            model.recover_instance_weights(X),
            expected.predict_with_details(
                query, details=("instance_weights",)
            ).instance_weights.cpu(),
        )
        assert model.loss_curve_ == expected.diagnostics.loss_history

    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_fine_tuning_a_pruned_hidden_topology_does_not_reseed_it(self, coupling):
        model, X, y = continuation_case(EONClassifier)
        model.set_params(
            recipe=model.recipe.chain(
                model.recipe.blocks[0],
                Hidden(K=8, epsilon=0),
                model.recipe.blocks[-1],
                coupling=coupling,
            )
        ).fit(X, y)
        source = model.network_
        assert dict(source.schema.K_active)["hidden_1"] < 8
        model.set_params(warm_start="fine_tune", max_iter=6).fit(X, y)
        assert (
            dict(model.network_.schema.K_active)["hidden_1"]
            <= dict(source.schema.K_active)["hidden_1"]
        )

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_explicit_dtype_migration_leaves_source_unchanged(self, estimator_type, mode):
        model, X, y = continuation_case(estimator_type, dtype=torch.float32, max_iter=1)
        model.fit(X, y)
        source = model.network_
        model.set_params(warm_start=mode, dtype=torch.float64, max_iter=6).fit(X, y)
        assert source.schema.computation_dtype is torch.float32
        assert model.network_.schema.computation_dtype is torch.float64
        result = model.predict_proba(X) if estimator_type is EONClassifier else model.predict(X)
        assert result.dtype == np.float64
