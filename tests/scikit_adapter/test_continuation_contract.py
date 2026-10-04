"""Continuation preserves fitted behaviour, diagnostics and failure atomicity."""

import warnings

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from entlearn import ConvergenceWarning, Coupling, Hidden, Network, PredictConfig
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import continuation_case, fitted_query_model, tensor_data


class TestLifecycle:
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("column", [False, True])
    def test_continuation_preserves_the_original_regression_response_rank(self, mode, column):
        model, X, y = continuation_case(EONRegressor, warm_start=mode)
        model.fit(X, y[:, None] if column else y)
        expected_shape = model.predict(X).shape
        model.set_params(max_iter=6).fit(X, y if column else y[:, None])
        assert model.predict(X).shape == expected_shape
        assert model.n_outputs_ == 1
        model.set_params(warm_start=False).fit(X, y if column else y[:, None])
        assert model.predict(X).shape == (y.shape if column else y[:, None].shape)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_resume_reproduces_uninterrupted_history_and_obeys_total_ceiling(self, estimator_type):
        model, X, y = continuation_case(estimator_type, warm_start="resume")
        full = clone(model).set_params(max_iter=8).fit(X, y)
        model.fit(X, y)
        history = model.loss_curve_
        model.fit(X, y)
        assert model.loss_curve_ == history
        model.set_params(max_iter=8).fit(X, y)
        assert model.loss_curve_ == full.loss_curve_
        assert model.n_iter_ == full.n_iter_
        np.testing.assert_array_equal(model.predict(X), full.predict(X))

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_failure_preserves_every_fitted_attribute(self, estimator_type, mode, monkeypatch):
        model, X, y = continuation_case(estimator_type, warm_start=mode)
        model.fit(X, y)
        before = vars(model).copy()
        prediction = model.predict(X)

        def fail(*args, **kwargs):
            raise RuntimeError("finalisation failed")

        monkeypatch.setattr(Network, mode, fail)
        with pytest.raises(RuntimeError, match="finalisation"):
            model.fit(X, y)
        assert vars(model).keys() == before.keys()
        assert all(vars(model)[name] is value for name, value in before.items())
        np.testing.assert_array_equal(model.predict(X), prediction)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_warning_as_error_preserves_the_source(self, estimator_type, mode):
        model, X, y = continuation_case(estimator_type, warm_start=mode, max_iter=1)
        model.fit(X, y)
        source = model.network_
        model.set_params(max_iter=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(ConvergenceWarning):
                model.fit(X, y)
        assert model.network_ is source

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_continuation_does_not_accept_a_different_prediction_policy(self, mode):
        model, X, y = continuation_case(EONClassifier, warm_start=mode)
        model.fit(X, y)
        source = model.network_
        model.set_params(predict_config=PredictConfig(output_mode="arithmetic"))
        with pytest.raises(ValueError, match="predict_config"):
            model.fit(X, y)
        assert model.network_ is source

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_continuation_reads_the_geometric_classification_default(self, mode):
        model, X, y = continuation_case(EONClassifier, warm_start=mode)
        model.fit(X, y)
        source = model.network_
        # An explicit geometric read-out is the policy the fit already derived.
        model.set_params(predict_config=PredictConfig(output_mode="geometric"), max_iter=6)
        model.fit(X, y)
        assert model.network_ is not source
        model.set_params(predict_config="geometric")
        with pytest.raises(ValueError, match="predict_config"):
            model.fit(X, y)

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_class_weights_reach_continuation_unchanged(self, mode):
        # Sharply unequal weights: on this balanced fixture near-uniform ones move
        # the loss by less than its last retained digit.
        weights = [0.98, 0.01, 0.01]
        model, X, y = continuation_case(
            EONClassifier, warm_start=mode, class_weights=weights, max_iter=1
        )
        model.fit(X, y)
        features, codes = tensor_data(X, y)
        expected = getattr(model.network_, mode)(
            features,
            codes,
            class_weights=torch.tensor(weights, dtype=DTYPE, device=DEVICE),
            max_iter=20,
            tol=0,
        )
        model.set_params(max_iter=20).fit(X, y)
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
            warm_start=mode,
            predict_config=PredictConfig(predict_mode="iterative", epsilon_P=epsilon_P, max_iter=3),
        )
        model.fit(X, y)
        source = model.network_
        features = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        codes = torch.as_tensor(np.unique(y, return_inverse=True)[1], device=DEVICE)
        expected = getattr(source, mode)(features, codes, max_iter=6, tol=0)
        model.set_params(max_iter=6).fit(X, y)
        assert model.network_.predict_config == expected.predict_config
        np.testing.assert_array_equal(
            model.predict_proba(X), expected.predict(features).cpu().numpy()
        )
        if epsilon_P is not None:
            assert model.network_.predict_config.epsilon_P == epsilon_P

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_regression_continuation_keeps_its_fitted_prediction_policy(self, mode):
        policy = PredictConfig(predict_mode="iterative", max_iter=3)
        model, X, y = continuation_case(EONRegressor, warm_start=mode, predict_config=policy)
        model.fit(X, y)
        model.set_params(max_iter=6).fit(X, y)
        assert model.network_.predict_config.predict_mode == "iterative"

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("supplied", [False, True])
    def test_explicit_configuration_cannot_change_temperature_provenance(self, mode, supplied):
        model, X, y = continuation_case(
            EONClassifier,
            warm_start=mode,
            predict_config=PredictConfig(epsilon_P=0.4) if supplied else None,
        )
        model.fit(X, y)
        source = model.network_
        replacement = PredictConfig(epsilon_P=None if supplied else source.predict_config.epsilon_P)
        model.set_params(predict_config=replacement)
        with pytest.raises(ValueError, match="predict_config"):
            model.fit(X, y)
        assert model.network_ is source
        # Omission, unlike an explicit configuration, requests no policy replacement.
        model.set_params(predict_config=None).fit(X, y)
        assert model.network_.predict_config.predict_mode == source.predict_config.predict_mode
        if supplied:
            assert model.network_.predict_config.epsilon_P == 0.4

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_cloned_explicit_selective_state_is_fresh_and_search_compatible(
        self, estimator_type, mode
    ):
        model, X, y = continuation_case(estimator_type)
        model.fit(X, y)
        state = model.network_.capture_current_state()
        model.set_params(initial_state=state, warm_start=mode)
        copied = clone(model)
        assert not hasattr(copied, "network_")
        assert copied.initial_state.block_names == state.block_names
        copied.fit(X, y)
        assert copied.n_iter_ <= copied.max_iter
        search = GridSearchCV(copied, {"max_iter": [2, 3]}, cv=2, error_score="raise").fit(X, y)
        assert search.best_estimator_.network_.can_resume
        assert copied.network_ is not model.network_

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_regression_weighting_reaches_continuation_unchanged(self, mode, monkeypatch):
        model, X, y = continuation_case(EONRegressor, warm_start=mode)
        model.fit(X, y)
        called = []

        def weighting(target):
            called.append(target.clone())
            return torch.ones(len(target), dtype=target.dtype, device=target.device)

        original = getattr(Network, mode)

        def operation(source, *args, **kwargs):
            assert kwargs["task_weights"] is weighting
            return original(source, *args, **kwargs)

        monkeypatch.setattr(Network, mode, operation)
        model.set_params(regression_weighting=weighting).fit(X, y)
        assert len(called) == 1
        torch.testing.assert_close(
            called[0].reshape(-1), torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
        )


class TestContinuedQueries:
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
        for name in ("loss_curve_", "n_iter_"):
            with pytest.raises(AttributeError):
                setattr(model, name, None)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_pruned_hidden_topology_is_not_reseeded(self, estimator_type, mode, coupling):
        model, X, y = continuation_case(estimator_type)
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
        model.set_params(warm_start=mode, max_iter=6).fit(X, y)
        assert (
            dict(model.network_.schema.K_active)["hidden_1"]
            <= dict(source.schema.K_active)["hidden_1"]
        )
        assert model.network_.initial_state.block_names == source.initial_state.block_names

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
