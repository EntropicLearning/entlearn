"""Fitted query behaviour through the estimator boundary."""

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from entlearn import Network, PredictConfig
from entlearn.helpers.reporting import active_features, count_parameters, effective_dimensions
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import fitted_query_model

# Every name that needs a fitted Network, by how it is used: an attribute is
# read, a query is called with rows and a report is called without arguments.
# A name the estimator type does not define is skipped; n_outputs_ is
# regressor-only, and classes_ is absent because it raises AttributeError.
_FITTED_NAMES = {
    "n_iter_": "attribute",
    "loss_curve_": "attribute",
    "feature_importances_": "attribute",
    "n_outputs_": "attribute",
    "score_samples": "query",
    "recover_instance_weights": "query",
    "reconstruct": "query",
    "active_features": "report",
    "count_parameters": "report",
    "effective_dimensions": "report",
}


class TestQueries:
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("manifold", [False, True])
    def test_single_pass_recovery_preserves_rows_and_batching(self, estimator_type, manifold):
        model, X = fitted_query_model(estimator_type, manifold)
        query = X * 0.9 + 0.03
        weights = model.recover_instance_weights(query)
        tolerance = 512 * torch.finfo(DTYPE).eps
        np.testing.assert_allclose(
            model.recover_instance_weights(query[::-1])[::-1],
            weights,
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_allclose(
            np.concatenate([model.recover_instance_weights(row[None, :]) for row in query]),
            weights,
            atol=tolerance,
            rtol=tolerance,
        )
        # These queries differ from the training reference; no exact reference
        # ties are required to survive floating-point changes in matrix batching.
        expected = model.score_samples(query)
        np.testing.assert_array_equal(model.score_samples(query[::-1])[::-1], expected)
        np.testing.assert_array_equal(
            np.concatenate([model.score_samples(row[None, :]) for row in query]), expected
        )

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("manifold", [False, True])
    def test_query_recovery_and_scores_delegate_complete_policy(self, estimator_type, manifold):
        model, X = fitted_query_model(estimator_type, manifold)
        tensor = torch.tensor(X, dtype=DTYPE, device=DEVICE)
        for policy in (None, PredictConfig(predict_mode="iterative", max_iter=2, tol=0)):
            expected = model.network_.predict_with_details(
                tensor, predict_config=policy, details=("instance_weights",)
            ).instance_weights
            actual = model.recover_instance_weights(X, predict_config=policy)
            np.testing.assert_array_equal(actual, expected.cpu().numpy())
            np.testing.assert_array_equal(
                model.score_samples(X, predict_config=policy),
                model.network_.score_samples(tensor, predict_config=policy).cpu().numpy(),
            )
            assert actual.dtype == expected.cpu().numpy().dtype

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("manifold", [False, True])
    def test_reports_use_current_network(self, estimator_type, manifold):
        model, _ = fitted_query_model(estimator_type, manifold)
        expected = effective_dimensions(model.network_, normalise=False)
        assert model.effective_dimensions(normalise=False) == {
            name: {field: float(value) for field, value in values.items()}
            for name, values in expected.items()
        }
        assert model.effective_dimensions() == {
            name: {field: float(value) for field, value in values.items()}
            for name, values in effective_dimensions(model.network_, normalise=True).items()
        }
        assert model.effective_dimensions() != model.effective_dimensions(normalise=False)
        assert model.count_parameters() == count_parameters(model.network_)
        if not manifold:
            np.testing.assert_array_equal(
                model.active_features(tol=0.5),
                active_features(model.network_, tol=0.5).cpu().numpy(),
            )
            np.testing.assert_array_equal(
                model.active_features(), active_features(model.network_, tol=1.0).cpu().numpy()
            )
        else:
            with pytest.raises(ValueError):
                model.active_features()

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("raw", [False, True])
    @pytest.mark.parametrize("include_affiliations", [False, True])
    @pytest.mark.parametrize("active_tol", [None, 0.5])
    def test_counting_forwards_all_options(
        self, estimator_type, raw, include_affiliations, active_tol
    ):
        model, _ = fitted_query_model(estimator_type, False)
        options = dict(raw=raw, include_affiliations=include_affiliations, active_tol=active_tol)
        assert model.count_parameters(**options) == count_parameters(model.network_, **options)

    @pytest.mark.parametrize("operation", [active_features, count_parameters])
    def test_tensor_helpers_do_not_unwrap_estimators(self, operation):
        model, _ = fitted_query_model(EONClassifier, False)
        with pytest.raises(ValueError, match="requires a fitted Network"):
            operation(model)

    @pytest.mark.parametrize(
        ("estimator_type", "n_names"), [(EONClassifier, 9), (EONRegressor, 10)]
    )
    def test_fitted_names_require_fitted_estimator(self, estimator_type, n_names):
        fitted, X = fitted_query_model(estimator_type, False)
        del fitted.network_
        exercised = 0
        for name, kind in _FITTED_NAMES.items():
            if not hasattr(estimator_type, name):
                continue
            exercised += 1
            for model in (clone(fitted), fitted):
                with pytest.raises(NotFittedError):
                    member = getattr(model, name)
                    if kind == "query":
                        member(X)
                    elif kind == "report":
                        member()
        assert exercised == n_names

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_prediction_only_reporting_keeps_its_capability_limits(self, tmp_path, estimator_type):
        model, _ = fitted_query_model(estimator_type, False)
        expected_count = model.count_parameters()
        expected_active = model.active_features(tol=0.5)
        path = tmp_path / "prediction.safetensors"
        model.network_.save(path)
        model.network_ = Network.load(path, device=DEVICE)
        assert model.count_parameters() == expected_count
        np.testing.assert_array_equal(model.active_features(tol=0.5), expected_active)
        with pytest.raises(ValueError, match="unavailable"):
            model.count_parameters(include_affiliations=True)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("epsilon_T", [float("inf"), 0.5])
    @pytest.mark.parametrize("sample_weight", [False, True])
    def test_manifold_importance_uses_fitted_instance_weights(
        self, estimator_type, epsilon_T, sample_weight
    ):
        model, _ = fitted_query_model(
            estimator_type, True, epsilon_T=epsilon_T, sample_weight=sample_weight
        )
        network = model.network_
        basis = network.inspect("manifold_projectors")["input"]
        gamma = network.inspect("training_affiliations")["input"]
        weights = network.inspect("training_instance_weights")["input"]
        expected = (
            sum(
                (weights * gamma[:, k]).sum() / weights.sum() * (basis[k] @ basis[k].T).diagonal()
                for k in range(basis.shape[0])
            )
            / basis.shape[2]
        )
        np.testing.assert_allclose(model.feature_importances_, expected.cpu().numpy(), rtol=1e-6)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1, rtol=1e-6)
        assert not hasattr(model, "manifold_projectors")
        model.set_params(recipe=None)
        np.testing.assert_allclose(model.feature_importances_, expected.cpu().numpy(), rtol=1e-6)
