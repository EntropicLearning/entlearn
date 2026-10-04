"""Query staging and delegation through public Network operations only."""

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import Network, PredictConfig
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import fitted_query_model


class TestQueryDelegation:
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize(
        "method,operation",
        [
            ("score_samples", "score_samples"),
            ("recover_instance_weights", "predict_with_details"),
            ("reconstruct", "reconstruct"),
        ],
    )
    def test_exactly_one_public_call_with_complete_policy(
        self, monkeypatch, estimator_type, method, operation
    ):
        model, X = fitted_query_model(estimator_type)
        original = getattr(Network, operation)
        calls = []
        policy = PredictConfig(predict_mode="iterative", max_iter=2)

        def observe(network, continuous, **kwargs):
            calls.append((continuous, kwargs))
            return original(network, continuous, **kwargs)

        monkeypatch.setattr(Network, operation, observe)
        getattr(model, method)(X, predict_config=policy)
        assert len(calls) == 1
        continuous, kwargs = calls[0]
        torch.testing.assert_close(continuous, torch.tensor(X, dtype=DTYPE, device=DEVICE))
        assert kwargs["predict_config"] is policy
        if method == "recover_instance_weights":
            assert kwargs["details"] == ("instance_weights",)

    @pytest.mark.parametrize("method", ["score_samples", "recover_instance_weights"])
    def test_no_instance_weight_recovery_raises_without_fabricating_weights(self, method):
        model, X = fitted_query_model(EONRegressor, epsilon_T=float("inf"))
        with pytest.raises(ValueError, match=r"recover|instance|scor"):
            getattr(model, method)(X)

    @pytest.mark.parametrize("unavailable", ["training_affiliations", "training_instance_weights"])
    def test_manifold_reporting_requires_both_row_bound_values(self, monkeypatch, unavailable):
        model, _ = fitted_query_model(EONRegressor, True)
        original = Network.inspect

        def inspect(network, name):
            if name == unavailable:
                raise ValueError("row-bound value unavailable")
            return original(network, name)

        monkeypatch.setattr(Network, "inspect", inspect)
        with pytest.raises(ValueError, match="row-bound value unavailable"):
            _ = model.feature_importances_

    def test_standard_importances_are_detached_fitted_weights(self):
        model, _ = fitted_query_model(EONRegressor)
        expected = model.network_.inspect("feature_weights")["input"].cpu().numpy()
        values = model.feature_importances_
        np.testing.assert_array_equal(values, expected)
        values.fill(-1)
        np.testing.assert_array_equal(model.feature_importances_, expected)
