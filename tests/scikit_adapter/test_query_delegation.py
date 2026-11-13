"""Query staging and delegation through public Network operations only."""

import numpy as np
import pytest

from entlearn.scikit_adapter import EONRegressor

from ._fixtures import fitted_query_model


class TestQueryDelegation:
    def test_no_instance_weight_recovery_raises(self):
        model, X = fitted_query_model(EONRegressor, epsilon_T=float("inf"))
        with pytest.raises(ValueError, match="instance-weight recovery"):
            model.recover_instance_weights(X)

    def test_standard_importances_are_detached_fitted_weights(self):
        model, _ = fitted_query_model(EONRegressor)
        expected = model.network_.inspect("feature_weights")["input"].cpu().numpy()
        values = model.feature_importances_
        np.testing.assert_array_equal(values, expected)
        values.fill(-1)
        np.testing.assert_array_equal(model.feature_importances_, expected)
