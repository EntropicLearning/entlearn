"""Original-space reconstruction through both public tabular estimators."""

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.exceptions import NotFittedError

from entlearn import (
    ClassificationHead,
    Coupling,
    Input,
    ManifoldInput,
    PredictConfig,
    Recipe,
    RegressionHead,
)
from entlearn.scikit_adapter import EONClassifier, EONRegressor, TabularReconstruction

from ._fixtures import regression_data, tabular_data


class TestReconstruction:
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("constant", [False, True])
    def test_categorical_reconstruction_preserves_distributions_and_vocabulary(
        self, estimator_type, constant
    ):
        X, labels = tabular_data()
        _, Y = regression_data()
        frame = pd.DataFrame(X, columns=["a", "b"]).assign(zone="constant" if constant else labels)
        head = (
            ClassificationHead(Coupling.M) if estimator_type is EONClassifier else RegressionHead()
        )
        estimator = estimator_type(
            Recipe.chain(Input(K=3), head),
            categorical_features="from_dtype",
            dtype=DTYPE,
            device=DEVICE,
            random_state=2,
        )
        if constant:
            with pytest.warns(UserWarning, match="single level"):
                estimator.fit(frame, labels if estimator_type is EONClassifier else Y)
        else:
            estimator.fit(frame, labels if estimator_type is EONClassifier else Y)
        result = estimator.reconstruct(frame)
        assert result.continuous_indices == (0, 1)
        assert result.categorical_indices == (2,)
        assert result.dropped_indices == ((2,) if constant else ())
        assert result.feature_names == ("a", "b", "zone")
        np.testing.assert_array_equal(
            result.categories[0], ["constant"] if constant else np.unique(labels)
        )
        continuous = torch.tensor(X, dtype=DTYPE, device=DEVICE)
        categorical = (
            ()
            if constant
            else (torch.tensor(np.unique(labels, return_inverse=True)[1], device=DEVICE),)
        )
        expected = estimator.network_.reconstruct(continuous, X_cat=categorical)
        np.testing.assert_array_equal(result.continuous, expected.continuous.cpu().numpy())
        np.testing.assert_array_equal(
            result.categorical[0],
            np.ones((len(X), 1)) if constant else expected.categorical[0].cpu().numpy(),
        )
        result.categories[0][0] = "changed"
        result.categorical[0].fill(-1)
        repeated = estimator.reconstruct(frame)
        assert repeated.categories[0][0] != "changed"
        assert np.all(repeated.categorical[0] >= 0)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("manifold", [False, True])
    @pytest.mark.parametrize("predict_mode", ["single", "iterative"])
    def test_reconstruct_delegates_in_original_feature_order(
        self, estimator_type, manifold, predict_mode
    ):
        X, labels = tabular_data()
        _, Y = regression_data()
        frame = pd.DataFrame({"northing": X[:, 1], "easting": X[:, 0]})
        input_block = (
            ManifoldInput(K=3, subspace_dimension=2, epsilon=0.05)
            if manifold
            else Input(K=3, epsilon=0.05)
        )
        head = (
            ClassificationHead(Coupling.M) if estimator_type is EONClassifier else RegressionHead()
        )
        estimator = estimator_type(
            Recipe.chain(input_block, head),
            dtype=DTYPE,
            device=DEVICE,
            random_state=2,
            predict_config=PredictConfig(predict_mode=predict_mode, max_iter=3, tol=0),
        )
        with pytest.raises(NotFittedError):
            estimator.reconstruct(frame)
        estimator.fit(frame, labels if estimator_type is EONClassifier else Y)
        expected = (
            estimator.network_.reconstruct(
                torch.tensor(np.array(frame, order="C"), dtype=DTYPE, device=DEVICE)
            )
            .continuous.cpu()
            .numpy()
        )
        original = frame.to_numpy(copy=True)
        prediction = estimator.predict(frame)
        result = estimator.reconstruct(frame)
        assert isinstance(result, TabularReconstruction)
        assert result.continuous_indices == (0, 1)
        assert result.categorical == result.categorical_indices == result.categories == ()
        assert result.dropped_indices == ()
        assert result.feature_names == ("northing", "easting")
        reconstructed = result.continuous
        assert reconstructed.dtype == expected.dtype
        assert not np.shares_memory(reconstructed, frame.to_numpy())
        np.testing.assert_array_equal(reconstructed, expected)
        repeated = estimator.reconstruct(frame).continuous
        assert not np.shares_memory(reconstructed, repeated)
        repeated.fill(-1)
        np.testing.assert_array_equal(reconstructed, expected)
        np.testing.assert_array_equal(frame.to_numpy(), original)
        np.testing.assert_array_equal(estimator.reconstruct(frame).continuous, reconstructed)
        np.testing.assert_array_equal(estimator.predict(frame), prediction)
        if manifold:
            # Round-off in this two-dimensional projection follows computation precision,
            # not the DataFrame's original storage dtype.
            tolerance = 8 * np.finfo(reconstructed.dtype).eps
            np.testing.assert_allclose(
                reconstructed, frame.to_numpy(), atol=tolerance, rtol=tolerance
            )
        assert not hasattr(estimator, "transform")
        assert not hasattr(estimator, "inverse_transform")
        with pytest.raises(ValueError, match="feature names"):
            estimator.reconstruct(frame.iloc[:, ::-1])
        estimator.set_params(recipe=None, device="invalid", dtype="invalid")
        np.testing.assert_array_equal(estimator.reconstruct(frame).continuous, reconstructed)
