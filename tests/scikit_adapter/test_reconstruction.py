"""Original-space reconstruction through both public tabular estimators."""

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import (
    ClassificationHead,
    Coupling,
    Input,
    ManifoldInput,
    PredictConfig,
    Recipe,
    RegressionHead,
)
from entlearn.scikit_adapter import EONClassifier, EONRegressor

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

    def test_reconstruct_delegates_in_original_feature_order(self):
        X, _ = tabular_data()
        _, Y = regression_data()
        frame = pd.DataFrame({"northing": X[:, 1], "easting": X[:, 0]})
        estimator = EONRegressor(
            Recipe.chain(ManifoldInput(K=3, subspace_dimension=1, epsilon=0.05), RegressionHead()),
            dtype=DTYPE,
            device=DEVICE,
            random_state=2,
            predict_config=PredictConfig(predict_mode="iterative", max_iter=3, tol=0),
        ).fit(frame, Y)
        expected = (
            estimator.network_.reconstruct(
                torch.tensor(np.array(frame, order="C"), dtype=DTYPE, device=DEVICE)
            )
            .continuous.cpu()
            .numpy()
        )
        result = estimator.reconstruct(frame)
        assert result.continuous_indices == (0, 1)
        assert result.categorical == result.categorical_indices == result.categories == ()
        assert result.dropped_indices == ()
        assert result.feature_names == ("northing", "easting")
        reconstructed = result.continuous
        assert reconstructed.dtype == expected.dtype
        np.testing.assert_array_equal(reconstructed, expected)
        estimator.reconstruct(frame).continuous.fill(-1)
        np.testing.assert_array_equal(estimator.reconstruct(frame).continuous, expected)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_no_transform_so_scikit_learn_does_not_treat_it_as_a_transformer(self, estimator_type):
        assert not hasattr(estimator_type, "transform")
        assert not hasattr(estimator_type, "inverse_transform")
