"""Regression through the public tabular estimator and tensor lifecycle."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.base import clone

from entlearn import Network, PredictConfig
from entlearn.scikit_adapter import EONRegressor

from ._fixtures import classifier_recipe, regression_data, regression_recipe


class TestRegressor:
    @pytest.mark.parametrize("target_shape", ["vector", "column", "multi"])
    def test_fit_predict_preserves_target_shape_and_matches_tensor_lifecycle(self, target_shape):
        X, Y = regression_data()
        y = Y[:, 0] if target_shape == "vector" else Y[:, :1] if target_shape == "column" else Y
        estimator = EONRegressor(regression_recipe(), random_state=4, dtype=DTYPE, device=DEVICE)
        estimator.fit(X, y)
        X_t = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        direct = Network.fit(
            estimator.recipe, X_t, torch.as_tensor(y, dtype=DTYPE, device=DEVICE), seed=4
        )
        prediction = estimator.predict(X)
        expected = direct.predict(X_t).cpu().numpy()
        if target_shape == "vector":
            expected = expected[:, 0]
        np.testing.assert_array_equal(prediction, expected)
        assert prediction.shape == y.shape
        assert estimator.n_outputs_ == (1 if y.ndim == 1 else y.shape[1])

    def test_a_classification_recipe_is_refused(self):
        X, y = regression_data()
        with pytest.raises(ValueError, match="EONRegressor requires a regression Recipe"):
            EONRegressor(classifier_recipe()).fit(X, y)

    def test_a_call_time_policy_replaces_the_fitted_policy_for_prediction(self):
        X, y = regression_data()
        estimator = EONRegressor(
            regression_recipe(), random_state=3, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        X_t = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        override = PredictConfig(predict_mode="iterative", max_iter=3, tol=0)
        expected = estimator.network_.predict(X_t, predict_config=override).cpu().numpy()
        assert not np.array_equal(expected, estimator.predict(X))
        np.testing.assert_array_equal(estimator.predict(X, predict_config=override), expected)

    def test_a_classification_policy_is_refused(self):
        X, y = regression_data()
        estimator = EONRegressor(
            regression_recipe(), predict_config=PredictConfig(output_mode="geometric")
        )
        with pytest.raises(ValueError, match="no classification read-out"):
            estimator.fit(X, y)


class TestRegressionStaging:
    @pytest.mark.parametrize("layout", ["readonly", "reversed"])
    def test_target_layout_does_not_warn_or_change_values(self, monkeypatch, layout):
        X, Y = regression_data()
        if layout == "readonly":
            Y.flags.writeable = False
        else:
            Y = Y[::-1]
        staged = []
        real_fit = Network.fit

        def fit(recipe, X_cont, y, **kwargs):
            staged.append(y)
            return real_fit(recipe, X_cont, y, **kwargs)

        monkeypatch.setattr(Network, "fit", fit)
        estimator = EONRegressor(regression_recipe(), dtype=DTYPE, device=DEVICE, random_state=2)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The given NumPy array is not writable")
            estimator.fit(X, Y)
        assert not np.shares_memory(staged[0].cpu().numpy(), Y)
        copied = clone(estimator).fit(X, Y.copy())
        np.testing.assert_array_equal(estimator.predict(X), copied.predict(X))

    @pytest.mark.parametrize("backend", ["numpy", "pandas", "polars"])
    @pytest.mark.parametrize("storage_dtype", [np.int64, np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_numeric_features_and_targets_reach_network_in_computation_dtype(
        self, monkeypatch, backend, storage_dtype, dtype
    ):
        X, Y = regression_data()
        X = (
            (X > 0.5).astype(storage_dtype)
            if storage_dtype == np.int64
            else X.astype(storage_dtype)
        )
        Y = (Y * 100).astype(storage_dtype)
        if backend == "pandas":
            X, Y = pd.DataFrame(X), pd.DataFrame(Y)
        elif backend == "polars":
            X, Y = pl.DataFrame(X), pl.DataFrame(Y)
        expected_X = torch.tensor(
            np.array(X, dtype=np.float32 if dtype == torch.float32 else np.float64),
            dtype=dtype,
            device=DEVICE,
        )
        expected_y = torch.tensor(
            np.array(Y, dtype=np.float32 if dtype == torch.float32 else np.float64),
            dtype=dtype,
            device=DEVICE,
        )
        calls = []

        def fit(recipe, X_cont, y, **kwargs):
            torch.testing.assert_close(X_cont, expected_X, rtol=0, atol=0)
            torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
            assert kwargs["computation_dtype"] == dtype
            calls.append(True)
            raise RuntimeError("public fit reached")

        monkeypatch.setattr(Network, "fit", fit)
        with pytest.raises(RuntimeError, match="public fit reached"):
            EONRegressor(regression_recipe(), dtype=dtype, device=DEVICE).fit(X, Y)
        assert calls == [True]

    @pytest.mark.parametrize("frame_type", [pd.DataFrame, pl.DataFrame])
    def test_mixed_integer_and_float_frames_round_each_column_once(self, monkeypatch, frame_type):
        X, y = regression_data()
        large = 2**60 + 2**36 + 1
        frame = frame_type({"count": np.full(len(X), large, dtype=np.int64), "value": X[:, 0]})
        staged = []

        def fit(recipe, X_cont, y, **kwargs):
            staged.append(X_cont)
            raise RuntimeError("public fit reached")

        monkeypatch.setattr(Network, "fit", fit)
        with pytest.raises(RuntimeError, match="public fit reached"):
            EONRegressor(regression_recipe(), dtype=torch.float32, device=DEVICE).fit(frame, y)
        # A float64 intermediate would round the integer twice, down to 2**60.
        expected = torch.full((len(X),), 2**60 + 2**37, dtype=torch.float32)
        torch.testing.assert_close(staged[0][:, 0].cpu(), expected, rtol=0, atol=0)

    def test_mixed_frames_share_vocabularies_and_predict_in_fitted_precision(self):
        X, Y = regression_data()
        categories = np.repeat(["north", "east", "west"], 16)
        data = {"zone": categories, "a": X[:, 1], "b": X[:, 0]}
        frame = pd.DataFrame(data)
        estimator = EONRegressor(
            regression_recipe(),
            categorical_features="from_dtype",
            dtype=DTYPE,
            device=DEVICE,
            random_state=3,
        ).fit(frame, Y)
        prediction = estimator.predict(frame)
        np.testing.assert_array_equal(estimator.predict(pl.DataFrame(data)), prediction)
        direct = Network.fit(
            estimator.recipe,
            torch.tensor(np.column_stack((X[:, 1], X[:, 0])), dtype=DTYPE, device=DEVICE),
            torch.tensor(Y, dtype=DTYPE, device=DEVICE),
            X_cat=[
                torch.tensor(
                    np.unique(categories, return_inverse=True)[1], dtype=torch.int64, device=DEVICE
                )
            ],
            seed=3,
        )
        expected = direct.predict(
            torch.tensor(np.column_stack((X[:, 1], X[:, 0])), dtype=DTYPE, device=DEVICE),
            X_cat=[
                torch.tensor(
                    np.unique(categories, return_inverse=True)[1], dtype=torch.int64, device=DEVICE
                )
            ],
        )
        np.testing.assert_array_equal(prediction, expected.cpu().numpy())
        assert prediction.dtype == (np.float32 if DTYPE is torch.float32 else np.float64)

    def test_fully_missing_target_rows_are_unlabelled(self):
        X, Y = regression_data()
        Y[::3] = np.nan
        estimator = EONRegressor(
            regression_recipe(), dtype=DTYPE, device=DEVICE, random_state=5
        ).fit(X, Y)
        direct = Network.fit(
            estimator.recipe,
            torch.tensor(X, dtype=DTYPE, device=DEVICE),
            torch.tensor(Y, dtype=DTYPE, device=DEVICE),
            seed=5,
        )
        np.testing.assert_array_equal(
            estimator.predict(X),
            direct.predict(torch.tensor(X, dtype=DTYPE, device=DEVICE)).cpu().numpy(),
        )

    @pytest.mark.parametrize(
        ("target", "message"),
        [
            ("short", "inconsistent numbers of samples"),
            ("partial", "partially-NaN"),
            ("unlabelled", "y must contain a labelled row"),
            ("none", "requires y"),
        ],
    )
    def test_invalid_targets_are_refused(self, target, message):
        X, Y = regression_data()
        partial = Y.copy()
        partial[1, 0] = np.nan
        invalid = {
            "short": Y[:-4],
            "partial": partial,
            "unlabelled": np.full_like(Y, np.nan),
            "none": None,
        }[target]
        estimator = EONRegressor(regression_recipe(), dtype=DTYPE, device=DEVICE)
        with pytest.raises(ValueError, match=message):
            estimator.fit(X, invalid)


class TestRegressionWeights:
    def test_callable_is_passed_unchanged_and_only_network_invokes_it(self, monkeypatch):
        X, Y = regression_data()
        Y[::4] = np.nan
        seen = []

        def weighting(target):
            seen.append(target.clone())
            return torch.linspace(0.5, 1.5, len(target), dtype=target.dtype, device=target.device)

        real_fit = Network.fit

        def fit(*args, **kwargs):
            assert not seen
            return real_fit(*args, **kwargs)

        monkeypatch.setattr(Network, "fit", fit)
        weights = np.linspace(0.5, 1, len(Y))
        estimator = EONRegressor(
            regression_recipe(),
            regression_weighting=weighting,
            dtype=DTYPE,
            device=DEVICE,
            random_state=2,
        ).fit(X, Y, sample_weight=weights)
        direct = real_fit(
            estimator.recipe,
            torch.tensor(X, dtype=DTYPE, device=DEVICE),
            torch.tensor(Y, dtype=DTYPE, device=DEVICE),
            task_weights=weighting,
            sample_weights=torch.tensor(weights, dtype=DTYPE, device=DEVICE),
            seed=2,
        )
        np.testing.assert_array_equal(
            estimator.predict(X),
            direct.predict(torch.tensor(X, dtype=DTYPE, device=DEVICE)).cpu().numpy(),
        )
