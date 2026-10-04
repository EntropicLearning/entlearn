"""Regression through the public tabular estimator and tensor lifecycle."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.base import clone
from sklearn.metrics import r2_score

from entlearn import ConvergenceWarning, Network, PredictConfig
from entlearn.scikit_adapter import EONRegressor

from ._fixtures import classifier_recipe, regression_data, regression_recipe


class TestRegressor:
    def test_nested_recipe_and_original_state_support_fresh_fits(self):
        X, Y = regression_data()
        estimator = EONRegressor(regression_recipe(), dtype=DTYPE, device=DEVICE, random_state=2)
        estimator.set_params(recipe__blocks__input__K=4, recipe__blocks__output__W_M=(1, 2))
        estimator.fit(X, Y)
        state = estimator.network_.initial_state
        fresh = clone(estimator).set_params(initial_state=state)
        assert not hasattr(fresh, "network_")
        fresh.fit(X[::2], Y[::2])
        direct = Network.fit(
            fresh.recipe,
            torch.tensor(X[::2], dtype=DTYPE, device=DEVICE),
            torch.tensor(Y[::2], dtype=DTYPE, device=DEVICE),
            initial_state=state,
            seed=2,
        )
        np.testing.assert_array_equal(
            fresh.predict(X),
            direct.predict(torch.tensor(X, dtype=DTYPE, device=DEVICE)).cpu().numpy(),
        )

    @pytest.mark.parametrize("target_shape", ["vector", "column", "multi"])
    def test_fit_predict_preserves_target_shape_and_matches_tensor_lifecycle(self, target_shape):
        X, Y = regression_data()
        y = Y[:, 0] if target_shape == "vector" else Y[:, :1] if target_shape == "column" else Y
        estimator = EONRegressor(regression_recipe(), random_state=4, dtype=DTYPE, device=DEVICE)
        assert estimator.fit(X, y) is estimator
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
        assert estimator.score(X, y) == r2_score(y, prediction)
        fresh = clone(estimator)
        assert not hasattr(fresh, "network_")
        np.testing.assert_array_equal(fresh.fit(X, y).predict(X), prediction)

    def test_required_recipe_checks(self):
        X, y = regression_data()
        with pytest.raises(TypeError, match="recipe"):
            EONRegressor()  # ty: ignore[missing-argument]
        for recipe, message in (
            (None, "must be a valid Recipe"),
            ("default", "must be a valid Recipe"),
            (classifier_recipe(), "EONRegressor requires a regression Recipe"),
        ):
            with pytest.raises(ValueError, match=message):
                EONRegressor(recipe).fit(X, y)

    def test_prediction_and_complete_policy_delegate_without_constructor_leakage(self, monkeypatch):
        X, y = regression_data()
        estimator = EONRegressor(
            regression_recipe(), random_state=3, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        estimator.set_params(recipe=None, dtype="invalid", device="invalid")
        override = PredictConfig(tol=0.03, max_iter=17)
        calls = []

        def predict(network, X_cont, *, X_cat, predict_config):
            assert network is estimator.network_
            assert X_cont.dtype == DTYPE and X_cont.device.type == "meta"
            assert X_cat is None
            calls.append(predict_config)
            return torch.full((len(X), 2), 0.37, dtype=DTYPE, device=DEVICE)

        monkeypatch.setattr(Network, "predict", predict)
        # Queries follow the fitted Network's device, which meta separates from the lane's.
        monkeypatch.setattr(Network, "device", property(lambda network: torch.device("meta")))
        numpy_dtype = np.float32 if DTYPE is torch.float32 else np.float64
        expected = np.full((len(X), 2), 0.37, dtype=numpy_dtype)
        for config in (None, override):
            result = estimator.predict(X, predict_config=config)
            np.testing.assert_array_equal(result, expected, strict=True)
        assert calls == [None, override]

    def test_constructor_device_is_resolved_before_staging(self):
        X, Y = regression_data()
        with pytest.raises(RuntimeError, match="device type"):
            EONRegressor(regression_recipe(), device="bogus", dtype=DTYPE).fit(X, Y)

    def test_dispatch_controls_reach_candidate_selection_verbatim(self):
        X, Y = regression_data()
        controls = dict(dtype=DTYPE, device=DEVICE, random_state=7, max_iter=3, n_inits=2)
        parallel = EONRegressor(regression_recipe(), n_jobs=2, **controls).fit(X, Y)
        assert parallel.get_params()["n_jobs"] == 2
        assert {
            outcome.effective_backend
            for outcome in parallel.network_.diagnostics.initialisation_outcomes
        } == {"threads"}
        # A single worker needs no pool, but still records the requested policy.
        serial = EONRegressor(regression_recipe(), parallel_backend="processes", **controls).fit(
            X, Y
        )
        assert {
            (outcome.requested_backend, outcome.effective_backend)
            for outcome in serial.network_.diagnostics.initialisation_outcomes
        } == {("processes", "serial")}

    def test_supplied_state_still_resolves_selection_scoring(self):
        X, Y = regression_data()
        recipe = regression_recipe()
        state = Network.initialise(
            recipe,
            torch.as_tensor(X, dtype=DTYPE, device=DEVICE),
            torch.as_tensor(Y, dtype=DTYPE, device=DEVICE),
            computation_dtype=DTYPE,
        )
        estimator = EONRegressor(
            recipe,
            initial_state=state,
            scoring="not-a-registered-scorer",
            dtype=DTYPE,
            device=DEVICE,
            max_iter=3,
        )
        with pytest.raises(ValueError, match="not-a-registered-scorer"):
            estimator.fit(X, Y)
        assert not hasattr(estimator, "network_")

    @pytest.mark.parametrize("removed", ["output_mode", "epsilon_last", "calibration"])
    def test_removed_controls_are_not_accepted(self, removed):
        with pytest.raises(TypeError):
            EONRegressor(regression_recipe(), **{removed: None})

    @pytest.mark.parametrize(
        "config", [PredictConfig(output_mode="geometric"), PredictConfig(epsilon_P=0.5)]
    )
    def test_classification_policy_is_rejected_at_fit_and_prediction(self, config):
        X, y = regression_data()
        with pytest.raises(ValueError, match="regression"):
            EONRegressor(regression_recipe(), predict_config=config).fit(X, y)
        estimator = EONRegressor(
            regression_recipe(), random_state=3, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        with pytest.raises(ValueError, match="regression"):
            estimator.predict(X, predict_config=config)


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
        with pytest.raises(ValueError, match="unseen"):
            estimator.predict(frame.assign(zone="new"))
        with pytest.raises(ValueError, match="feature names"):
            estimator.predict(frame.iloc[:, ::-1])

    def test_fully_missing_rows_are_unlabelled_but_partial_rows_are_rejected(self, monkeypatch):
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

        def unexpected_fit(*args, **kwargs):
            pytest.fail("invalid tabular targets must fail before Network.fit")

        monkeypatch.setattr(Network, "fit", unexpected_fit)
        partial = Y.copy()
        partial[1, 0] = np.nan
        for target, message in (
            (Y[:-4], "inconsistent numbers of samples"),
            (Y[:, :, None], "dim 3"),
            (partial, "partially-NaN"),
            (np.full_like(Y, np.nan), "no labelled"),
            (np.full_like(Y, np.inf), "infinity"),
            (None, "requires y"),
        ):
            with pytest.raises(ValueError, match=message):
                estimator.fit(X, target)


class TestRegressionWeightsAndPublication:
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
            assert kwargs["task_weights"] is weighting
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
        assert len(seen) == 1
        torch.testing.assert_close(
            seen[0], torch.tensor(Y[~np.isnan(Y).any(axis=1)], dtype=DTYPE, device=DEVICE)
        )
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

    def test_failed_refit_preserves_prediction_shape_and_fitted_names(self, monkeypatch):
        X, Y = regression_data()
        frame = pd.DataFrame(X, columns=["a", "b"])
        estimator = EONRegressor(
            regression_recipe(), dtype=DTYPE, device=DEVICE, random_state=2
        ).fit(frame, Y[:, 0])
        before = estimator.predict(frame)
        previous = estimator.network_

        def fail(*args, **kwargs):
            raise RuntimeError("fit failed")

        with monkeypatch.context() as patch:
            patch.setattr(Network, "fit", fail)
            with pytest.raises(RuntimeError, match="fit failed"):
                estimator.fit(X, Y)
        assert estimator.network_ is previous
        assert estimator.n_outputs_ == 1
        np.testing.assert_array_equal(estimator.predict(frame), before)
        np.testing.assert_array_equal(estimator.feature_names_in_, frame.columns)
        estimator.fit(X, Y)
        assert not hasattr(estimator, "feature_names_in_")
        assert estimator.predict(X).shape == Y.shape

    def test_warning_as_error_preserves_previous_fit(self):
        X, Y = regression_data()
        estimator = EONRegressor(
            regression_recipe(), dtype=DTYPE, device=DEVICE, random_state=2
        ).fit(X, Y[:, 0])
        before = estimator.predict(X)
        previous = estimator.network_
        estimator.set_params(max_iter=1, tol=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(ConvergenceWarning):
                estimator.fit(X, Y)
        assert estimator.network_ is previous
        np.testing.assert_array_equal(estimator.predict(X), before)
