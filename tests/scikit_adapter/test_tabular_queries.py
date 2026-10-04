"""Original column meanings across mixed tabular query representations."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import (
    ClassificationHead,
    Coupling,
    Input,
    PredictConfig,
    Recipe,
    ReconstructionResult,
    RegressionHead,
)
from entlearn.scikit_adapter import (
    EONClassifier,
    EONRegressor,
    FeatureLayout,
    TabularReconstruction,
)

from ._fixtures import tabular_data


class TestTabularRecords:
    @pytest.mark.parametrize(
        "build",
        [
            lambda: TabularReconstruction(
                continuous=np.ones((3, 1)),
                categorical=(np.ones((3, 1)),),
                continuous_indices=(1,),
                categorical_indices=(0,),
                categories=(np.array(["site"]),),
                dropped_indices=(0,),
                feature_names=None,
            ),
            lambda: FeatureLayout(
                continuous_indices=(0,),
                categorical_indices=(1,),
                dropped_indices=(),
                categories=(np.array(["a", "b"]),),
                dropped_categories=(),
            ),
        ],
        ids=["reconstruction", "layout"],
    )
    def test_records_compare_and_hash_by_identity(self, build):
        record, copy = build(), build()
        assert record == record  # noqa: PLR0124
        assert record != copy
        assert len({record, copy}) == 2

    def test_layout_maps_tensor_order_to_original_columns_and_counts_dropped(self):
        layout = FeatureLayout(
            continuous_indices=(1, 4),
            categorical_indices=(0, 3),
            dropped_indices=(2,),
            categories=(np.array(["a", "b"]), np.array([0, 1])),
            dropped_categories=(np.array(["c"]),),
        )
        assert layout.tensor_positions == (1, 4, 0, 3)
        assert layout.n_columns == 5

    @pytest.mark.parametrize(
        ("kept", "dropped", "blocks"),
        [((1, 2), (), 1), ((), (1, 2), 0)],
        ids=["kept", "dropped"],
    )
    def test_as_tabular_rejects_a_categorical_column_without_its_record(
        self, kept, dropped, blocks
    ):
        layout = FeatureLayout(
            continuous_indices=(0,),
            categorical_indices=kept,
            dropped_indices=dropped,
            categories=(np.array(["a", "b"]),) * len(kept),
            dropped_categories=(np.array(["c"]),) if dropped else (),
        )
        result = ReconstructionResult(torch.ones(3, 1), (torch.full((3, 2), 0.5),) * blocks)
        with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
            layout.as_tabular(result, None)


class TestTabularQueries:
    @pytest.mark.parametrize(
        "categorical", [(0, 1, 3), np.array([0, 1, 3], dtype=np.uint64)], ids=["tuple", "uint64"]
    )
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_active_indices_are_sorted_original_columns_and_detached(
        self, estimator_type, categorical
    ):
        X, labels = tabular_data()
        frame = pd.DataFrame(
            {
                "category": labels,
                "constant": "fixed",
                "continuous": X[:, 0],
                "other_category": np.arange(len(X)) % 2,
                "other_continuous": X[:, 1],
            }
        )
        head = (
            ClassificationHead(Coupling.M) if estimator_type is EONClassifier else RegressionHead()
        )
        model = estimator_type(
            Recipe.chain(Input(K=3, epsilon_D=0.2), head),
            categorical_features=categorical,
            max_iter=5,
            random_state=2,
            dtype=DTYPE,
            device=DEVICE,
        )
        with pytest.warns(UserWarning, match="single level"):
            model.fit(frame, labels if estimator_type is EONClassifier else X[:, 1])
        indices = model.active_features(tol=0)
        assert indices.dtype == np.int64
        np.testing.assert_array_equal(indices, [0, 2, 3, 4])
        indices.fill(-1)
        np.testing.assert_array_equal(model.active_features(tol=0), [0, 2, 3, 4])

    def test_dropped_columns_take_the_fitted_precision_not_the_numpy_default(self):
        X, _ = tabular_data()
        frame = pd.DataFrame({"constant": np.repeat("site", len(X)), "value": X[:, 0]})
        model = EONRegressor(
            Recipe.chain(Input(K=2, epsilon=0.1, epsilon_D=0.2), RegressionHead()),
            categorical_features=(0,),
            dtype=torch.float32,
            device=DEVICE,
            max_iter=2,
            random_state=1,
        )
        with pytest.warns(UserWarning, match="single level"):
            model.fit(frame, X[:, 1])

        result = model.reconstruct(frame)

        assert result.dropped_indices == (0,)
        assert result.continuous.dtype == np.float32
        assert result.categorical[0].dtype == result.continuous.dtype

    @pytest.mark.parametrize("container", ["numpy", "pandas", "polars"])
    @pytest.mark.parametrize("continuous", [False, True])
    def test_reconstruction_and_reporting_preserve_interleaved_feature_meanings(
        self, container, continuous
    ):
        X, labels = tabular_data()
        columns = {"constant": np.repeat("site", len(X)), "zone": labels}
        if continuous:
            columns = {"constant": columns["constant"], "value": X[:, 0], "zone": labels}
        frame = pd.DataFrame(columns)
        data = (
            frame
            if container == "pandas"
            else pl.DataFrame(columns)
            if container == "polars"
            else frame.to_numpy()
        )
        positions = (0, len(columns) - 1)
        estimator = EONRegressor(
            Recipe.chain(Input(K=3, epsilon=0.1, epsilon_D=0.2, epsilon_T=0.5), RegressionHead()),
            categorical_features=positions,
            dtype=DTYPE,
            device=DEVICE,
            max_iter=10,
            random_state=2,
        )
        with pytest.warns(UserWarning, match="single level"):
            estimator.fit(data, X[:, 1])
        policy = PredictConfig(predict_mode="iterative", max_iter=2)
        result = estimator.reconstruct(data, predict_config=policy)
        assert isinstance(result, TabularReconstruction)
        assert result.feature_names == (None if container == "numpy" else tuple(columns))
        assert result.categorical_indices == positions
        assert result.continuous_indices == ((1,) if continuous else ())
        np.testing.assert_array_equal(result.categories[0], ["site"])
        np.testing.assert_array_equal(result.categories[1], np.unique(labels))
        codes = torch.tensor(np.unique(labels, return_inverse=True)[1], device=DEVICE)
        tensor = torch.tensor(X[:, :1] if continuous else X[:, :0], dtype=DTYPE, device=DEVICE)
        expected = estimator.network_.reconstruct(tensor, X_cat=(codes,), predict_config=policy)
        np.testing.assert_array_equal(result.continuous, expected.continuous.cpu().numpy())
        np.testing.assert_array_equal(result.categorical[1], expected.categorical[0].cpu().numpy())
        np.testing.assert_array_equal(result.categorical[0], np.ones((len(X), 1)))
        assert result.categorical[0].dtype == result.categorical[1].dtype
        np.testing.assert_array_equal(
            estimator.score_samples(data),
            estimator.network_.score_samples(tensor, X_cat=(codes,)).cpu().numpy(),
        )
        np.testing.assert_array_equal(
            estimator.recover_instance_weights(data),
            estimator.network_.predict_with_details(
                tensor, X_cat=(codes,), details=("instance_weights",)
            )
            .instance_weights.cpu()
            .numpy(),
        )
        expected_weights = estimator.network_.inspect("feature_weights")["input"].cpu().numpy()
        actual = estimator.feature_importances_
        np.testing.assert_array_equal(actual[1:], expected_weights)
        assert actual[0] == 0
        if continuous:
            np.testing.assert_array_equal(
                estimator.active_features(tol=0.5),
                np.flatnonzero(actual > 0.5 / len(expected_weights)),
            )
        query = frame.copy()
        query["constant"] = "ignored"
        query_data = (
            query
            if container == "pandas"
            else pl.DataFrame({name: query[name].to_numpy() for name in query})
            if container == "polars"
            else query.to_numpy()
        )
        repeated = estimator.reconstruct(query_data)
        np.testing.assert_array_equal(repeated.categories[0], ["site"])
        np.testing.assert_array_equal(repeated.categorical[0], np.ones((len(X), 1)))
