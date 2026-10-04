"""Tabular validation stays at the adapter boundary."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from conftest import DEVICE, DTYPE
from sklearn.exceptions import DataConversionWarning

from entlearn import Network
from entlearn.scikit_adapter import EONClassifier
from entlearn.scikit_adapter.classifier import _prepare_class_weights
from entlearn.scikit_adapter.features import (
    _apply_feature_layout,
    _check_tabular_input,
    _continuous_block,
    _feature_tensors,
    _fit_feature_layout,
    _resolve_device_dtype,
)

from ._fixtures import classifier_recipe, tabular_data


class TestTabularStaging:
    @pytest.mark.parametrize("scalar_dtype", [np.float32, np.float64, np.int32])
    def test_non_native_byte_order_preserves_fit_and_query_values(self, scalar_dtype):
        X, y = tabular_data()
        native = (X * 20).astype(scalar_dtype)
        swapped = native.astype(native.dtype.newbyteorder("S"))
        estimator = EONClassifier(
            classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4
        ).fit(native, y)
        expected = estimator.predict_proba(native)
        np.testing.assert_array_equal(estimator.predict_proba(swapped), expected)
        estimator.fit(swapped, y)
        np.testing.assert_array_equal(estimator.predict_proba(native), expected)

    @pytest.mark.parametrize("fit_as_list", [True, False])
    @pytest.mark.parametrize("label_column", ["string", "numeric"])
    def test_mixed_lists_and_arrays_share_fit_and_query_staging(self, fit_as_list, label_column):
        X, y = tabular_data()
        # A numeric label column keeps the table numeric, so the continuous block is
        # selected out of the whole array rather than assembled column by column.
        labels = y if label_column == "string" else np.unique(y, return_inverse=True)[1]
        table = np.column_stack([X[:, 0], labels, X[:, 1]])
        estimator = EONClassifier(
            classifier_recipe(),
            categorical_features=[1],
            dtype=DTYPE,
            device=DEVICE,
            random_state=7,
        ).fit(table.tolist() if fit_as_list else table, y)
        assert estimator.network_.schema.D_cont == 2
        np.testing.assert_array_equal(
            estimator.predict_proba(table.tolist()), estimator.predict_proba(table)
        )
        np.testing.assert_array_equal(estimator.predict(table.tolist()), y)

    def test_dataframe_names_modality_order_and_predictions_match_tensor_input(self):
        X, y = tabular_data()
        frame = pd.DataFrame(
            {
                "category": pd.Categorical(y),
                "northing": X[:, 1],
                "string_category": pd.Series(y, dtype="string"),
                "easting": X[:, 0],
            }
        )
        estimator = EONClassifier(
            classifier_recipe(),
            categorical_features="from_dtype",
            random_state=3,
            dtype=DTYPE,
            device=DEVICE,
        ).fit(frame, y)
        classes, codes = np.unique(y, return_inverse=True)
        continuous = torch.as_tensor(X[:, ::-1].copy(), dtype=DTYPE, device=DEVICE)
        categorical = torch.as_tensor(codes, dtype=torch.int64, device=DEVICE)
        direct = Network.fit(
            estimator.recipe,
            continuous,
            categorical,
            X_cat=[categorical, categorical],
            seed=3,
        )
        np.testing.assert_array_equal(
            estimator.predict_proba(frame),
            direct.predict(continuous, X_cat=[categorical, categorical]).cpu().numpy(),
        )
        np.testing.assert_array_equal(estimator.feature_names_in_, frame.columns)
        np.testing.assert_array_equal(estimator.classes_, classes)
        assert estimator.feature_layout_.categorical_indices == (0, 2)
        assert estimator.network_.schema.D_cont == 2
        for query in (frame.rename(columns={"easting": "other"}), frame.iloc[:, ::-1]):
            with pytest.raises(ValueError, match="feature names"):
                estimator.predict(query)
        with pytest.warns(UserWarning, match="valid feature names"):
            estimator.predict(frame.to_numpy())

    def test_categorical_only_and_dropped_columns_keep_original_query_width(self):
        _, y = tabular_data()
        frame = pd.DataFrame({"constant": ["one"] * len(y), "label": y})
        estimator = EONClassifier(
            classifier_recipe(), categorical_features="from_dtype", dtype=DTYPE, device=DEVICE
        )
        with pytest.warns(UserWarning, match="single level") as seen:
            estimator.fit(frame, y)
        assert [notice.filename for notice in seen] == [__file__]
        assert estimator.network_.schema.D_cont == 0
        assert estimator.score(frame.assign(constant="ignored"), y) == 1
        with pytest.raises(ValueError, match="feature"):
            estimator.predict(frame[["label"]])

    @pytest.mark.parametrize(
        ("categorical", "error", "match"),
        [
            ([True], TypeError, r"entry True is not an integer column index"),
            ([0, 0], ValueError, r"duplicate categorical feature index 0"),
            ([-1], ValueError, r"index -1 is outside \[0, 2\)"),
            ([2], ValueError, r"index 2 is outside \[0, 2\)"),
            (["first"], TypeError, r"entry 'first' is not an integer column index"),
            ("auto", ValueError, r"string must be 'from_dtype', got 'auto'"),
        ],
    )
    def test_invalid_categorical_selection(self, categorical, error, match):
        X, y = tabular_data()
        with pytest.raises(error, match=match):
            EONClassifier(classifier_recipe(), categorical_features=categorical).fit(X, y)

    @pytest.mark.parametrize("column_dtype", [object, "string"])
    @pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
    def test_missing_categorical_values_rejected_at_fit_and_prediction(self, column_dtype, missing):
        _, y = tabular_data()
        X = pd.DataFrame({"label": pd.Series(y, dtype=column_dtype)})
        estimator = EONClassifier(
            classifier_recipe(), categorical_features="from_dtype", dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        invalid = X.copy()
        invalid.iloc[0, 0] = missing
        with pytest.raises(ValueError, match="missing"):
            estimator.predict(invalid)
        with pytest.raises(ValueError, match="missing"):
            estimator.fit(invalid, y)

    def test_unseen_categorical_levels_are_named_in_the_rejection(self):
        _, y = tabular_data()
        X = pd.DataFrame({"label": y})
        estimator = EONClassifier(
            classifier_recipe(), categorical_features="from_dtype", dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        with pytest.raises(ValueError, match=r"unseen at fit time: \['south'\]"):
            estimator.predict(X.replace({"east": "south"}))

    @pytest.mark.parametrize(
        ("column_dtype", "marker"),
        [
            ("datetime64[D]", "NaT"),
            ("timedelta64[D]", "NaT"),
            ("float32", np.nan),
            ("float16", np.nan),
        ],
    )
    def test_categorical_missing_marker_rejected_without_replacing_a_fit(
        self, column_dtype, marker
    ):
        _, y = tabular_data()
        X = np.unique(y, return_inverse=True)[1].astype(column_dtype)[:, None]
        estimator = EONClassifier(
            classifier_recipe(),
            categorical_features=[0],
            dtype=DTYPE,
            device=DEVICE,
            random_state=4,
        ).fit(X, y)
        previous = estimator.network_
        expected = estimator.predict_proba(X)
        invalid = X.copy()
        invalid[0, 0] = marker
        with pytest.raises(ValueError, match="missing"):
            estimator.fit(invalid, y)
        with pytest.raises(ValueError, match="missing"):
            estimator.predict(invalid)
        assert estimator.network_ is previous
        np.testing.assert_array_equal(estimator.predict_proba(X), expected)

    def test_ambiguous_object_dtype_and_no_usable_features_are_rejected(self):
        X, y = tabular_data()
        with pytest.raises(ValueError, match="cannot discriminate"):
            EONClassifier(classifier_recipe(), categorical_features="from_dtype").fit(
                X.astype(object), y
            )
        with (
            pytest.warns(UserWarning, match="single level"),
            pytest.raises(ValueError, match="usable"),
        ):
            EONClassifier(classifier_recipe(), categorical_features=[0]).fit(
                np.zeros((len(y), 1)), y
            )

    @pytest.mark.parametrize(
        "kind", ["integer", "negative_stride", "readonly", "extended_precision", "dataframe"]
    )
    def test_continuous_input_converts_to_computation_precision(self, kind):
        X, y = tabular_data()
        X = (X * 20).astype(np.int32) if kind == "integer" else X
        X = X[:, ::-1] if kind == "negative_stride" else X
        # ``np.longdouble`` can be wider than float64, so staging must narrow it.
        X = X.astype(np.longdouble) if kind == "extended_precision" else X
        if kind == "readonly":
            X.setflags(write=False)
        X = pd.DataFrame(X) if kind == "dataframe" else X
        estimator = EONClassifier(
            classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4
        ).fit(X, y)
        assert estimator.network_.schema.computation_dtype == DTYPE
        assert estimator.predict_proba(X).dtype == (
            np.float32 if torch.float32 == DTYPE else np.float64
        )


class TestStagingPrimitives:
    """Staging contracts the estimator surface restates and so cannot discriminate."""

    @pytest.mark.parametrize("as_frame", [False, True])
    def test_complex_columns_are_rejected(self, as_frame):
        values = np.array([[0.1 + 2j, 0.3 + 4j], [0.5 + 6j, 0.7 + 8j]])
        X = pd.DataFrame(values, columns=["easting", "northing"]) if as_frame else values
        with pytest.raises(ValueError, match="Complex"):
            _continuous_block(X, (0, 1), np.float64)

    @pytest.mark.parametrize("as_frame", [False, True])
    def test_non_numeric_continuous_column_names_its_position(self, as_frame):
        values = np.array([["east", "north"], ["west", "east"]])
        X = pd.DataFrame(values, columns=["first", "second"]) if as_frame else values
        with pytest.raises(ValueError, match="position 1 is not numeric"):
            _continuous_block(X, (1,), np.float64)

    @pytest.mark.parametrize("staging_dtype", [np.float32, np.float64])
    def test_staged_blocks_honour_the_requested_dtype_and_c_order(self, staging_dtype):
        X, _ = tabular_data()
        frame = pd.DataFrame(X, columns=["easting", "northing"])
        # Byte-swapped, Fortran-ordered input must still stage as native, C-ordered
        # values of the requested dtype.
        fortran = np.asfortranarray(X).astype(X.dtype.newbyteorder("S"))
        for source in (X, frame):
            empty = _continuous_block(source, (), staging_dtype)
            assert empty.shape == (len(X), 0)
            assert empty.dtype == staging_dtype
        assert _continuous_block(frame, (0, 1), staging_dtype).dtype == staging_dtype
        staged = _continuous_block(fortran, (0, 1), staging_dtype)
        assert staged.dtype == staging_dtype
        assert staged.flags.c_contiguous

    @pytest.mark.parametrize("as_frame", [False, True])
    def test_out_of_range_values_saturate_without_an_overflow_warning(self, as_frame):
        values = np.array([[1e300, 2e300]])
        values.setflags(write=False)
        X = pd.DataFrame(values, columns=["easting", "northing"]) if as_frame else values
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            staged = _continuous_block(X, (0, 1), np.float32)
        assert np.isinf(staged).all()

    def test_categorical_codes_stage_as_int64(self):
        _, y = tabular_data()
        frame = pd.DataFrame({"label": y})
        layout, _, codes = _fit_feature_layout(frame, [0], np.float64)
        assert codes[0].dtype == np.int64
        assert _apply_feature_layout(frame, layout, np.float64)[1][0].dtype == np.int64

    def test_features_and_class_weights_stage_on_the_requested_device(self):
        X, y = tabular_data()
        classes, codes = np.unique(y, return_inverse=True)
        # The meta device needs no accelerator, so it discriminates on the cpu lane.
        meta = torch.device("meta")
        continuous, categorical = _feature_tensors(X, [codes], device=meta, dtype=torch.float64)
        weights = _prepare_class_weights([1, 2, 3], classes, dtype=torch.float64, device=meta)
        assert continuous.device.type == "meta"
        assert categorical[0].device.type == "meta"
        assert weights.device.type == "meta"

    @pytest.mark.parametrize(
        ("device", "dtype", "expected"),
        [
            (None, None, (torch.device("cpu"), torch.float64)),
            ("meta", torch.float32, (torch.device("meta"), torch.float32)),
        ],
    )
    def test_requested_device_and_dtype_survive_resolution(self, device, dtype, expected):
        assert _resolve_device_dtype(device, dtype) == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_an_unindexed_accelerator_resolves_to_its_current_device(self):
        current = torch.device("cuda", torch.cuda.current_device())
        assert _resolve_device_dtype("cuda", None) == (current, torch.float64)

    @pytest.mark.parametrize(
        ("frame", "message"),
        [
            (pd.DataFrame({"easting": pd.Series([], dtype=float)}), "0 sample"),
            (pd.DataFrame(index=[0, 1]), "at least one array or dtype is required"),
            (pl.DataFrame({"easting": pl.Series([], dtype=pl.Float64)}), "0 sample"),
        ],
        ids=["pandas_no_rows", "pandas_no_columns", "polars_no_rows"],
    )
    def test_empty_frames_are_rejected(self, frame, message):
        with pytest.raises(ValueError, match=message):
            _check_tabular_input(frame)


class TestHardLabels:
    def test_numeric_unlabelled_rows_and_column_vector_warning(self):
        X, labels = tabular_data()
        _, codes = np.unique(labels, return_inverse=True)
        y = np.array([10, 20, 40])[codes]
        y[::7] = -1
        estimator = EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4)
        with pytest.warns(DataConversionWarning, match="column-vector"):
            estimator.fit(X, y[:, None])
        np.testing.assert_array_equal(estimator.classes_, [10, 20, 40])
        encoded = np.where(y == -1, -1, np.searchsorted(estimator.classes_, y))
        continuous = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        direct = Network.fit(
            estimator.recipe,
            continuous,
            torch.as_tensor(encoded, dtype=torch.int64, device=DEVICE),
            seed=4,
        )
        np.testing.assert_array_equal(
            estimator.predict_proba(X), direct.predict(continuous).cpu().numpy()
        )

    @pytest.mark.parametrize(
        "target,message",
        [
            ("distribution", "1d array"),
            ("continuous", "Unknown label type"),
            ("one_class", "only 1 class"),
            ("unlabelled", "no labelled instances"),
            ("none", "target y is None"),
        ],
    )
    def test_incompatible_targets(self, target, message):
        X, y = tabular_data()
        invalid = {
            "distribution": np.ones((len(y), 3)) / 3,
            "continuous": np.linspace(0, 1, len(y)),
            "one_class": np.zeros(len(y)),
            "unlabelled": np.full(len(y), -1),
            "none": None,
        }[target]
        with pytest.raises(ValueError, match=message):
            EONClassifier(classifier_recipe()).fit(X, invalid)

    @pytest.mark.parametrize(
        "values,expected_dtype",
        [
            pytest.param([10, 20, 40], np.int32, id="int32"),
            pytest.param([0, 5, 2**63 - 1], np.int64, id="object-integers"),
            pytest.param(["east", "north", "west"], object, id="object-strings"),
        ],
    )
    def test_label_vocabulary_keeps_the_supplied_value_precision(self, values, expected_dtype):
        X, labels = tabular_data()
        codes = np.unique(labels, return_inverse=True)[1]
        y = (
            np.asarray(values, dtype=np.int32)[codes]
            if expected_dtype is np.int32
            else np.asarray([values[code] for code in codes], dtype=object)
        )
        estimator = EONClassifier(
            classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4
        ).fit(X, y)
        assert estimator.classes_.dtype == expected_dtype
        np.testing.assert_array_equal(estimator.classes_, sorted(values))
        assert set(estimator.predict(X)) <= set(values)

    def test_integer_labels_outside_both_integer_ranges_are_rejected(self):
        X, labels = tabular_data()
        codes = np.unique(labels, return_inverse=True)[1]
        y = np.asarray([[-(2**70), 0, 1][code] for code in codes], dtype=object)
        with pytest.raises(ValueError, match="must fit in int64 or uint64"):
            EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE).fit(X, y)
