"""Tabular validation stays at the adapter boundary."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import Network
from entlearn.scikit_adapter import EONClassifier

from ._fixtures import classifier_recipe, tabular_data


class TestTabularStaging:
    @pytest.mark.parametrize("scalar_dtype", [np.float32, np.float64, np.int32])
    def test_non_native_byte_order_preserves_fit_and_query_values(self, scalar_dtype):
        X, y = tabular_data()
        integer = np.issubdtype(scalar_dtype, np.integer)
        native = (X > 0.5).astype(scalar_dtype) if integer else X.astype(scalar_dtype)
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

    def test_a_dropped_column_still_counts_in_the_query_width(self):
        _, y = tabular_data()
        X = np.column_stack([np.repeat("one", len(y)), y])
        estimator = EONClassifier(
            classifier_recipe(), categorical_features=[0, 1], dtype=DTYPE, device=DEVICE
        )
        with pytest.warns(UserWarning, match="single level") as seen:
            estimator.fit(X, y)
        assert [notice.filename for notice in seen] == [__file__]
        assert estimator.n_features_in_ == 2
        with pytest.raises(ValueError, match="expecting 2 features"):
            estimator.predict(X[:, 1:])

    @pytest.mark.parametrize(
        ("categorical", "error", "match"),
        [
            ([True], TypeError, r"entry True is not an integer"),
            ([0, 0], ValueError, r"duplicate categorical feature index 0"),
            ([-1], ValueError, r"index -1 is outside \[0, 2\)"),
            ([2], ValueError, r"index 2 is outside \[0, 2\)"),
            (["first"], TypeError, r"entry 'first' is not an integer"),
            ("auto", ValueError, r"got 'auto'"),
        ],
    )
    def test_invalid_categorical_selection(self, categorical, error, match):
        X, y = tabular_data()
        with pytest.raises(error, match=match):
            EONClassifier(classifier_recipe(), categorical_features=categorical).fit(X, y)

    def test_missing_categorical_values_rejected_at_fit_and_prediction(self):
        _, y = tabular_data()
        X = pd.DataFrame({"label": pd.Series(y, dtype=object)})
        estimator = EONClassifier(
            classifier_recipe(), categorical_features="from_dtype", dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        invalid = X.copy()
        invalid.iloc[0, 0] = None
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
    def test_continuous_input_fits_like_its_float64_values(self, kind):
        X, y = tabular_data()
        X = (X > 0.5).astype(np.int32) if kind == "integer" else X
        X = X[:, ::-1] if kind == "negative_stride" else X
        # ``np.longdouble`` can be wider than float64, so staging must narrow it.
        X = X.astype(np.longdouble) if kind == "extended_precision" else X
        if kind == "readonly":
            X.setflags(write=False)
        reference = np.asarray(X, dtype=np.float64)
        X = pd.DataFrame(X) if kind == "dataframe" else X

        def probabilities(data):
            estimator = EONClassifier(
                classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4
            )
            return estimator.fit(data, y).predict_proba(data)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The given NumPy array is not writable")
            staged = probabilities(X)
        np.testing.assert_array_equal(staged, probabilities(reference))

    def test_fit_and_prediction_leave_the_callers_array_unchanged(self):
        X, y = tabular_data()
        # An array already in the computation dtype is staged without a copy.
        X = X.astype(np.float32 if DTYPE is torch.float32 else np.float64)
        original = X.copy()
        estimator = EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4)
        estimator.fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(X, original)

    def test_complex_columns_are_rejected(self):
        X, y = tabular_data()
        frame = pd.DataFrame(X + 1j * X, columns=["easting", "northing"])
        with pytest.raises(ValueError, match="Complex data not supported"):
            EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE).fit(frame, y)

    @pytest.mark.parametrize("as_frame", [False, True])
    def test_non_numeric_continuous_column_names_its_position(self, as_frame):
        X, y = tabular_data()
        data = (
            pd.DataFrame({"first": X[:, 0], "second": y})
            if as_frame
            else np.column_stack([X[:, 0].astype(str), y])
        )
        with pytest.raises(ValueError, match="position 1 is not numeric"):
            EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE).fit(data, y)

    def test_out_of_range_frame_values_are_refused_without_an_overflow_warning(self):
        X, y = tabular_data()
        frame = pd.DataFrame(X, columns=["easting", "northing"])
        frame.iloc[0, 0] = 1e300
        estimator = EONClassifier(classifier_recipe(), dtype=torch.float32, device=DEVICE)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="infinity"):
                estimator.fit(frame, y)

    def test_a_default_fit_computes_in_float64_on_the_cpu(self):
        X, y = tabular_data()
        network = EONClassifier(classifier_recipe(), random_state=4).fit(X, y).network_
        assert network.schema.computation_dtype == torch.float64
        assert network.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_an_unindexed_accelerator_resolves_to_its_current_device(self):
        X, y = tabular_data()
        estimator = EONClassifier(classifier_recipe(), device="cuda", random_state=4)
        network = estimator.fit(X, y).network_
        assert network.device == torch.device("cuda", torch.cuda.current_device())

    @pytest.mark.parametrize(
        "frame",
        [
            pd.DataFrame({"easting": pd.Series([], dtype=float)}),
            pl.DataFrame({"easting": pl.Series([], dtype=pl.Float64)}),
        ],
        ids=["pandas", "polars"],
    )
    def test_empty_frames_are_rejected(self, frame):
        with pytest.raises(ValueError, match="0 sample"):
            EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE).fit(frame, [])


class TestHardLabels:
    def test_numeric_minus_one_marks_unlabelled_rows(self):
        X, labels = tabular_data()
        _, codes = np.unique(labels, return_inverse=True)
        y = np.array([10, 20, 40])[codes]
        y[::7] = -1
        estimator = EONClassifier(classifier_recipe(), dtype=DTYPE, device=DEVICE, random_state=4)
        estimator.fit(X, y)
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
            ("one_class", "only 1 class"),
            ("unlabelled", "no labelled instances"),
            ("none", "target y is None"),
            ("infinite", "y contains infinity"),
            ("missing", "y contains NaN"),
        ],
    )
    def test_incompatible_targets(self, target, message):
        X, y = tabular_data()
        codes = np.unique(y, return_inverse=True)[1].astype(float)
        invalid = {
            "one_class": np.zeros(len(y)),
            "unlabelled": np.full(len(y), -1),
            "none": None,
            "infinite": np.r_[np.inf, codes[1:]],
            "missing": np.r_[np.nan, codes[1:]],
        }[target]
        with pytest.raises(ValueError, match=message):
            EONClassifier(classifier_recipe()).fit(X, invalid)

    @pytest.mark.parametrize(
        "values,expected_dtype",
        [
            pytest.param([10, 20, 40], np.int32, id="int32"),
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
