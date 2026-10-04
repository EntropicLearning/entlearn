"""Native eager dataframes preserve feature meanings across fitting and prediction."""

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from ._fixtures import dataframe_classifier, mixed_frames, tabular_data


class TestEagerDataframes:
    @pytest.mark.parametrize(
        ("nullable_dtype", "large_label"),
        [
            ("Int64", 2**53),
            ("UInt64", 2**53),
            (object, 2**53),
            ("UInt64", 2**63),
            (object, 2**63),
        ],
    )
    @pytest.mark.parametrize("weighted", [False, True])
    def test_nullable_integer_labels_preserve_exact_vocabulary(
        self, nullable_dtype, large_label, weighted
    ):
        X, _ = tabular_data()
        classes = np.array(
            [0, large_label, large_label + 1],
            dtype=np.uint64 if large_label > np.iinfo(np.int64).max else np.int64,
        )
        y = np.repeat(classes, 16)
        weights = {0: 1.0, large_label: 2.0, large_label + 1: 3.0} if weighted else None
        reference = dataframe_classifier().set_params(class_weights=weights).fit(X, y)
        estimator = dataframe_classifier().set_params(class_weights=weights)
        estimator.fit(X, pd.Series(y, dtype=nullable_dtype))
        np.testing.assert_array_equal(estimator.classes_, classes)
        probabilities = estimator.predict_proba(X)
        assert probabilities.shape == (len(y), 3)
        np.testing.assert_array_equal(probabilities, reference.predict_proba(X))
        np.testing.assert_array_equal(estimator.predict(X), y)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_mixed_schema_and_predictions_agree_across_backends(self, backend):
        pandas, polars, y = mixed_frames()
        estimator = dataframe_classifier().fit(pandas if backend == "pandas" else polars, y)
        np.testing.assert_array_equal(estimator.feature_names_in_, pandas.columns)
        assert estimator.feature_layout_.categorical_indices == (0,)
        np.testing.assert_array_equal(estimator.feature_layout_.categories[0], np.unique(y))
        np.testing.assert_array_equal(
            estimator.predict_proba(pandas), estimator.predict_proba(polars)
        )
        np.testing.assert_array_equal(estimator.predict(polars), y)

    @pytest.mark.parametrize("wrapped", [False, True])
    def test_lazy_queries_require_explicit_collection_at_fit_and_prediction(self, wrapped):
        X, y = tabular_data()
        frame = pl.DataFrame({"easting": X[:, 0], "northing": X[:, 1]})
        estimator = dataframe_classifier().fit(frame, y)
        before = estimator.predict_proba(frame)
        lazy = nw.from_native(frame.lazy()) if wrapped else frame.lazy()
        with pytest.raises(TypeError, match=r"collect\(\)"):
            estimator.fit(lazy, y)
        with pytest.raises(TypeError, match=r"collect\(\)"):
            estimator.predict(lazy)
        np.testing.assert_array_equal(estimator.predict_proba(frame), before)

    @pytest.mark.parametrize("kind", ["categorical", "enum", "string"])
    def test_logical_categories_use_values_not_backend_codes(self, kind):
        pandas, polars, y = mixed_frames()
        dtype = {
            "categorical": pl.Categorical,
            "enum": pl.Enum(["west", "east", "north"]),
            "string": pl.String,
        }[kind]
        polars = polars.with_columns(pl.col("category").cast(dtype))
        pandas["category"] = pd.Categorical(y, categories=["north", "west", "east"])
        estimator = dataframe_classifier().fit(polars, y)
        np.testing.assert_array_equal(estimator.feature_layout_.categories[0], np.unique(y))
        np.testing.assert_array_equal(
            estimator.predict_proba(polars), estimator.predict_proba(pandas)
        )
        # The query subset omits a fitted level but keeps the original vocabulary.
        subset = polars.filter(pl.col("category") != "east")
        np.testing.assert_array_equal(
            estimator.predict_proba(subset),
            estimator.predict_proba(polars)[y != "east"],
        )

    @pytest.mark.parametrize(
        "invalid", ["name", "order", "missing", "extra", "unknown", "null", "nan"]
    )
    def test_invalid_queries_and_refits_preserve_previous_fit(self, invalid):
        _, frame, y = mixed_frames()
        estimator = dataframe_classifier().fit(frame, y)
        before = estimator.predict_proba(frame)
        network = estimator.network_
        queries = {
            "name": frame.rename({"easting": "other"}),
            "order": frame.select(frame.columns[::-1]),
            "missing": frame.drop("easting"),
            "extra": frame.with_columns(pl.lit(0.0).alias("extra")),
            "unknown": frame.with_columns(pl.lit("unseen").alias("category")),
            "null": frame.with_columns(pl.lit(None, dtype=pl.String).alias("category")),
            "nan": frame.with_columns(pl.lit(float("nan")).alias("easting")),
        }
        with pytest.raises(ValueError):
            estimator.predict(queries[invalid])
        if invalid in ("null", "nan"):
            with pytest.raises(ValueError):
                estimator.fit(queries[invalid], y)
            assert estimator.network_ is network
        np.testing.assert_array_equal(estimator.predict_proba(frame), before)

    def test_categorical_only_and_dropped_columns(self):
        _, _, y = mixed_frames()
        frame = pl.DataFrame({"constant": ["one"] * len(y), "category": y})
        estimator = dataframe_classifier()
        with pytest.warns(UserWarning, match="single level"):
            estimator.fit(frame, y)
        assert estimator.network_.schema.D_cont == 0
        np.testing.assert_array_equal(
            estimator.predict(frame.with_columns(pl.lit("ignored").alias("constant"))), y
        )
