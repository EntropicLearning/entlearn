"""Public behaviour of optional statistical feature-weight helpers."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import ClassificationHead, Coupling, Input, Network, Recipe
from entlearn.helpers import feature_weights
from helpers._fixtures import (
    assert_feature_weights,
    classification_tensors,
    informative_cases,
    mixed_classification_data,
    two_target_regression_data,
)

# Each statistic, with the seed mutual information needs for reproducible scores.
METHODS = [
    pytest.param("correlation", {}, id="correlation"),
    pytest.param("mutual_info", {"random_state": 0}, id="mutual-information"),
]


class TestNetworkFeatureWeights:
    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    def test_statistical_weights_pass_through_network_initialisation(
        self,
        method: str,
        extra_kwargs: dict[str, int],
    ) -> None:
        X_cont_array, categorical_codes, target_array = mixed_classification_data()

        weights = feature_weights(
            X_cont_array,
            categorical_codes,
            target_array,
            task="classification",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )

        assert_feature_weights(
            weights,
            width=X_cont_array.shape[1] + len(categorical_codes),
            dtype=DTYPE,
            device=DEVICE,
        )
        assert weights.argmax().item() == 0
        X_cont, X_cat, target = classification_tensors(
            X_cont_array,
            categorical_codes,
            target_array,
            dtype=DTYPE,
            device=DEVICE,
        )
        recipe = Recipe.chain(Input(K=2), ClassificationHead(coupling=Coupling.M))

        state = Network.initialise(
            recipe,
            X_cont,
            target,
            X_cat=X_cat,
            feature_weights=weights,
        )

        assert state.input_geometry.feature_weights is not None
        torch.testing.assert_close(state.input_geometry.feature_weights, weights)

    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    def test_statistical_weights_remain_frozen_through_network_fit(
        self,
        method: str,
        extra_kwargs: dict[str, int],
    ) -> None:
        X_cont_array, categorical_codes, target_array = mixed_classification_data()
        weights = feature_weights(
            X_cont_array,
            categorical_codes,
            target_array,
            task="classification",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )
        X_cont, X_cat, target = classification_tensors(
            X_cont_array,
            categorical_codes,
            target_array,
            dtype=DTYPE,
            device=DEVICE,
        )
        recipe = Recipe.chain(
            Input(K=2, epsilon_D=float("inf")),
            ClassificationHead(coupling=Coupling.M),
        )
        state = Network.initialise(
            recipe,
            X_cont,
            target,
            X_cat=X_cat,
            feature_weights=weights,
        )

        network = Network.fit(
            recipe,
            X_cont,
            target,
            X_cat=X_cat,
            initial_state=state,
            max_iter=2,
        )

        torch.testing.assert_close(network.inspect("feature_weights")["input"], weights)


class TestCorrelationFeatureWeights:
    @pytest.mark.parametrize("case", informative_cases())
    def test_correlation_ranks_the_informative_feature_first(self, case: str) -> None:
        X_cont, categorical_codes, target, task, expected = informative_cases()[case]

        weights = feature_weights(
            X_cont,
            categorical_codes,
            target,
            task=task,
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
        )

        assert weights.argmax().item() == expected

    def test_correlation_scores_match_their_analytic_values(self) -> None:
        """Pin each modality's score against its closed form on a designed case.

        Two classes split eight rows. The continuous features are, in order,
        perfectly separated (correlation ratio 1), half separated (ratio
        ``sqrt(0.5)``) and constant (ratio 0). The categorical features are a
        code whose contingency table is ``[[3, 1], [1, 3]]`` (Cramér's V 0.5
        without bias correction) and a constant code (degenerate table, 0).
        """
        X_cont = np.column_stack(
            (
                np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
                np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]),
                np.full(8, 0.5),
            )
        )
        categorical_codes = (np.array([0, 0, 0, 1, 0, 1, 1, 1]), np.zeros(8, dtype=np.int64))
        target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([1.0, math.sqrt(0.5), 0.0, 0.5, 0.0])

        with np.errstate(invalid="raise", divide="raise"):
            weights = feature_weights(
                X_cont,
                categorical_codes,
                target,
                task="classification",
                method="correlation",
                dtype=DTYPE,
                device=DEVICE,
            )

        torch.testing.assert_close(
            weights,
            torch.tensor(scores / scores.sum(), dtype=DTYPE, device=DEVICE),
        )

    def test_correlation_ratio_that_rounds_above_one_is_clamped(self) -> None:
        # Between-group over total variance rounds above one for these values.
        classes = np.array([0, 0, 0, 1, 1, 1])
        weights = feature_weights(
            np.array([[0.7], [0.7], [0.7], [1.3], [1.3], [1.3]]),
            (classes,),
            classes,
            task="classification",
            method="correlation",
            dtype=torch.float64,
            device=DEVICE,
        )

        assert torch.equal(weights, torch.tensor([0.5, 0.5], dtype=torch.float64, device=DEVICE))

    def test_correlation_scores_two_labelled_rows_from_the_available_pair(self) -> None:
        """Two rows still carry a Pearson correlation; degenerate columns score zero.

        The first feature is constant and the second output is constant, so
        both must be answered from the standard-deviation guard rather than by
        dividing by zero inside ``numpy.corrcoef``.
        """
        X_cont = np.array([[0.5, 0.0], [0.5, 1.0]])
        target = np.array([[0.0, 2.0], [1.0, 2.0]])

        with np.errstate(invalid="raise", divide="raise"):
            weights = feature_weights(
                X_cont,
                (),
                target,
                task="regression",
                method="correlation",
                dtype=DTYPE,
                device=DEVICE,
            )

        torch.testing.assert_close(weights, torch.tensor([0.0, 1.0], dtype=DTYPE, device=DEVICE))

    def test_correlation_drops_a_feature_whose_correlation_overflows(self) -> None:
        """A non-finite correlation scores zero instead of poisoning every weight.

        The first feature and the first output are scaled so that their
        variances overflow float64 and ``numpy.corrcoef`` returns NaN; only the
        second feature then carries mass.
        """
        overflowing = 1e200 * np.array([1.0, -1.0, 3.0, 2.0])
        ordinary = np.array([0.1, 0.4, 0.6, 0.9])

        with np.errstate(over="ignore", invalid="ignore"):
            weights = feature_weights(
                np.column_stack((overflowing, ordinary)),
                (),
                np.column_stack((2.0 * overflowing, 2.0 * ordinary)),
                task="regression",
                method="correlation",
                dtype=DTYPE,
                device=DEVICE,
            )

        torch.testing.assert_close(weights, torch.tensor([0.0, 1.0], dtype=DTYPE, device=DEVICE))

    def test_correlation_ratio_drops_a_categorical_feature_whose_target_variance_overflows(
        self,
    ) -> None:
        # Both scores are non-finite, so neither feature may take all the weight.
        with np.errstate(over="ignore", invalid="ignore"):
            weights = feature_weights(
                np.array([[0.1], [0.4], [0.6], [0.9]]),
                (np.array([0, 1, 0, 1]),),
                1e200 * np.array([1.0, -1.0, 3.0, 2.0]),
                task="regression",
                method="correlation",
                dtype=DTYPE,
                device=DEVICE,
            )

        torch.testing.assert_close(weights, torch.tensor([0.5, 0.5], dtype=DTYPE, device=DEVICE))

    def test_correlation_output_weights_change_the_multi_output_ranking(
        self,
    ) -> None:
        X_cont, target = two_target_regression_data()

        first = feature_weights(
            X_cont,
            (),
            target,
            task="regression",
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
            output_weights=np.array([10.0, 1.0]),
        )
        second = feature_weights(
            X_cont,
            (),
            target,
            task="regression",
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
            output_weights=np.array([1.0, 10.0]),
        )

        assert first[0] > first[1]
        assert second[1] > second[0]

    @pytest.mark.parametrize(
        "mask_dtype",
        [pytest.param(bool, id="boolean-mask"), pytest.param(np.int64, id="integer-mask")],
    )
    def test_correlation_uses_only_labelled_rows(self, mask_dtype: type) -> None:
        X_cont, categorical_codes, target = mixed_classification_data()
        labelled = np.array([True, True, False, True, True, False])
        corrupted = X_cont.copy()
        corrupted[~labelled, 0] = np.array([100.0, -100.0])

        masked = feature_weights(
            corrupted,
            categorical_codes,
            target,
            task="classification",
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
            labelled_mask=labelled.astype(mask_dtype),
        )
        subset = feature_weights(
            X_cont[labelled],
            tuple(code[labelled] for code in categorical_codes),
            target[labelled],
            task="classification",
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
        )

        assert masked.argmax().item() == 0
        torch.testing.assert_close(masked, subset)

    @pytest.mark.parametrize("width", [1, 3])
    def test_correlation_falls_back_to_uniform_for_degenerate_features(self, width: int) -> None:
        weights = feature_weights(
            np.ones((20, width)),
            (),
            np.arange(20) % 2,
            task="classification",
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
        )

        torch.testing.assert_close(
            weights,
            torch.full((width,), 1.0 / width, dtype=DTYPE, device=DEVICE),
        )

    @pytest.mark.parametrize("method", ["correlation", "mutual_info"])
    def test_small_labelled_subset_falls_back_without_unsolicited_logging(
        self, caplog: pytest.LogCaptureFixture, method: str
    ) -> None:
        X_cont, categorical_codes, target = mixed_classification_data()
        mask = np.array([True, False, False, False, False, False])

        with caplog.at_level(logging.INFO):
            weights = feature_weights(
                X_cont,
                categorical_codes,
                target,
                task="classification",
                method=method,
                dtype=DTYPE,
                device=DEVICE,
                labelled_mask=mask,
            )

        torch.testing.assert_close(
            weights,
            torch.full((3,), 1.0 / 3.0, dtype=DTYPE, device=DEVICE),
        )
        assert caplog.records == []


class TestMutualInformationFeatureWeights:
    @pytest.mark.parametrize("case", informative_cases())
    def test_mutual_information_ranks_the_informative_feature_first(self, case: str) -> None:
        X_cont, categorical_codes, target, task, expected = informative_cases()[case]

        weights = feature_weights(
            X_cont,
            categorical_codes,
            target,
            task=task,
            method="mutual_info",
            random_state=0,
            dtype=DTYPE,
            device=DEVICE,
        )

        assert weights.argmax().item() == expected

    def test_mutual_information_is_reproducible_with_a_fixed_seed(
        self,
    ) -> None:
        X_cont, categorical_codes, target = mixed_classification_data()

        first = feature_weights(
            X_cont,
            categorical_codes,
            target,
            task="classification",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=7,
        )
        second = feature_weights(
            X_cont,
            categorical_codes,
            target,
            task="classification",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=7,
        )

        assert torch.equal(first, second)

    def test_mutual_information_regression_is_reproducible_with_a_fixed_seed(self) -> None:
        """Rounded features tie, so the estimator's jitter decides the neighbours.

        Both features drive the output, so the jitter moves both scores and
        survives the normalisation into weights.
        """
        rng = np.random.default_rng(2)
        X_cont = np.round(rng.random((200, 2)), 1)
        target = (X_cont[:, 0] + X_cont[:, 1] + rng.normal(0, 0.1, 200)).reshape(-1, 1)

        first = feature_weights(
            X_cont,
            (),
            target,
            task="regression",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=5,
        )
        second = feature_weights(
            X_cont,
            (),
            target,
            task="regression",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=5,
        )

        assert torch.equal(first, second)

    def test_mutual_information_scores_two_labelled_rows(self) -> None:
        """Two rows are enough for the discrete estimator to separate the codes."""
        weights = feature_weights(
            np.empty((2, 0)),
            (np.array([0, 1]), np.zeros(2, dtype=np.int64)),
            np.array([0, 1]),
            task="classification",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=0,
        )

        torch.testing.assert_close(weights, torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE))

    def test_mutual_information_output_weights_change_the_multi_output_ranking(
        self,
    ) -> None:
        X_cont, target = two_target_regression_data()

        first = feature_weights(
            X_cont,
            (),
            target,
            task="regression",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=0,
            output_weights=np.array([10.0, 1.0]),
        )
        second = feature_weights(
            X_cont,
            (),
            target,
            task="regression",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=0,
            output_weights=np.array([1.0, 10.0]),
        )

        assert first[0] > first[1]
        assert second[1] > second[0]

    def test_mutual_information_falls_back_to_uniform_for_degenerate_features(
        self,
    ) -> None:
        target = np.arange(20) % 2

        weights = feature_weights(
            np.empty((20, 0)),
            (np.zeros(20, dtype=int), np.ones(20, dtype=int)),
            target,
            task="classification",
            method="mutual_info",
            dtype=DTYPE,
            device=DEVICE,
            random_state=0,
        )

        torch.testing.assert_close(weights, torch.full((2,), 0.5, dtype=DTYPE, device=DEVICE))


class TestStatisticalFeatureWeights:
    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    def test_statistical_helpers_are_state_free_and_do_not_mutate_inputs(
        self,
        method: str,
        extra_kwargs: dict[str, int],
    ) -> None:
        X_cont, categorical_codes, target = mixed_classification_data()
        original = (
            X_cont.copy(),
            tuple(code.copy() for code in categorical_codes),
            target.copy(),
        )

        first = feature_weights(
            X_cont,
            categorical_codes,
            target,
            task="classification",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )
        second = feature_weights(
            X_cont,
            categorical_codes,
            target,
            task="classification",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )

        assert np.array_equal(X_cont, original[0])
        assert all(
            np.array_equal(actual, expected)
            for actual, expected in zip(categorical_codes, original[1], strict=True)
        )
        assert np.array_equal(target, original[2])
        assert torch.equal(first, second)
        assert first.data_ptr() != second.data_ptr()

    @pytest.mark.parametrize("method", ["correlation", "mutual_info"])
    @pytest.mark.parametrize("truncated", ["categorical_codes", "target", "labelled_mask"])
    def test_statistical_helpers_reject_inconsistent_row_counts(
        self, method: str, truncated: str
    ) -> None:
        X_cont, categorical_codes, target = mixed_classification_data()
        rows = X_cont.shape[0]
        call: dict[str, object] = {
            "categorical_codes": categorical_codes,
            "target": target,
            "labelled_mask": np.ones(rows, dtype=bool),
        }
        call[truncated] = {
            "categorical_codes": (categorical_codes[0][:-1],),
            "target": target[:-1],
            "labelled_mask": np.ones(rows - 1, dtype=bool),
        }[truncated]

        with pytest.raises(ValueError, match="row"):
            feature_weights(
                X_cont,
                call["categorical_codes"],
                call["target"],
                task="classification",
                method=method,
                labelled_mask=call["labelled_mask"],
                dtype=DTYPE,
                device=DEVICE,
            )

    def test_statistical_scores_are_computed_in_float64(self) -> None:
        """A float32 input scores exactly as its widening, not in float32."""
        case = informative_cases()["continuous-classification"]
        X_cont, categorical_codes, target, task, _ = case
        single = X_cont.astype(np.float32)

        weights = feature_weights(
            single,
            categorical_codes,
            target,
            task=task,
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
        )
        widened = feature_weights(
            single.astype(np.float64),
            categorical_codes,
            target,
            task=task,
            method="correlation",
            dtype=DTYPE,
            device=DEVICE,
        )
        X_regression, outputs = two_target_regression_data()
        single_weighted, widened_weighted = (
            feature_weights(
                X_regression,
                (),
                outputs,
                task="regression",
                method="correlation",
                output_weights=np.array([3.0, 1.0], dtype=output_dtype),
                dtype=torch.float64,
                device=DEVICE,
            )
            for output_dtype in (np.float32, np.float64)
        )

        assert torch.equal(weights, widened)
        assert torch.equal(single_weighted, widened_weighted)

    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("labelled_rows", [1, 6], ids=["fallback", "scored"])
    def test_statistical_helpers_honour_the_requested_placement(
        self,
        method: str,
        extra_kwargs: dict[str, int],
        dtype: torch.dtype,
        labelled_rows: int,
    ) -> None:
        """The meta device pins placement on any machine, with or without CUDA."""
        X_cont, categorical_codes, target = mixed_classification_data()

        weights = feature_weights(
            X_cont,
            categorical_codes,
            target,
            task="classification",
            method=method,
            labelled_mask=np.arange(X_cont.shape[0]) < labelled_rows,
            dtype=dtype,
            device="meta",
            **extra_kwargs,
        )

        assert weights.dtype is dtype
        assert weights.device == torch.device("meta")

    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    def test_categorical_codes_are_scored_by_grouping_not_by_code_value(
        self,
        method: str,
        extra_kwargs: dict[str, int],
    ) -> None:
        rng = np.random.default_rng(4)
        codes = rng.integers(0, 20, 300)
        X_cont = rng.random(300).reshape(-1, 1)
        target = (codes % 2) + X_cont[:, 0] + rng.normal(0, 0.05, 300)
        relabelled = rng.permutation(20)[codes]

        weights = feature_weights(
            X_cont,
            (codes,),
            target,
            task="regression",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )
        permuted = feature_weights(
            X_cont,
            (relabelled,),
            target,
            task="regression",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )

        assert weights.min().item() > 0.0
        assert torch.equal(weights, permuted)

    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    def test_class_codes_float64_would_merge_stay_distinct(
        self,
        method: str,
        extra_kwargs: dict[str, int],
    ) -> None:
        # 2**53 and 2**53 + 1 are one float64 value; the matching code must still score.
        weights = feature_weights(
            np.empty((4, 0)),
            (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])),
            np.array([2**53, 2**53, 2**53 + 1, 2**53 + 1]),
            task="classification",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )

        torch.testing.assert_close(weights, torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE))

    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    def test_a_zero_output_weight_matches_dropping_that_output(
        self,
        method: str,
        extra_kwargs: dict[str, int],
    ) -> None:
        rng = np.random.default_rng(5)
        X_cont = rng.random(200).reshape(-1, 1)
        codes = rng.integers(0, 4, 200)
        target = np.column_stack(
            (
                X_cont[:, 0] + rng.normal(0, 0.1, 200),
                codes + rng.normal(0, 0.1, 200),
            )
        )

        weighted = feature_weights(
            X_cont,
            (codes,),
            target,
            task="regression",
            method=method,
            output_weights=np.array([1.0, 0.0]),
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )
        dropped = feature_weights(
            X_cont,
            (codes,),
            target[:, :1],
            task="regression",
            method=method,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )

        assert weighted.argmax().item() == 0
        assert torch.equal(weighted, dropped)

    @pytest.mark.parametrize("method", ["correlation", "mutual_info"])
    def test_statistical_helpers_reject_an_empty_feature_space(self, method: str) -> None:
        with pytest.raises(ValueError, match="feature"):
            feature_weights(
                np.empty((4, 0)),
                (),
                np.array([0, 0, 1, 1]),
                task="classification",
                method=method,
                dtype=DTYPE,
                device=DEVICE,
            )

    @pytest.mark.parametrize("method", ["correlation", "mutual_info"])
    @pytest.mark.parametrize(
        "output_weights",
        [
            pytest.param(np.array([1.0, -1.0]), id="negative"),
            pytest.param(np.array([1.0, np.inf]), id="infinite"),
            pytest.param(np.array([1.0, np.nan]), id="nan"),
            pytest.param(np.array([0.0, 0.0]), id="zero-mass"),
            pytest.param(np.ones(3), id="wrong-width"),
        ],
    )
    @pytest.mark.parametrize("labelled_rows", [0, 1, 400])
    def test_regression_output_weights_require_finite_non_negative_positive_mass(
        self,
        method: str,
        output_weights: np.ndarray,
        labelled_rows: int,
    ) -> None:
        X_cont, target = two_target_regression_data()

        with pytest.raises(ValueError, match="output_weights"):
            feature_weights(
                X_cont,
                (),
                target,
                task="regression",
                method=method,
                dtype=DTYPE,
                device=DEVICE,
                output_weights=output_weights,
                labelled_mask=np.arange(target.shape[0]) < labelled_rows,
            )

    def test_row_shaped_output_weights_match_their_vector(self) -> None:
        X_cont, target = two_target_regression_data()
        row, vector = (
            feature_weights(
                X_cont,
                (),
                target,
                task="regression",
                method="correlation",
                output_weights=output_weights,
                dtype=DTYPE,
                device=DEVICE,
            )
            for output_weights in (np.array([[10.0, 1.0]]), np.array([10.0, 1.0]))
        )

        assert torch.equal(row, vector)

    @pytest.mark.parametrize("method", ["correlation", "mutual_info"])
    @pytest.mark.parametrize("rows", [0, 1, 4])
    def test_regression_requires_at_least_one_output(self, method: str, rows: int) -> None:
        with pytest.raises(ValueError, match="output"):
            feature_weights(
                np.ones((rows, 2)),
                (),
                np.empty((rows, 0)),
                task="regression",
                method=method,
                dtype=DTYPE,
                device=DEVICE,
            )

    @pytest.mark.parametrize(("method", "extra_kwargs"), METHODS)
    def test_regression_output_weights_are_invariant_to_finite_rescaling(
        self,
        method: str,
        extra_kwargs: dict[str, int],
    ) -> None:
        X_cont, target = two_target_regression_data()
        output_weights = np.array([1.0, 0.9])
        expected = feature_weights(
            X_cont,
            (),
            target,
            task="regression",
            method=method,
            output_weights=output_weights,
            dtype=DTYPE,
            device=DEVICE,
            **extra_kwargs,
        )

        with np.errstate(over="raise", invalid="raise"):
            actual = feature_weights(
                X_cont,
                (),
                target,
                task="regression",
                method=method,
                output_weights=output_weights * 1e308,
                dtype=DTYPE,
                device=DEVICE,
                **extra_kwargs,
            )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("method", ["correlation", "mutual_info"])
    def test_statistical_helpers_reject_a_non_computation_dtype(self, method: str) -> None:
        X_cont, categorical_codes, target = mixed_classification_data()

        with pytest.raises(ValueError, match="dtype"):
            feature_weights(
                X_cont,
                categorical_codes,
                target,
                task="classification",
                method=method,
                dtype=torch.float16,
            )

    @pytest.mark.parametrize(
        ("controls", "match"),
        [
            pytest.param({"method": "correlation", "random_state": 0}, "random_state", id="seed"),
            pytest.param({"method": "pearson"}, "method", id="unknown-method"),
        ],
    )
    def test_statistical_helpers_reject_controls_the_method_cannot_use(
        self, controls: dict[str, object], match: str
    ) -> None:
        X_cont, categorical_codes, target = mixed_classification_data()

        with pytest.raises(ValueError, match=match):
            feature_weights(
                X_cont,
                categorical_codes,
                target,
                task="classification",
                dtype=DTYPE,
                device=DEVICE,
                **controls,
            )
