"""Classification data contract through Network initialisation."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import classification_recipe, stage

from entlearn import Network
from entlearn.network.data import _ClassificationSupervision


class TestCategoricalFeatures:
    @pytest.mark.parametrize("wrap", (tuple, list))
    def test_initialises_categorical_only_data_from_hard_codes(self, wrap):
        X_cont = torch.empty(4, 0, dtype=torch.float64)
        X_cat = wrap((torch.tensor([0, 1, 1, 0], dtype=torch.int64),))
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

        state = Network.initialise(classification_recipe(), X_cont, y, X_cat=X_cat, seed=4)

        assert state.input_geometry.continuous_centroids.shape == (2, 0)
        assert len(state.input_geometry.categorical_centroids) == 1
        assert state.input_geometry.categorical_centroids[0].shape == (2, 2)
        one_hot_codes = {(1.0, 0.0), (0.0, 1.0)}
        for centroid in state.input_geometry.categorical_centroids[0]:
            assert tuple(centroid.tolist()) in one_hot_codes
        assert state.input_geometry.feature_weights is not None
        assert torch.equal(state.input_geometry.feature_weights, torch.ones(1, dtype=torch.float64))

    def test_initialises_distribution_valued_categorical_data_without_mutation(self):
        X_cont = torch.empty(4, 0, dtype=torch.float32)
        distribution = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        )
        original = distribution.clone()
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

        state = Network.initialise(
            classification_recipe(), X_cont, y, X_cat=(distribution,), seed=4
        )

        assert state.input_geometry.categorical_centroids[0].dtype is torch.float32
        assert all(
            any(torch.equal(centroid, row) for row in original)
            for centroid in state.input_geometry.categorical_centroids[0]
        )
        assert torch.equal(distribution, original)

    @pytest.mark.parametrize(
        "feature,message",
        (
            pytest.param(
                torch.tensor([0, 1, 1, 0], dtype=torch.int32), "int64 code tensor", id="int32-codes"
            ),
            pytest.param(
                torch.tensor([0, 1, -1, 0], dtype=torch.int64),
                "codes must be non-negative",
                id="negative-code",
            ),
            pytest.param(
                torch.zeros(4, dtype=torch.int64), "at least two levels", id="one-level-codes"
            ),
            pytest.param(
                torch.tensor(
                    [[1.0, 0.0], [0.5, 0.5], [float("nan"), 0.0], [0.0, 1.0]],
                    dtype=torch.float64,
                ),
                "only finite values",
                id="non-finite-distribution",
            ),
            pytest.param(
                torch.tensor(
                    [[1.0, 0.0], [0.5, 0.5], [-0.1, 1.1], [0.0, 1.0]],
                    dtype=torch.float64,
                ),
                r"X_cat\[0\] must be non-negative",
                id="negative-distribution",
            ),
            pytest.param(
                torch.tensor(
                    [[1.0, 0.0], [0.25, 0.25], [0.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float64,
                ),
                "sum to one",
                id="non-simplex-distribution",
            ),
            pytest.param(
                torch.ones(4, 1, dtype=torch.float64),
                r"\(T, M_i\) computation-dtype",
                id="one-level-distribution",
            ),
            pytest.param(
                torch.ones(3, 2, dtype=torch.float64) / 2,
                r"\(T, M_i\) computation-dtype",
                id="wrong-row-count",
            ),
            pytest.param(
                torch.ones(4, 2, dtype=torch.float32) / 2,
                r"\(T, M_i\) computation-dtype",
                id="wrong-dtype",
            ),
            pytest.param(
                torch.ones(4, 2, dtype=torch.float16) / 2,
                r"\(T, M_i\) computation-dtype",
                id="lower-precision",
            ),
        ),
    )
    def test_rejects_invalid_categorical_features(self, feature, message):
        X_cont = torch.empty(4, 0, dtype=torch.float64)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match=message):
            Network.initialise(classification_recipe(), X_cont, y, X_cat=(feature,))


class TestClassificationTargets:
    def test_distribution_targets_do_not_set_the_computation_dtype(self):
        other = torch.float32 if DTYPE is torch.float64 else torch.float64
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75], [0.0, 0.0], [0.0, 1.0]], dtype=other, device=DEVICE
        )

        state = Network.initialise(classification_recipe(), X_cont, target, seed=3)

        assert state.input_geometry.continuous_centroids.dtype == DTYPE


# Six rows with sample weights summing to 16; rows 1 and 3 are unlabelled, so the labelled
# rows carry 9 / 16 of the weight.
_SIX_ROW_WEIGHTS = (1.0, 4.0, 2.0, 3.0, 5.0, 1.0)
_SIX_ROW_LABELLED_WEIGHTS = (1 / 16, 0.0, 2 / 16, 0.0, 5 / 16, 1 / 16)


class TestClassificationWeights:
    def test_output_weights_are_the_sample_weights_on_labelled_rows(self):
        X_cont = torch.linspace(0, 1, 6, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.tensor([0, -1, 1, -1, 1, 0], device=DEVICE)
        sample_weights = torch.tensor(_SIX_ROW_WEIGHTS, dtype=DTYPE, device=DEVICE)

        _, data = stage(classification_recipe(), X_cont, target, sample_weights=sample_weights)

        expected = torch.tensor(_SIX_ROW_LABELLED_WEIGHTS, dtype=DTYPE, device=DEVICE)
        supervision = data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        torch.testing.assert_close(supervision.weighted_target.sum(dim=1), expected)

    def test_class_weights_keep_the_output_weights_at_the_labelled_share(self):
        X_cont = torch.linspace(0, 1, 6, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.tensor([0, -1, 1, -1, 1, 0], device=DEVICE)
        sample_weights = torch.tensor(_SIX_ROW_WEIGHTS, dtype=DTYPE, device=DEVICE)
        class_weights = torch.tensor([3.0, 0.5], dtype=DTYPE, device=DEVICE)

        _, data = stage(
            classification_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            class_weights=class_weights,
        )

        labelled_share = torch.tensor(9 / 16, dtype=DTYPE, device=DEVICE)
        supervision = data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        torch.testing.assert_close(supervision.weighted_target.sum(), labelled_share)

    def test_underflowing_class_products_keep_the_sample_weights(self):
        X_cont = torch.linspace(0, 1, 6, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        labelled = torch.tensor([True, False, True, False, True, True], device=DEVICE)
        target = torch.tensor([1e-20, 1.0], dtype=torch.float32, device=DEVICE).repeat(6, 1)
        target[~labelled] = 0.0
        sample_weights = torch.tensor(_SIX_ROW_WEIGHTS, dtype=torch.float32, device=DEVICE)
        class_weights = torch.tensor([1e-30, 0.0], dtype=torch.float32, device=DEVICE)

        _, data = stage(
            classification_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            class_weights=class_weights,
        )

        # Each supported product is below 1e-50 and underflows in float32, so only the
        # log-domain path can recover the weights.
        supervision = data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        expected = torch.tensor(_SIX_ROW_LABELLED_WEIGHTS, dtype=torch.float32, device=DEVICE)
        torch.testing.assert_close(supervision.weighted_target.sum(dim=1), expected)

    def test_normalises_large_finite_sample_weights_without_overflow(self):
        X_cont = torch.linspace(0, 1, 4, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        target = torch.tensor([0, 1, 0, 1], device=DEVICE)
        # Each weight is finite in float32, but their sum overflows.
        sample_weights = torch.full((4,), 3e38, dtype=torch.float32, device=DEVICE)

        _, data = stage(classification_recipe(K=1), X_cont, target, sample_weights=sample_weights)

        quarter = torch.full((4,), 0.25, dtype=torch.float32, device=DEVICE)
        torch.testing.assert_close(data.sample_weights, quarter, rtol=0, atol=0)

    def test_class_weights_do_not_select_centroids(self):
        X_cont = torch.tensor([[0.0], [0.5], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([0, 1, 1], device=DEVICE)
        # The sample weights alone pick row 2; times these class weights, row 0 would outweigh it.
        sample_weights = torch.tensor([1e-30, 1e-30, 3.0], dtype=DTYPE, device=DEVICE)
        class_weights = torch.tensor([1e35, 1.0], dtype=DTYPE, device=DEVICE)
        sample_original = sample_weights.clone()
        class_original = class_weights.clone()

        state = Network.initialise(
            classification_recipe(K=1),
            X_cont,
            target,
            sample_weights=sample_weights,
            class_weights=class_weights,
            seed=9,
        )

        assert torch.equal(
            state.input_geometry.continuous_centroids,
            torch.tensor([[1.0]], dtype=DTYPE, device=DEVICE),
        )
        assert torch.equal(sample_weights, sample_original)
        assert torch.equal(class_weights, class_original)

    @pytest.mark.parametrize(
        ("keyword", "weight", "message"),
        (
            pytest.param(
                "sample_weights",
                torch.ones(2),
                "sample_weights must be a length-3",
                id="sample-shape",
            ),
            pytest.param(
                "sample_weights",
                torch.ones(3, dtype=torch.int64),
                "sample_weights must be a length-3",
                id="sample-dtype",
            ),
            pytest.param(
                "sample_weights",
                torch.tensor([1.0, float("nan"), 1.0]),
                "sample_weights must contain only finite",
                id="sample-non-finite",
            ),
            pytest.param(
                "class_weights", torch.ones(3), "class_weights must be a length-2", id="class-shape"
            ),
            pytest.param(
                "class_weights",
                torch.ones(2, dtype=torch.int64),
                "class_weights must be a length-2",
                id="class-dtype",
            ),
            pytest.param(
                "class_weights",
                torch.tensor([1.0, float("inf")]),
                "class_weights must contain only finite",
                id="class-non-finite",
            ),
            pytest.param(
                "class_weights",
                torch.tensor([1.0, -1.0]),
                r"class_weights .*non-negative",
                id="class-negative",
            ),
        ),
    )
    def test_rejects_invalid_weights(self, keyword, weight, message):
        X_cont = torch.arange(3, dtype=torch.float64).unsqueeze(1)
        target = torch.tensor([0, 1, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match=message):
            Network.initialise(classification_recipe(K=1), X_cont, target, **{keyword: weight})

    def test_requires_positive_weight_on_a_labelled_row(self):
        X_cont = torch.arange(3, dtype=torch.float64).unsqueeze(1)
        target = torch.tensor([-1, 0, 1], dtype=torch.int64)
        class_weights = torch.zeros(2, dtype=torch.float64)

        with pytest.raises(ValueError, match="positive-weight labelled mass"):
            Network.initialise(
                classification_recipe(K=1), X_cont, target, class_weights=class_weights
            )

    @pytest.mark.parametrize("weight", [-1.0, 0.0])
    def test_fit_rejects_a_non_positive_sample_weight(self, weight):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.tensor([0, 1, 0, 1], device=DEVICE)
        sample_weights = torch.tensor([1.0, weight, 1.0, 1.0], dtype=DTYPE, device=DEVICE)

        with pytest.raises(ValueError, match="sample_weights must be positive"):
            Network.fit(classification_recipe(), X_cont, target, sample_weights=sample_weights)

    def test_fit_rejects_sample_weights_that_normalise_to_zero(self):
        X_cont = torch.linspace(0, 1, 4, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        target = torch.tensor([0, 1, 0, 1], device=DEVICE)
        sample_weights = torch.tensor([3e38, 1e-10, 1.0, 1.0], dtype=torch.float32, device=DEVICE)

        with pytest.raises(ValueError, match="too wide a range"):
            Network.fit(classification_recipe(), X_cont, target, sample_weights=sample_weights)

    def test_rejects_per_row_task_weights(self):
        X_cont = torch.arange(3, dtype=torch.float64).unsqueeze(1)
        target = torch.tensor([0, 1, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match="does not accept task_weights"):
            Network.initialise(
                classification_recipe(K=1), X_cont, target, task_weights=torch.ones(3)
            )

    @pytest.mark.parametrize(
        "target,message",
        (
            pytest.param(
                torch.tensor([0, 1, 1, 0], dtype=torch.int32),
                "y must be a length-T int64",
                id="int32-codes",
            ),
            pytest.param(
                torch.tensor([0, 1, -2, 0], dtype=torch.int64),
                "permits -1 as its only negative",
                id="negative-code",
            ),
            pytest.param(
                torch.full((4,), -1, dtype=torch.int64),
                "y must contain a labelled row",
                id="no-labelled-code",
            ),
            pytest.param(
                torch.ones(4, 2, dtype=torch.float16) / 2,
                "int64 codes or a floating",
                id="lower-precision",
            ),
            pytest.param(
                torch.tensor(
                    [[1.0, 0.0], [0.25, 0.25], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
                "rows must sum to one",
                id="non-simplex-distribution",
            ),
            pytest.param(
                torch.zeros(4, 2, dtype=torch.float64),
                "y must contain a labelled row",
                id="no-labelled-distribution",
            ),
        ),
    )
    def test_rejects_invalid_classification_targets(self, target, message):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1)

        with pytest.raises(ValueError, match=message):
            Network.initialise(classification_recipe(), X_cont, target)

    def test_inferred_classes_require_contiguous_codes(self):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1)
        target = torch.tensor([0, 2, 0, 2], dtype=torch.int64)

        with pytest.raises(ValueError, match="n_classes is not specified"):
            Network.initialise(classification_recipe(), X_cont, target)


class TestDistributionTolerance:
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_accepts_roundoff_at_the_dtype_aware_simplex_tolerance(self, dtype):
        delta = 8 * torch.finfo(dtype).eps
        X_cont = torch.empty(4, 0, dtype=dtype)
        X_cat = (torch.tensor([[1 + delta, 0.0]] * 4, dtype=dtype),)
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        state = Network.initialise(classification_recipe(K=1), X_cont, target, X_cat=X_cat)

        assert len(state.input_geometry.categorical_centroids) == 1

    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_rejects_rows_outside_the_dtype_aware_simplex_tolerance(self, dtype):
        delta = 32 * torch.finfo(dtype).eps
        X_cont = torch.empty(4, 0, dtype=dtype)
        X_cat = (torch.tensor([[1 + delta, 0.0]] * 4, dtype=dtype),)
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match="dtype tolerance"):
            Network.initialise(classification_recipe(K=1), X_cont, target, X_cat=X_cat)
