"""Regression data contract through Network initialisation."""

import warnings

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import regression_recipe, stage

from entlearn import Network
from entlearn.network.data import _RegressionSupervision


class TestRegressionTargets:
    def test_warns_once_and_masks_a_partially_missing_row(self):
        X_cont = torch.arange(8, dtype=torch.float64).reshape(4, 2)
        target = torch.tensor(
            [[1.0, 2.0], [3.0, float("nan")], [float("nan"), float("nan")], [5.0, 6.0]],
            dtype=torch.float32,
        )

        with pytest.warns(UserWarning, match="1 regression target row.*partially missing") as seen:
            state = Network.initialise(regression_recipe(), X_cont, target, seed=3)

        assert len(seen) == 1
        assert state.input_geometry.continuous_centroids.shape == (2, 2)
        assert state.input_geometry.continuous_centroids.dtype is torch.float64

    def test_a_fully_missing_row_is_unlabelled_without_a_warning(self):
        X_cont = torch.arange(6, dtype=torch.float64).reshape(3, 2)
        target = torch.tensor(
            [[1.0, 2.0], [float("nan"), float("nan")], [5.0, 6.0]],
            dtype=torch.float64,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            state = Network.initialise(regression_recipe(), X_cont, target)

        assert state.input_geometry.continuous_centroids.shape == (2, 2)

    def test_a_partially_missing_row_remains_available_to_input_initialisation(self):
        X_cont = torch.tensor([[0.0], [10.0], [20.0]], dtype=torch.float64)
        target = torch.tensor(
            [[1.0, 2.0], [3.0, float("nan")], [5.0, 6.0]],
            dtype=torch.float64,
        )
        sample_weights = torch.tensor([1e-30, 1.0, 1e-30], dtype=torch.float64)

        with pytest.warns(UserWarning, match="partially missing"):
            state = Network.initialise(
                regression_recipe(K=1),
                X_cont,
                target,
                sample_weights=sample_weights,
                seed=3,
            )

        assert torch.equal(
            state.input_geometry.continuous_centroids,
            torch.tensor([[10.0]], dtype=torch.float64),
        )

    @pytest.mark.parametrize(
        "target",
        (
            pytest.param(torch.arange(4, dtype=torch.int64), id="integer"),
            pytest.param(torch.arange(4, dtype=torch.float16), id="lower-precision"),
            pytest.param(torch.ones(4, dtype=torch.complex64), id="complex"),
            pytest.param(torch.ones(4, 1, 1), id="three-dimensional"),
            pytest.param(torch.ones(3), id="wrong-row-count"),
            pytest.param(torch.empty(4, 0), id="empty-output"),
            pytest.param(
                torch.tensor([1.0, float("inf"), 2.0, 3.0]),
                id="infinite-labelled-value",
            ),
            pytest.param(torch.full((4,), float("nan")), id="no-labelled-row"),
        ),
    )
    def test_rejects_invalid_regression_targets(self, target):
        X_cont = torch.arange(8, dtype=torch.float64).reshape(4, 2)

        with pytest.raises(ValueError):
            Network.initialise(regression_recipe(), X_cont, target)


class TestRegressionWeights:
    def test_output_weights_are_the_sample_weights_on_labelled_rows(self):
        X_cont = torch.linspace(0, 1, 6, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        nan = float("nan")
        target = torch.tensor([0.2, nan, 0.7, nan, 0.9, 0.1], dtype=DTYPE, device=DEVICE)
        sample_weights = torch.tensor([1.0, 4.0, 2.0, 3.0, 5.0, 1.0], dtype=DTYPE, device=DEVICE)

        _, data = stage(regression_recipe(), X_cont, target, sample_weights=sample_weights)

        expected = torch.where(target.isnan(), 0.0, data.sample_weights)
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        torch.testing.assert_close(supervision.row_weights, expected)

    def test_task_weights_keep_the_output_weights_at_the_labelled_share(self):
        X_cont = torch.linspace(0, 1, 6, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        nan = float("nan")
        target = torch.tensor([0.2, nan, 0.7, nan, 0.9, 0.1], dtype=DTYPE, device=DEVICE)
        sample_weights = torch.tensor([1.0, 4.0, 2.0, 3.0, 5.0, 1.0], dtype=DTYPE, device=DEVICE)
        task_weights = torch.tensor([3.0, 1.0, 0.5, 1.0, 2.0, 7.0], dtype=DTYPE, device=DEVICE)

        _, data = stage(
            regression_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            task_weights=task_weights,
        )

        labelled_share = data.sample_weights[~target.isnan()].sum()
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        torch.testing.assert_close(supervision.row_weights.sum(), labelled_share)

    def test_task_weights_do_not_change_centroid_selection(self):
        X_cont = torch.tensor([[0.0], [10.0], [20.0], [30.0]], dtype=torch.float64)
        target = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        sample_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        first = Network.initialise(
            regression_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            task_weights=torch.tensor([1.0, 7.0, 2.0, 4.0]),
            seed=11,
        )
        second = Network.initialise(
            regression_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            task_weights=torch.tensor([9.0, 1.0, 5.0, 2.0]),
            seed=11,
        )

        assert torch.equal(
            first.input_geometry.continuous_centroids, second.input_geometry.continuous_centroids
        )

    @pytest.mark.parametrize(
        "weight",
        (
            pytest.param(torch.ones(3), id="shape"),
            pytest.param(torch.ones(4, dtype=torch.int64), id="dtype"),
            pytest.param(torch.tensor([1.0, float("nan"), 1.0, 1.0]), id="non-finite"),
            pytest.param(torch.tensor([1.0, -1.0, 1.0, 1.0]), id="negative"),
            pytest.param(torch.zeros(4), id="zero-labelled-mass"),
        ),
    )
    def test_rejects_invalid_task_weights(self, weight):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1)
        target = torch.arange(4, dtype=torch.float64)

        with pytest.raises(ValueError):
            Network.initialise(regression_recipe(), X_cont, target, task_weights=weight)

    def test_rejects_class_weights(self):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1)
        target = torch.arange(4, dtype=torch.float64)

        with pytest.raises(ValueError, match="regression does not accept class_weights"):
            Network.initialise(regression_recipe(), X_cont, target, class_weights=torch.ones(1))

    def test_rejects_task_weights_that_are_neither_a_tensor_nor_a_callable(self):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)

        with pytest.raises(ValueError, match="task_weights must be a tensor or callable"):
            Network.initialise(regression_recipe(), X_cont, X_cont[:, 0], task_weights=[1.0] * 4)

    @pytest.mark.parametrize("W_M", [(1e39, 1.0), (1e-50, 1.0)], ids=["overflow", "underflow"])
    def test_rejects_fixed_output_weights_outside_float32(self, W_M):
        X_cont = torch.linspace(0, 1, 8, dtype=DTYPE, device=DEVICE).reshape(4, 2)

        with pytest.raises(ValueError, match="fixed W_M must be finite and positive"):
            Network.initialise(
                regression_recipe(W_M=W_M), X_cont, X_cont, computation_dtype=torch.float32
            )

    def test_a_one_dimensional_target_has_one_fixed_output_weight(self):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = X_cont[:, 0]

        Network.initialise(regression_recipe(W_M=(1.0,)), X_cont, target)
        with pytest.raises(ValueError, match="length 2 but the output dimension is 1"):
            Network.initialise(regression_recipe(W_M=(1.0, 2.0)), X_cont, target)

    def test_accepts_and_validates_fixed_output_weights_at_the_data_width(self):
        X_cont = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        target = torch.arange(8, dtype=torch.float64).reshape(4, 2)

        state = Network.initialise(regression_recipe(W_M=(2.0, 3.0)), X_cont, target)

        assert state.input_geometry.continuous_centroids.dtype is torch.float32
