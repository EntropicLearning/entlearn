"""Regression data contract through Network initialisation."""

import warnings

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import regression_recipe, stage

from entlearn import Network
from entlearn.network.data import _RegressionSupervision


class TestRegressionTargets:
    def test_a_fully_missing_row_is_accepted_without_a_warning(self):
        X_cont = torch.linspace(0, 1, 6, dtype=DTYPE, device=DEVICE).reshape(3, 2)
        target = torch.tensor(
            [[1.0, 2.0], [float("nan"), float("nan")], [5.0, 6.0]], dtype=DTYPE, device=DEVICE
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Network.initialise(regression_recipe(), X_cont, target)

    def test_a_partially_missing_row_remains_available_to_input_initialisation(self):
        X_cont = torch.tensor([[0.0], [0.5], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor(
            [[1.0, 2.0], [3.0, float("nan")], [5.0, 6.0]], dtype=DTYPE, device=DEVICE
        )
        sample_weights = torch.tensor([1e-30, 1.0, 1e-30], dtype=DTYPE, device=DEVICE)

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
            torch.tensor([[0.5]], dtype=DTYPE, device=DEVICE),
        )

    @pytest.mark.parametrize(
        "target,message",
        (
            pytest.param(
                torch.arange(4, dtype=torch.int64), "floating length-T vector", id="integer"
            ),
            pytest.param(
                torch.arange(4, dtype=torch.float16),
                "floating length-T vector",
                id="lower-precision",
            ),
            pytest.param(torch.ones(4, 1, 1), "floating length-T vector", id="three-dimensional"),
            pytest.param(torch.ones(3), "same row count", id="wrong-row-count"),
            pytest.param(torch.empty(4, 0), "positive output dimension", id="empty-output"),
            pytest.param(
                torch.tensor([1.0, float("inf"), 2.0, 3.0]),
                "y values must be finite",
                id="infinite-labelled-value",
            ),
            pytest.param(
                torch.full((4,), float("nan")),
                "y must contain a labelled row",
                id="no-labelled-row",
            ),
        ),
    )
    def test_rejects_invalid_regression_targets(self, target, message):
        X_cont = torch.linspace(0, 1, 8, dtype=DTYPE, device=DEVICE).reshape(4, 2)

        with pytest.raises(ValueError, match=message):
            Network.initialise(regression_recipe(), X_cont, target.to(DEVICE))


class TestRegressionWeights:
    # No public reading exposes the output weights omega, and a one-cluster head's prediction
    # is invariant to their total, so the staged weights are checked against loss.md directly.
    @pytest.mark.parametrize("weighted", [False, True], ids=["sample-only", "task-weighted"])
    def test_output_weights_follow_the_documented_formula(self, weighted):
        X_cont = torch.linspace(0, 1, 6, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        nan = float("nan")
        target = torch.tensor([0.2, nan, 0.7, nan, 0.9, 0.1], dtype=DTYPE, device=DEVICE)
        sample_weights = torch.tensor([1.0, 4.0, 2.0, 3.0, 5.0, 1.0], dtype=DTYPE, device=DEVICE)
        task_weights = (
            torch.tensor([3.0, 1.0, 0.5, 1.0, 2.0, 7.0], dtype=DTYPE, device=DEVICE)
            if weighted
            else None
        )

        _, data = stage(
            regression_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            task_weights=task_weights,
        )

        # omega_t = f w_t q_t / sum_L w_s q_s on labelled rows, f the labelled share of w.
        w = sample_weights / sample_weights.sum()
        q = torch.ones_like(w) if task_weights is None else task_weights
        labelled = ~target.isnan()
        f = w[labelled].sum()
        expected = torch.where(labelled, f * w * q / (w * q)[labelled].sum(), 0.0)
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        torch.testing.assert_close(supervision.row_weights, expected)

    def test_task_weights_do_not_change_centroid_selection(self):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        sample_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, device=DEVICE)

        first = Network.initialise(
            regression_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            task_weights=torch.tensor([1.0, 7.0, 2.0, 4.0], dtype=DTYPE, device=DEVICE),
            seed=11,
        )
        second = Network.initialise(
            regression_recipe(),
            X_cont,
            target,
            sample_weights=sample_weights,
            task_weights=torch.tensor([9.0, 1.0, 5.0, 2.0], dtype=DTYPE, device=DEVICE),
            seed=11,
        )

        assert torch.equal(
            first.input_geometry.continuous_centroids, second.input_geometry.continuous_centroids
        )

    @pytest.mark.parametrize(
        "weight,message",
        (
            pytest.param(torch.ones(3), "task_weights must be a length-4", id="shape"),
            pytest.param(torch.zeros(4), "positive-weight labelled mass", id="zero-labelled-mass"),
        ),
    )
    def test_rejects_invalid_task_weights(self, weight, message):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.arange(4, dtype=DTYPE, device=DEVICE)

        with pytest.raises(ValueError, match=message):
            Network.initialise(
                regression_recipe(), X_cont, target, task_weights=weight.to(DTYPE).to(DEVICE)
            )

    def test_rejects_class_weights(self):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.arange(4, dtype=DTYPE, device=DEVICE)

        with pytest.raises(ValueError, match="does not accept class_weights"):
            Network.initialise(
                regression_recipe(),
                X_cont,
                target,
                class_weights=torch.ones(1, dtype=DTYPE, device=DEVICE),
            )

    def test_rejects_task_weights_that_are_neither_a_tensor_nor_a_callable(self):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)

        with pytest.raises(ValueError, match=r"task_weights .*tensor or callable"):
            Network.initialise(regression_recipe(), X_cont, X_cont[:, 0], task_weights=[1.0] * 4)

    def test_a_target_without_a_labelled_row_is_refused_before_the_callable_runs(self):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.full((4,), float("nan"), dtype=DTYPE, device=DEVICE)

        def weighting(labelled_target):
            raise AssertionError("task_weights called without a labelled row")

        with pytest.raises(ValueError, match="y must contain a labelled row"):
            Network.initialise(regression_recipe(), X_cont, target, task_weights=weighting)

    @pytest.mark.parametrize(
        ("W_M", "message"),
        [
            pytest.param((1.0,), r"W_M has length 1 .* regression targets is 2", id="short"),
            pytest.param((1.0, 2.0, 3.0), r"W_M has length 3 .* targets is 2", id="long"),
            pytest.param((1e39, 1.0), r"W_M .*positive in the computation dtype", id="overflow"),
            pytest.param((1e-50, 1.0), r"W_M .*positive in the computation dtype", id="underflow"),
        ],
    )
    def test_rejects_invalid_fixed_output_weights(self, W_M, message):
        X_cont = torch.linspace(0, 1, 8, dtype=DTYPE, device=DEVICE).reshape(4, 2)

        with pytest.raises(ValueError, match=message):
            Network.initialise(
                regression_recipe(W_M=W_M), X_cont, X_cont, computation_dtype=torch.float32
            )

    def test_fixed_output_weights_match_the_output_dimension(self):
        X_cont = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = X_cont[:, 0]

        Network.initialise(regression_recipe(W_M=(1.0,)), X_cont, target)
        Network.initialise(regression_recipe(W_M=(2.0, 3.0)), X_cont, torch.stack([target] * 2, 1))
