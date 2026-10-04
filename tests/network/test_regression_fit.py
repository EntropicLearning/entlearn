"""Public behaviour of a fitted shallow regression Network."""

from itertools import pairwise

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE
from network._fixtures import (
    assert_non_increasing,
    assert_simplex_rows,
    materialise,
    noisy_regression_blobs,
    regression_blobs,
    regression_recipe,
)

from entlearn import Input, Network
from entlearn.network.fit import _fit_iteration_


class TestRegressionFit:
    @pytest.mark.parametrize("output_width", (1, 3), ids=("single-output", "multi-output"))
    def test_fits_and_predicts_piecewise_constant_targets(self, output_width: int) -> None:
        X_cont, target, values = regression_blobs(output_width)
        public_target = target[:, 0] if output_width == 1 else target
        recipe = regression_recipe(Input(K=2, epsilon=0.0))

        network = Network.fit(recipe, X_cont, public_target, max_iter=10, seed=7)
        prediction = network.predict(torch.tensor([[0.15], [0.85]], dtype=DTYPE, device=DEVICE))

        assert prediction.shape == (2, output_width)
        torch.testing.assert_close(prediction, values)
        assert network.schema.task == "regression"
        assert output_width == network.schema.M

    @pytest.mark.parametrize(
        ("epsilon_M", "fixed"),
        (
            (float("inf"), None),
            (float("inf"), (1.0, 2.0, 3.0)),
            (0.0, None),
            (0.2, None),
        ),
        ids=("implicit-uniform", "fixed", "learned-hard", "learned-soft"),
    )
    def test_diagnostics_record_non_increasing_complete_iterations(
        self,
        epsilon_M: float,
        fixed: tuple[float, ...] | None,
    ) -> None:
        X_cont, target = noisy_regression_blobs()

        network = Network.fit(
            regression_recipe(epsilon_M=epsilon_M, W_M=fixed),
            X_cont,
            target,
            max_iter=4,
            tol=0.0,
            seed=4,
        )

        assert network.diagnostics.n_iter == len(network.diagnostics.loss_history) - 1
        for previous, current in pairwise(network.diagnostics.loss_history):
            assert_non_increasing(previous, current, "iteration")

    def test_concrete_task_weights_combine_with_sample_weights(self) -> None:
        X_cont = torch.arange(4, dtype=DTYPE, device=DEVICE).unsqueeze(1) / 4
        target = torch.tensor([1.0, 3.0, 9.0, float("nan")], dtype=DTYPE, device=DEVICE)
        sample_weights = torch.tensor([1.0, 2.0, 1.0, 8.0], dtype=DTYPE, device=DEVICE)
        task_weights = torch.tensor([3.0, 1.0, 2.0, 7.0], dtype=DTYPE, device=DEVICE)
        expected = (target[:3] * sample_weights[:3] * task_weights[:3]).sum()
        expected /= (sample_weights[:3] * task_weights[:3]).sum()

        network = Network.fit(
            regression_recipe(K=1),
            X_cont,
            target,
            sample_weights=sample_weights,
            task_weights=task_weights,
            max_iter=1,
        )

        prediction = network.predict(X_cont)
        torch.testing.assert_close(prediction[:, 0], expected.expand(4))

    @pytest.mark.parametrize("output_width", (1, 2), ids=("one-dimensional", "multi-output"))
    def test_callable_task_weights_receive_only_labelled_targets_once(
        self, output_width: int
    ) -> None:
        X_cont = torch.arange(5, dtype=DTYPE, device=DEVICE).unsqueeze(1) / 4
        target = torch.tensor(
            [
                [1.0, 10.0],
                [float("nan"), float("nan")],
                [4.0, 20.0],
                [8.0, 40.0],
                [float("nan"), float("nan")],
            ],
            dtype=DTYPE,
            device=DEVICE,
        )[:, :output_width]
        public_target = target[:, 0] if output_width == 1 else target
        calls: list[torch.Tensor] = []

        def weighting(labelled_target: torch.Tensor) -> torch.Tensor:
            calls.append(labelled_target.clone())
            return torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)

        network = Network.fit(
            regression_recipe(K=1),
            X_cont,
            public_target,
            task_weights=weighting,
            max_iter=1,
        )

        assert len(calls) == 1
        labelled = torch.tensor([True, False, True, True, False], device=DEVICE)
        assert torch.equal(calls[0], public_target[labelled])
        callable_weights = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        expected = (target[labelled] * callable_weights[:, None]).sum(
            dim=0
        ) / callable_weights.sum()
        torch.testing.assert_close(network.predict(X_cont), expected.expand(5, output_width))

    def test_partially_missing_rows_warn_once_and_do_not_reach_the_head(self) -> None:
        X_cont = torch.arange(4, dtype=DTYPE, device=DEVICE).unsqueeze(1) / 4
        target = torch.tensor(
            [[1.0, 2.0], [100.0, float("nan")], [5.0, 8.0], [float("nan"), float("nan")]],
            dtype=DTYPE,
            device=DEVICE,
        )

        with pytest.warns(UserWarning, match="1 regression target row.*partially missing") as seen:
            network = Network.fit(regression_recipe(K=1), X_cont, target)

        assert len(seen) == 1
        expected = torch.tensor([3.0, 5.0], dtype=DTYPE, device=DEVICE).expand(4, 2)
        torch.testing.assert_close(network.predict(X_cont), expected)

    @pytest.mark.parametrize(
        ("returned", "message"),
        (
            (torch.ones(2, dtype=DTYPE, device=DEVICE), "3 labelled rows"),
            (torch.ones((3, 1), dtype=DTYPE, device=DEVICE), "3 labelled rows"),
            ([1.0, 2.0, 3.0], "3 labelled rows"),
            (
                torch.tensor([1.0, float("nan"), 1.0], dtype=DTYPE, device=DEVICE),
                "callable task_weights must contain only finite values",
            ),
            (torch.tensor([1.0, 0.0, 1.0], dtype=DTYPE, device=DEVICE), "strictly positive"),
        ),
        ids=("length", "rank", "not-a-tensor", "finite", "positive"),
    )
    def test_callable_task_weights_are_validated(
        self, returned: torch.Tensor | list[float], message: str
    ) -> None:
        X_cont = torch.arange(4, dtype=DTYPE, device=DEVICE).unsqueeze(1) / 4
        target = torch.tensor([1.0, float("nan"), 4.0, 8.0], dtype=DTYPE, device=DEVICE)

        with pytest.raises(ValueError, match=message):
            Network.fit(
                regression_recipe(K=1),
                X_cont,
                target,
                task_weights=lambda _: returned,
                max_iter=1,
            )

    def test_callable_task_weights_are_validated_in_the_computation_dtype(self) -> None:
        """A weight that survives the callable's dtype but not the fit's is not positive."""
        X_cont = torch.arange(4, dtype=torch.float64, device=DEVICE).unsqueeze(1) / 4
        target = torch.tensor([1.0, float("nan"), 4.0, 8.0], dtype=torch.float64, device=DEVICE)
        returned = torch.tensor([1.0, 1e-300, 1.0], dtype=torch.float64, device=DEVICE)

        with pytest.raises(ValueError, match="strictly positive"):
            Network.fit(
                regression_recipe(K=1),
                X_cont,
                target,
                task_weights=lambda _: returned,
                computation_dtype=torch.float32,
                max_iter=1,
            )

    @pytest.mark.parametrize(
        ("epsilon_M", "fixed", "expected"),
        (
            (float("inf"), None, None),
            (float("inf"), (2.0, 3.0, 5.0), (0.2, 0.3, 0.5)),
        ),
        ids=("implicit-uniform", "fixed"),
    )
    def test_pinned_output_weights_remain_fixed(
        self,
        epsilon_M: float,
        fixed: tuple[float, ...] | None,
        expected: tuple[float, ...] | None,
    ) -> None:
        X_cont, target, _ = regression_blobs(3)

        network = Network.fit(
            regression_recipe(epsilon_M=epsilon_M, W_M=fixed),
            X_cont,
            target,
            max_iter=3,
            seed=4,
        )

        if expected is None:
            assert network._graph.head.W_M is None
        else:
            torch.testing.assert_close(
                network._graph.head.W_M,
                torch.tensor(expected, dtype=DTYPE, device=DEVICE),
            )

    def test_finite_output_weight_temperature_learns_a_simplex(self) -> None:
        X_cont, target = noisy_regression_blobs()

        network = Network.fit(
            regression_recipe(epsilon_M=0.05),
            X_cont,
            target,
            max_iter=3,
            seed=4,
        )

        output_weights = network._graph.head.W_M
        assert output_weights is not None
        assert_simplex_rows(output_weights.unsqueeze(0))
        assert not torch.equal(
            output_weights,
            torch.full((3,), 1.0 / 3.0, dtype=DTYPE, device=DEVICE),
        )

    def test_publication_releases_regression_fit_caches(self) -> None:
        X_cont, target, _ = regression_blobs(2)

        network = Network.fit(
            regression_recipe(epsilon_M=0.2),
            X_cont,
            target,
            max_iter=1,
        )

        assert network._graph.head.cache is None
        with pytest.raises(RuntimeError, match="holds no cache"):
            _ = network._graph.head.live_cache

    @pytest.mark.parametrize(
        "epsilon_M",
        (float("inf"), 0.0, 0.2),
        ids=("fixed", "learned-hard", "learned-soft"),
    )
    def test_complete_regression_iteration_allocates_no_tensor_storage(
        self, epsilon_M: float
    ) -> None:
        X_cont, target = noisy_regression_blobs()
        session = materialise(
            regression_recipe(epsilon_M=epsilon_M),
            X_cont,
            target,
            seed=4,
            warmup=1,
        )

        assert_zero_alloc(_fit_iteration_, session)
