"""Per-coordinate monotonicity of the fitted Network."""

from itertools import product

import pytest
import torch
from conftest import DEVICE, DTYPE, by_depth
from network._fixtures import (
    BLOB_SPREADS,
    assert_head_then_input_non_increasing,
    blobs,
    classification_recipe,
    materialise,
    noisy_regression_blobs,
    regression_recipe,
)

from entlearn import Input
from entlearn.network.blocks import input as inp

# Affiliation temperature x feature-weight temperature x instance-weight temperature.
# Infinity pins a weight vector, zero assigns it hard, and a positive value learns it soft.
_TEMPERATURES = tuple(product((0.0, 0.2), (float("inf"), 0.0, 0.2), (float("inf"), 0.0, 0.2)))
# Blob count and separation. An overlapping case puts rows near a decision boundary, where
# an assignment tie exposes a coordinate step that is monotone only on separated data.
_SHAPES = tuple(product((2, 3), BLOB_SPREADS))
_SEEDS = by_depth(minimal=(0,), standard=(0, 3), exhaustive=(0, 1, 2, 3, 4, 5))
# A warmed session starts near a fixed point, where round-off dominates the loss changes.
_WARMUPS = (0, 3)


def _temperature_id(case: tuple[float, float, float]) -> str:
    epsilon, epsilon_D, epsilon_T = case
    return f"g{epsilon}-wd{epsilon_D}-wt{epsilon_T}"


def _shape_id(case: tuple[int, str]) -> str:
    n_blobs, spread = case
    return f"{n_blobs}blob-{spread}"


def _row_weights(rows: int, seed: int) -> torch.Tensor:
    """Return normalised per-row weights drawn from ``seed``."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    weights = torch.rand(rows, dtype=DTYPE, device=DEVICE, generator=generator)
    return weights / weights.sum()


def _categorical_features(
    X_cont: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one hard-coded and one distribution categorical feature derived from the blobs."""
    codes = y.remainder(2)
    split = (X_cont[:, 0] > X_cont[:, 0].median()).long()
    return codes, torch.nn.functional.one_hot(split, 2).to(dtype=DTYPE)


class TestShallowClassificationMonotonicity:
    @pytest.mark.parametrize("temperatures", _TEMPERATURES, ids=_temperature_id)
    @pytest.mark.parametrize("shape", _SHAPES, ids=_shape_id)
    @pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
    @pytest.mark.parametrize("seed", _SEEDS, ids=lambda value: f"s{value}")
    @pytest.mark.parametrize("warmup", _WARMUPS, ids=lambda value: f"w{value}")
    def test_each_coordinate_update_is_non_increasing(
        self,
        temperatures: tuple[float, float, float],
        shape: tuple[int, str],
        weighted: bool,
        seed: int,
        warmup: int,
    ) -> None:
        epsilon, epsilon_D, epsilon_T = temperatures
        n_blobs, spread = shape
        X_cont, y = blobs(n_blobs, BLOB_SPREADS[spread], seed=seed)
        sample_weights = _row_weights(X_cont.shape[0], seed) if weighted else None
        recipe = classification_recipe(
            Input(K=n_blobs, epsilon=epsilon, epsilon_D=epsilon_D, epsilon_T=epsilon_T)
        )
        session = materialise(
            recipe, X_cont, y, sample_weights=sample_weights, seed=seed, warmup=warmup
        )
        assert_head_then_input_non_increasing(session)

    @pytest.mark.parametrize("temperatures", _TEMPERATURES, ids=_temperature_id)
    @pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
    @pytest.mark.parametrize("seed", _SEEDS, ids=lambda value: f"s{value}")
    def test_each_coordinate_update_is_non_increasing_on_mixed_features(
        self,
        temperatures: tuple[float, float, float],
        weighted: bool,
        seed: int,
    ) -> None:
        """Price the coordinate steps on data that carries both categorical representations."""
        epsilon, epsilon_D, epsilon_T = temperatures
        X_cont, y = blobs(3, BLOB_SPREADS["overlapping"], seed=seed)
        X_cat = _categorical_features(X_cont, y)
        sample_weights = _row_weights(X_cont.shape[0], seed) if weighted else None
        recipe = classification_recipe(
            Input(K=3, epsilon=epsilon, epsilon_D=epsilon_D, epsilon_T=epsilon_T)
        )
        session = materialise(
            recipe, X_cont, y, X_cat=X_cat, sample_weights=sample_weights, seed=seed
        )
        assert_head_then_input_non_increasing(session)


class TestShallowRegressionMonotonicity:
    @pytest.mark.parametrize(
        "epsilon_M",
        (float("inf"), 0.0, 0.2),
        ids=("fixed", "learned-hard", "learned-soft"),
    )
    @pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
    def test_each_coordinate_update_is_non_increasing(
        self,
        epsilon_M: float,
        weighted: bool,
    ) -> None:
        X_cont, target = noisy_regression_blobs()
        sample_weights = None
        if weighted:
            sample_weights = torch.arange(
                1,
                X_cont.shape[0] + 1,
                dtype=DTYPE,
                device=DEVICE,
            )
        recipe = regression_recipe(
            Input(K=2, epsilon=0.2, epsilon_D=0.2, epsilon_T=0.2),
            epsilon_M=epsilon_M,
        )
        session = materialise(
            recipe,
            X_cont,
            target,
            sample_weights=sample_weights,
            seed=4,
        )
        assert_head_then_input_non_increasing(session)


class TestCoordinateStagingContract:
    def test_feature_weight_update_preserves_the_staged_cluster_masses(self) -> None:
        X_cont, y = blobs(2, BLOB_SPREADS["separated"], seed=1)
        recipe = classification_recipe(Input(K=2, epsilon=0.1, epsilon_D=0.2, epsilon_T=0.2))
        session = materialise(recipe, X_cont, y, seed=1)
        block = session.graph.input

        inp.stage_statistics_(block, session)
        masses = session.workspace.scratch_K[: block.K].clone()
        inp.update_feature_weights_(block, session)

        assert torch.equal(session.workspace.scratch_K[: block.K], masses)
