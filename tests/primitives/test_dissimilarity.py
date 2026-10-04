"""Tests for feature dissimilarity primitives."""

import pytest
import torch
from _alloc import assert_no_float64
from conftest import DEVICE, DTYPE

from entlearn.primitives import dissimilarity
from entlearn.primitives.dissimilarity import feature_d2_to_point, seed_dissimilarity
from primitives._fixtures import assert_close, generator, rand, stochastic


class TestFeatureDissimilarity:
    @pytest.mark.parametrize(
        ("point", "distribution", "expected"),
        [
            ((0.0, 1.0), (0.5, 0.5), (1.3862943611198906, 5.386294361119891)),
            ((1.0, 1.0), (0.25, 0.75), (3.022588722239781, 4.374670289237616)),
        ],
    )
    def test_point_combines_continuous_and_categorical_costs(
        self,
        point: tuple[float, float],
        distribution: tuple[float, float],
        expected: tuple[float, float],
    ) -> None:
        continuous = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=DTYPE, device=DEVICE)
        categorical = [torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=DTYPE, device=DEVICE)]

        result = feature_d2_to_point(
            continuous,
            categorical,
            [torch.tensor(distribution, dtype=DTYPE, device=DEVICE).log()],
            torch.tensor([0.25, 0.75], dtype=DTYPE, device=DEVICE),
            torch.tensor([2.0], dtype=DTYPE, device=DEVICE),
            torch.tensor(point, dtype=DTYPE, device=DEVICE),
        )

        assert_close(result, torch.tensor(expected, dtype=DTYPE))

    def test_seed_uses_the_selected_instance_as_the_point(self) -> None:
        continuous = torch.tensor([[0.0], [2.0]], dtype=DTYPE, device=DEVICE)
        categorical = [torch.full((2, 2), 0.5, dtype=DTYPE, device=DEVICE)]
        categorical_logs = [categorical[0].log()]

        result = seed_dissimilarity(
            continuous,
            categorical,
            categorical_logs,
            torch.ones(1, dtype=DTYPE, device=DEVICE),
            torch.zeros(1, dtype=DTYPE, device=DEVICE),
            0,
        )

        assert_close(result, torch.tensor([0.0, 4.0], dtype=DTYPE))

    @pytest.mark.parametrize("block_bytes", [1, 64, 4096, 1 << 24])
    @pytest.mark.parametrize("continuous_width", [0, 1, 5])
    def test_point_result_is_bitwise_independent_of_row_blocks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        block_bytes: int,
        continuous_width: int,
    ) -> None:
        g = generator(4)
        continuous = rand(37, continuous_width, g=g)
        categorical = [stochastic(37, 3, dim=1, g=g)]
        point_logs = [categorical[0][3].clamp_min(torch.finfo(DTYPE).eps).log()]
        continuous_weights = rand(continuous_width, g=g)
        categorical_weights = rand(1, g=g)
        point = continuous[3].clone() if continuous_width else continuous.new_zeros(0)
        difference = continuous - point
        expected = (continuous_weights * difference * difference).sum(dim=1)
        expected += categorical_weights[0] * -(categorical[0] * point_logs[0]).sum(dim=1)

        monkeypatch.setattr(dissimilarity, "_D2_BLOCK_BYTES", block_bytes)
        monkeypatch.setattr(dissimilarity, "_MIN_D2_BLOCKS", 1)
        result = feature_d2_to_point(
            continuous,
            categorical,
            point_logs,
            continuous_weights,
            categorical_weights,
            point,
        )

        assert torch.equal(result, expected)

    def test_float32_inputs_stay_float32_and_read_only(self) -> None:
        continuous = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
        continuous_before = continuous.clone()
        categorical = [torch.full((2, 2), 0.5, dtype=torch.float32)]
        categorical_before = categorical[0].clone()

        result = assert_no_float64(
            feature_d2_to_point,
            continuous,
            categorical,
            [categorical[0][0].log()],
            torch.ones(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            continuous[0],
        )

        assert result.dtype is torch.float32
        assert torch.equal(continuous, continuous_before)
        assert torch.equal(categorical[0], categorical_before)
