"""Regression target weights are reported independently of training-row retention."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import prediction_model

from entlearn import Network
from entlearn.helpers import reporting

from ._fixtures import reporting_model


class TestTargetWeights:
    @pytest.mark.parametrize(
        ("epsilon_M", "fixed"),
        [(float("inf"), None), (float("inf"), (2.0, 1.0)), (0.2, None), (0.0, None)],
    )
    def test_learned_supplied_and_implicit_weights_are_detached(self, epsilon_M, fixed):
        model = reporting_model(epsilon_M=epsilon_M, W_M=fixed)
        result = reporting.target_weights(model)
        parameters = model.inspect("head_parameters")["output"]
        expected = parameters.get("W_M", torch.full((2,), 0.5, dtype=DTYPE, device=DEVICE))
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.sum(), torch.ones((), dtype=DTYPE, device=DEVICE))
        assert not result.requires_grad
        with torch.inference_mode():
            result.zero_()
        torch.testing.assert_close(reporting.target_weights(model), expected)
        if fixed is not None:
            torch.testing.assert_close(
                expected, torch.tensor([2 / 3, 1 / 3], dtype=DTYPE, device=DEVICE)
            )

    @pytest.mark.parametrize("epsilon_M", [float("inf"), 0.2])
    def test_no_training_values_are_needed(self, monkeypatch, epsilon_M):
        model = reporting_model(epsilon_M=epsilon_M)
        inspect = Network.inspect

        def fitted_parameters_only(network, name):
            if name.startswith("training_"):
                raise ValueError("row-bound state unavailable")
            return inspect(network, name)

        monkeypatch.setattr(Network, "inspect", fitted_parameters_only)
        assert reporting.target_weights(model).shape == (2,)

    def test_classification_and_unfitted_values_are_rejected(self):
        model, _, _ = prediction_model()
        with pytest.raises(ValueError, match="regression"):
            reporting.target_weights(model)
        with pytest.raises(ValueError, match="fitted Network"):
            reporting.target_weights(object())
