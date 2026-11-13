"""Regression target weights are reported independently of training-row retention."""

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import prediction_model

from entlearn import Network
from entlearn.helpers import reporting

from ._fixtures import reporting_model

pytestmark = FIXED_BUDGET


class TestTargetWeights:
    @pytest.mark.parametrize(
        ("epsilon_M", "fixed"),
        [
            pytest.param(float("inf"), (2.0, 1.0), id="supplied"),
            pytest.param(0.2, None, id="learned"),
            pytest.param(0.0, None, id="learned-hard"),
        ],
    )
    def test_supplied_and_learned_weights_are_the_fitted_output_weights(self, epsilon_M, fixed):
        model = reporting_model(epsilon_M=epsilon_M, W_M=fixed)
        fitted = model.inspect("head_parameters")["output"]["W_M"].clone()
        result = reporting.target_weights(model)
        assert torch.equal(result, fitted)
        with torch.inference_mode():
            result.zero_()  # the report is a copy, not the fitted state
        assert torch.equal(reporting.target_weights(model), fitted)

    def test_implicit_weights_are_uniform(self):
        result = reporting.target_weights(reporting_model())
        assert torch.equal(result, torch.tensor([0.5, 0.5], dtype=DTYPE, device=DEVICE))

    @pytest.mark.parametrize("epsilon_M", [float("inf"), 0.2], ids=["implicit", "learned"])
    def test_no_training_values_are_needed(self, tmp_path, epsilon_M):
        model = reporting_model(epsilon_M=epsilon_M)
        path = tmp_path / "model.safetensors"
        model.save(path)  # without resumable=True: no training rows
        loaded = Network.load(path, device=DEVICE)
        assert torch.equal(reporting.target_weights(loaded), reporting.target_weights(model))

    def test_classification_and_unfitted_values_are_rejected(self):
        model, _, _ = prediction_model()
        with pytest.raises(ValueError, match="regression"):
            reporting.target_weights(model)
        with pytest.raises(ValueError, match="fitted Network"):
            reporting.target_weights(object())
