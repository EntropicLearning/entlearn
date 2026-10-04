"""Manifold reporting invariants through public fitted inspection."""

import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import Network
from entlearn.helpers.reporting import feature_importances

from ._fixtures import fitted_manifold_report


class TestTangentParticipation:
    def test_basis_rotation_and_weight_rescaling_leave_participation_unchanged(self, monkeypatch):
        model = fitted_manifold_report()
        expected = feature_importances(model)
        # Participation is a simplex: the diagonal tangent leverage sums to the
        # subspace dimension, which the reported value divides out.
        torch.testing.assert_close(expected.sum(), torch.ones((), dtype=DTYPE, device=DEVICE))
        inspect = Network.inspect
        rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=DTYPE, device=DEVICE)

        def equivalent(network, name):
            values = inspect(network, name)
            if name == "manifold_projectors":
                return {owner: basis @ rotation for owner, basis in values.items()}
            if name == "training_instance_weights":
                return {owner: weights * 3 for owner, weights in values.items()}
            return values

        monkeypatch.setattr(Network, "inspect", equivalent)
        actual = feature_importances(model)
        torch.testing.assert_close(actual, expected)
        assert actual.dtype is DTYPE
        assert actual.device == DEVICE
        with torch.inference_mode():
            actual.zero_()
        torch.testing.assert_close(feature_importances(model), expected)

    @pytest.mark.parametrize("missing", ["training_affiliations", "training_instance_weights"])
    def test_missing_row_bound_values_are_not_replaced(self, monkeypatch, missing):
        model = fitted_manifold_report()
        inspect = Network.inspect

        def incomplete(network, name):
            if name == missing:
                raise ValueError("training values unavailable")
            return inspect(network, name)

        monkeypatch.setattr(Network, "inspect", incomplete)
        with pytest.raises(ValueError, match="training values unavailable"):
            feature_importances(model)
