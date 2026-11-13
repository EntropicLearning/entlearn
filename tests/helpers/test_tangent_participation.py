"""Manifold reporting invariants through public fitted inspection."""

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET, RANK_DEFICIENT

from entlearn import Network
from entlearn.helpers.reporting import feature_importances

from ._fixtures import fitted_manifold_report

pytestmark = FIXED_BUDGET


@RANK_DEFICIENT
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

    def test_participation_weights_each_clusters_leverage_by_its_instance_weighted_occupancy(self):
        model = fitted_manifold_report()
        basis = model.inspect("manifold_projectors")["features"]
        gamma = model.inspect("training_affiliations")["features"]
        weights = model.inspect("training_instance_weights")["features"]
        expected = (
            sum(
                (weights * gamma[:, k]).sum() / weights.sum() * (basis[k] @ basis[k].T).diagonal()
                for k in range(basis.shape[0])
            )
            / basis.shape[2]
        )
        unweighted = (
            sum(
                gamma[:, k].mean() * (basis[k] @ basis[k].T).diagonal()
                for k in range(basis.shape[0])
            )
            / basis.shape[2]
        )
        # The learned instance weights are not uniform, so they change the result here.
        assert not torch.allclose(expected, unweighted)
        # The reference sums the same products in another order: a few units of rounding.
        tolerance = 16 * torch.finfo(DTYPE).eps
        torch.testing.assert_close(
            feature_importances(model), expected, rtol=tolerance, atol=tolerance
        )

    def test_missing_row_bound_values_are_not_replaced(self, tmp_path):
        path = tmp_path / "manifold.safetensors"
        fitted_manifold_report().save(path)  # without resumable=True: no training rows
        loaded = Network.load(path, device=DEVICE)
        with pytest.raises(ValueError, match=r"training_affiliations.*unavailable"):
            feature_importances(loaded)
