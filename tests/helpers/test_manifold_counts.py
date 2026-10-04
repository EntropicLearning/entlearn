"""Stored entries and basis-invariant manifold geometry have different counts."""

import pytest

from entlearn.helpers.reporting import count_parameters

from ._fixtures import manifold_count_model


class TestManifoldCounts:
    @pytest.mark.parametrize("d", [1, 2, 3])
    @pytest.mark.parametrize("alpha", [0.0, 0.1, 1e-12])
    def test_geometric_count_deducts_basis_rotations_and_affine_shift(self, d, alpha):
        model = manifold_count_model(d, alpha)
        # One three-dimensional centroid and a 3xd basis, plus two output centroids.
        # Fixed output and instance weights are excluded in both modes.
        assert count_parameters(model, raw=True) == 3 + 3 * d + 2
        centre = 3 if alpha > 0 else 3 - d
        assert count_parameters(model) == centre + d * (3 - d) + 2
        # Four one-cluster affiliations add stored entries but no freedom.
        assert count_parameters(model, raw=True, include_affiliations=True) == 3 + 3 * d + 6
        assert count_parameters(model, include_affiliations=True) == centre + d * (3 - d) + 2

    @pytest.mark.parametrize("alpha", [0.0, 0.1])
    def test_pruned_geometry_counts_active_not_declared_width(self, alpha):
        model = manifold_count_model(1, alpha, prune=True)
        assert model.recipe.blocks[0].K == 3
        assert dict(model.schema.K_active)["input"] == 1
        assert count_parameters(model, raw=True) == 8
        assert count_parameters(model) == (6 if alpha == 0 else 7)
