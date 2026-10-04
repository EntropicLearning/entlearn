"""Data-supported manifold directions and deterministic orthogonal completion."""

import warnings

import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import ManifoldInput, Network, Recipe, RegressionHead


def initialise_basis(X, centres, d, *, seed=13):
    recipe = Recipe.chain(ManifoldInput(K=len(centres), subspace_dimension=d), RegressionHead())
    return Network.initialise(
        recipe,
        X,
        X[:, 0],
        continuous_centroids=centres,
        seed=seed,
    ).input_geometry.manifold_projectors


class TestManifoldInitialDirections:
    def test_completion_handles_a_supported_direction_nearly_aligned_with_an_axis(self):
        direction = torch.tensor([1.0, 1e-3, 0.0, 0.0], dtype=DTYPE, device=DEVICE)
        direction /= torch.linalg.vector_norm(direction)
        X = torch.arange(-3, 4, dtype=DTYPE, device=DEVICE)[:, None] * direction
        centres = torch.zeros(1, 4, dtype=DTYPE, device=DEVICE)
        with pytest.warns(UserWarning, match="rank=1"):
            basis = initialise_basis(X, centres, 4)[0]
        torch.testing.assert_close(
            basis[:, :1] @ basis[:, :1].T, direction[:, None] @ direction[None, :]
        )
        torch.testing.assert_close(basis.T @ basis, torch.eye(4, dtype=DTYPE, device=DEVICE))

    def test_weak_but_resolvable_direction_is_not_lost_to_covariance_roundoff(self):
        generator = torch.Generator().manual_seed(23)
        directions = torch.linalg.qr(torch.randn(5, 2, generator=generator, dtype=DTYPE)).Q.to(
            DEVICE
        )
        scale = 1e-4 if DTYPE is torch.float32 else 1e-9
        coefficients = torch.tensor(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, -scale], [0.0, scale]],
            dtype=DTYPE,
            device=DEVICE,
        ).repeat(2, 1)
        X = coefficients @ directions.T
        centres = torch.zeros(1, 5, dtype=DTYPE, device=DEVICE)
        assert torch.linalg.matrix_rank(X) == 2
        with pytest.warns(UserWarning, match="rank"):
            basis = initialise_basis(X, centres, 3)[0]
        torch.testing.assert_close(
            basis[:, :2] @ basis[:, :2].T,
            directions @ directions.T,
            rtol=0,
            atol=16 * torch.finfo(DTYPE).eps,
        )

    def test_empty_cluster_has_a_finite_basis_and_warning(self):
        X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE)
        centres = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE)
        with pytest.warns(UserWarning, match=r"1 \(rank=0\)"):
            basis = initialise_basis(X, centres, 2)
        torch.testing.assert_close(
            basis.transpose(1, 2) @ basis,
            torch.eye(2, dtype=DTYPE, device=DEVICE).expand(2, -1, -1),
        )

    def test_tied_supported_directions_are_reused_in_the_same_order(self):
        X = torch.cat(
            (
                torch.eye(4, dtype=DTYPE, device=DEVICE)[:2],
                -torch.eye(4, dtype=DTYPE, device=DEVICE)[:2],
            )
        )
        centres = torch.zeros(1, 4, dtype=DTYPE, device=DEVICE)
        with pytest.warns(UserWarning, match="rank"):
            wider = initialise_basis(X, centres, 3)
        narrow = initialise_basis(X, centres, 1)
        torch.testing.assert_close(narrow, wider[:, :, :1], rtol=0, atol=0)

    def test_rank_deficiency_keeps_supported_directions_before_completion(self):
        X = torch.zeros(8, 5, dtype=DTYPE, device=DEVICE)
        X[:4, 0] = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=DTYPE, device=DEVICE)
        X[4:, 1] = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)
        centres = torch.zeros(1, 5, dtype=DTYPE, device=DEVICE)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            basis = initialise_basis(X, centres, 4)[0]
        supported = basis[:, :2] @ basis[:, :2].T
        expected = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=DTYPE, device=DEVICE))
        torch.testing.assert_close(supported, expected)
        torch.testing.assert_close(basis.T @ basis, torch.eye(4, dtype=DTYPE, device=DEVICE))
        assert any("rank" in str(item.message) for item in caught)
        assert basis.dtype == DTYPE

    @pytest.mark.parametrize("rows,features,rank", [(12, 5, 2), (3, 8, 2), (4, 5, 0)])
    def test_requested_dimension_does_not_change_leading_directions(self, rows, features, rank):
        generator = torch.Generator().manual_seed(29)
        orientation = torch.linalg.qr(
            torch.randn(features, features, generator=generator, dtype=DTYPE)
        ).Q.to(DEVICE)
        coefficients = torch.randn(rows, rank, generator=generator, dtype=DTYPE).to(DEVICE)
        X = coefficients @ orientation[:, :rank].T
        centres = torch.zeros(1, features, dtype=DTYPE, device=DEVICE)
        with pytest.warns(UserWarning, match="rank"):
            full = initialise_basis(X, centres, features)
        for dimension in range(1, features):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                smaller = initialise_basis(X, centres, dimension)
            torch.testing.assert_close(smaller, full[:, :, :dimension], rtol=0, atol=0)
            assert bool(caught) == (dimension > rank)
        torch.testing.assert_close(
            full.transpose(1, 2) @ full,
            torch.eye(features, dtype=DTYPE, device=DEVICE).unsqueeze(0),
        )

    def test_completion_does_not_depend_on_seed_or_other_cluster_draws(self):
        X = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            dtype=DTYPE,
            device=DEVICE,
        )
        centres = X[[0, 2]]
        with pytest.warns(UserWarning, match="rank"):
            first = initialise_basis(X, centres, 3, seed=7)
        with pytest.warns(UserWarning, match="rank"):
            replay = initialise_basis(X, centres, 3, seed=101)
        torch.testing.assert_close(first, replay, rtol=0, atol=0)
