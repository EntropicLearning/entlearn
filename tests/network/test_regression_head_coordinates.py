"""Coordinate behaviour of the regression head inside a fit session."""

import torch
from conftest import DEVICE, DTYPE
from network._fixtures import materialise, noisy_regression_blobs, regression_recipe

from entlearn import Input
from entlearn.network.blocks import regression as reg
from entlearn.network.data import _RegressionSupervision


def _expected_output_weights(
    target: torch.Tensor,
    centroids: torch.Tensor,
    gamma: torch.Tensor,
    row_weights: torch.Tensor,
    *,
    delta: float,
    epsilon_M: float,
) -> torch.Tensor:
    """Return the analytical output-weight coordinate minimiser."""
    residual_squared = (target[:, :, None] - centroids[None, :, :]).square()
    costs = delta * (row_weights[:, None, None] * gamma[:, None, :] * residual_squared).sum(
        dim=(0, 2)
    )
    return torch.softmax(-costs / epsilon_M, dim=0)


class TestRegressionHeadCoordinates:
    def test_seed_and_empty_clusters_use_weighted_mean(self) -> None:
        X_cont = torch.arange(4, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.tensor(
            [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [float("nan"), float("nan")]],
            dtype=DTYPE,
            device=DEVICE,
        )
        sample_weights = torch.tensor([1.0, 8.0, 2.0, 5.0], dtype=DTYPE, device=DEVICE)
        # The plain labelled mean is (3, 20); the weights pull the seed away from it.
        expected_weighted_mean = torch.tensor(
            [35.0 / 11.0, 230.0 / 11.0], dtype=DTYPE, device=DEVICE
        )

        session = materialise(
            regression_recipe(K=2, epsilon_M=0.2),
            X_cont,
            target,
            sample_weights=sample_weights,
            seed=2,
        )

        torch.testing.assert_close(
            session.graph.head.C_y, expected_weighted_mean.unsqueeze(1).expand(2, 2)
        )
        torch.testing.assert_close(
            session.graph.head.W_M,
            torch.full((2,), 0.5, dtype=DTYPE, device=DEVICE),
        )

        head = session.graph.head
        gamma = session.graph.source_affiliations(head)
        gamma.zero_()
        gamma[:, 0] = 1.0
        reg.update_centroids_(head, session)
        torch.testing.assert_close(head.C_y[:, 1], expected_weighted_mean)
        torch.testing.assert_close(head.live_cache.weighted_target_mean, expected_weighted_mean)

    def test_centroids_update_before_learnable_output_weights(self) -> None:
        X_cont, target = noisy_regression_blobs()
        epsilon_M = 0.2
        session = materialise(
            regression_recipe(Input(K=2, epsilon=0.0), epsilon_M=epsilon_M),
            X_cont,
            target,
            seed=4,
        )
        head = session.graph.head
        gamma = session.graph.source_affiliations(head)
        supervision = session.data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        row_weights = supervision.row_weights
        weighted_gamma = row_weights[:, None] * gamma
        expected_centroids = target.transpose(0, 1) @ weighted_gamma
        expected_centroids /= weighted_gamma.sum(dim=0).unsqueeze(0)
        edge = session.graph.connection_into(head)
        expected_weights = _expected_output_weights(
            target,
            expected_centroids,
            gamma,
            row_weights,
            delta=edge.delta,
            epsilon_M=epsilon_M,
        )
        stale_weights = _expected_output_weights(
            target,
            head.C_y,
            gamma,
            row_weights,
            delta=edge.delta,
            epsilon_M=epsilon_M,
        )

        head.update_parameters_(session)

        torch.testing.assert_close(head.C_y, expected_centroids)
        torch.testing.assert_close(head.W_M, expected_weights)
        assert not torch.allclose(expected_weights, stale_weights)
