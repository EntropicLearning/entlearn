"""Connection updates and objective scaling against independent native-count formulae."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    connection_initialisation_data,
    deep_classification_recipe,
    hidden_connection,
    materialise,
)

from entlearn import Coupling
from entlearn.primitives.normalise import _eps


class TestConnectionCountScale:
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("theta_alpha", [1.0, 1.7])
    def test_update_and_loss_preserve_native_count_prior_ratio(
        self, coupling, weighted, theta_alpha
    ):
        X, y = connection_initialisation_data()
        raw_weights = torch.arange(1, X.shape[0] + 1, dtype=DTYPE, device=DEVICE)
        session = materialise(
            deep_classification_recipe(
                (3, 2), coupling=coupling, epsilon=0.2, theta_alpha=theta_alpha
            ),
            X,
            y,
            sample_weights=raw_weights if weighted else None,
            seed=11,
        )
        connection = hidden_connection(session)
        source = session.graph.input.gamma
        target = session.graph.blocks["hidden_1"].gamma
        weights = raw_weights / raw_weights.sum() if weighted else torch.ones_like(raw_weights)
        native_counts = target.T @ (weights[:, None] * source)
        native_prior = (theta_alpha - 1) * (1.0 if weighted else X.shape[0]) / (2 * 3)
        counts_with_prior = native_counts + native_prior
        axis = 0 if coupling is Coupling.M else 1
        expected = counts_with_prior / counts_with_prior.sum(dim=axis, keepdim=True)

        connection.update_parameters_(target, source, session)

        torch.testing.assert_close(connection.theta, expected)
        logs = expected.clamp_min(_eps(DTYPE)).log()
        delta = connection.description.delta / (1.0 if weighted else X.shape[0])
        expected_loss = -delta * ((native_counts * logs).sum() + native_prior * logs.sum())
        session.workspace.loss.zero_()
        connection.partial_loss_(target, source, session)
        torch.testing.assert_close(session.workspace.loss.squeeze(), expected_loss)
