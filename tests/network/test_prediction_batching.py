"""Prediction preserves the fitted affiliation regime across query batches."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import two_row_input_model

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, PredictConfig, Recipe
from entlearn.primitives.normalise import _eps, floored_log_


class TestPredictionBatching:
    @pytest.mark.parametrize("weighted", (False, True))
    @pytest.mark.parametrize("temperature_scale", (0.5, 1.5))
    def test_input_prediction_is_independent_of_query_row_count(
        self, weighted: bool, temperature_scale: float
    ) -> None:
        sample_weights = torch.tensor([1.0, 2.0], dtype=DTYPE, device=DEVICE) if weighted else None
        network, query = two_row_input_model(
            temperature_scale * _eps(DTYPE), sample_weights=sample_weights
        )

        alone = network.predict(query)
        batched = network.predict(query.repeat(16, 1))

        assert torch.count_nonzero(alone).item() == (2 if temperature_scale > 1 else 1)
        torch.testing.assert_close(batched, alone.expand_as(batched))

    @pytest.mark.parametrize("coupling", (Coupling.M, Coupling.S))
    @pytest.mark.parametrize("temperature_scale", (0.5, 1.5))
    def test_hidden_propagation_keeps_its_fitted_regime(
        self, coupling: Coupling, temperature_scale: float
    ) -> None:
        epsilon = temperature_scale * _eps(DTYPE)
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        network = Network.fit(
            Recipe.chain(
                Input(K=2),
                Hidden(K=2, epsilon=epsilon),
                ClassificationHead(coupling=Coupling.M),
                coupling=coupling,
            ),
            X,
            target,
            max_iter=1,
            seed=2,
            predict_config=PredictConfig(output_mode="arithmetic"),
        )
        # A near-tied valid transition isolates the hidden gate from input assignment.
        with torch.inference_mode():
            network._graph.input.continuous_centroids.copy_(X)
            hidden = network._graph.blocks["hidden_1"]
            connection = hidden.incoming["input_to_hidden_1"]
            offset = _eps(DTYPE)
            connection.theta.copy_(
                torch.tensor(
                    [[0.5 + offset, 0.5 - offset], [0.5 - offset, 0.5 + offset]],
                    dtype=DTYPE,
                    device=DEVICE,
                )
            )
            floored_log_(connection.log_theta, connection.theta)
            network._graph.head.theta.copy_(torch.eye(2, dtype=DTYPE, device=DEVICE))
            floored_log_(network._graph.head.log_theta, network._graph.head.theta)

        alone = network.predict(X[:1])
        batched = network.predict(X[:1].repeat(16, 1))

        assert torch.count_nonzero(alone).item() == (2 if temperature_scale > 1 else 1)
        torch.testing.assert_close(batched, alone.expand_as(batched))

    def test_iterative_refinement_keeps_a_soft_regime_past_the_query_threshold(self) -> None:
        # Soft, since epsilon = 8 eps, while a 16-row query batch takes the refinement
        # temperature epsilon / 16 = eps / 2 below machine precision.
        epsilon = 8.0 * _eps(DTYPE)
        rows = 16
        assert epsilon / rows <= _eps(DTYPE)
        network, query = two_row_input_model(epsilon)
        config = PredictConfig(predict_mode="iterative", max_iter=2, tol=0)

        def refined(rows: int) -> torch.Tensor:
            result = network.predict_with_details(
                query.repeat(rows, 1), predict_config=config, details=("affiliations",)
            )
            assert result.affiliations is not None
            return result.affiliations["input"]

        alone = refined(1)
        batched = refined(rows)

        assert torch.count_nonzero(alone).item() == 2
        torch.testing.assert_close(batched, alone.expand_as(batched))
