"""Prediction preserves the fitted affiliation regime across query batches."""

from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import skip_fit_iterations, two_row_input_model

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, PredictConfig, Recipe

pytestmark = FIXED_BUDGET

# Test both sides of machine precision, as well as the explicit zero-temperature regime.
EPS = torch.finfo(DTYPE).eps


class TestPredictionBatching:
    @pytest.mark.parametrize("weighted", (False, True))
    @pytest.mark.parametrize("temperature_scale", (0.0, 0.5, 1.5))
    def test_input_prediction_is_independent_of_query_row_count(
        self, weighted: bool, temperature_scale: float
    ) -> None:
        sample_weights = torch.tensor([1.0, 2.0], dtype=DTYPE, device=DEVICE) if weighted else None
        network, query = two_row_input_model(temperature_scale * EPS, sample_weights=sample_weights)

        alone = network.predict(query)
        batched = network.predict(query.repeat(16, 1))
        affiliations = network.predict_with_details(query, details=("affiliations",)).affiliations

        # The regime decides whether the midpoint's tied affiliations split. The read-out of a
        # one-hot affiliation keeps a positive probability, as small as tiny, for the other class.
        assert affiliations is not None
        (gamma,) = affiliations.values()
        assert torch.count_nonzero(gamma).item() == (2 if temperature_scale > 0 else 1)
        torch.testing.assert_close(batched, alone.expand_as(batched))

    @pytest.mark.parametrize("coupling", (Coupling.M, Coupling.S))
    @pytest.mark.parametrize("temperature_scale", (0.0, 0.5, 1.5))
    def test_hidden_propagation_keeps_its_fitted_regime(
        self, monkeypatch, coupling: Coupling, temperature_scale: float
    ) -> None:
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2),
            Hidden(K=2, epsilon=temperature_scale * EPS),
            ClassificationHead(coupling=Coupling.M),
            coupling=coupling,
        )
        config = PredictConfig(output_mode="arithmetic")
        captured = Network.fit(
            recipe, X, target, max_iter=1, seed=2, predict_config=config
        ).capture_current_state()
        # A near-tied valid transition isolates the hidden gate from input assignment.
        tie = torch.tensor(
            [[0.5 + EPS, 0.5 - EPS], [0.5 - EPS, 0.5 + EPS]], dtype=DTYPE, device=DEVICE
        )
        hidden, head = captured.parameters
        state = replace(
            captured,
            input_geometry=replace(captured.input_geometry, continuous_centroids=X.clone()),
            parameters=(
                replace(hidden, theta=tie),
                replace(head, theta=torch.eye(2, dtype=DTYPE, device=DEVICE)),
            ),
        )
        skip_fit_iterations(monkeypatch)
        network = Network.fit(
            recipe, X, target, initial_state=state, max_iter=1, predict_config=config
        )

        alone = network.predict(X[:1])
        batched = network.predict(X[:1].repeat(16, 1))

        assert torch.count_nonzero(alone).item() == (2 if temperature_scale > 0 else 1)
        torch.testing.assert_close(batched, alone.expand_as(batched))

    def test_iterative_refinement_keeps_a_soft_regime_past_the_query_threshold(self) -> None:
        # Soft, since epsilon = 8 eps, while a 16-row query batch takes the refinement
        # temperature epsilon / 16 = eps / 2 below machine precision.
        epsilon = 8.0 * EPS
        rows = 16
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
