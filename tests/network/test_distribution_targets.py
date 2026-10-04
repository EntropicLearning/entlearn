"""Distribution-valued classification targets through Network."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import assert_simplex_rows, classification_recipe

from entlearn import Input, Network


class TestDistributionTargets:
    def test_class_weights_multiply_soft_target_components(self) -> None:
        X_cont = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([[0.6, 0.4], [0.9, 0.1]], dtype=X_cont.dtype, device=X_cont.device)
        sample_weights = torch.tensor([1.0, 3.0], dtype=X_cont.dtype, device=X_cont.device)
        class_weights = torch.tensor([1.0, 2.0], dtype=X_cont.dtype, device=X_cont.device)
        recipe = classification_recipe(Input(K=1))

        network = Network.fit(
            recipe,
            X_cont,
            target,
            sample_weights=sample_weights,
            class_weights=class_weights,
            max_iter=1,
            seed=2,
        )
        probabilities = network.predict(X_cont)

        expected_class_mass = (sample_weights[:, None] * target * class_weights[None, :]).sum(dim=0)
        expected_distribution = expected_class_mass / expected_class_mass.sum()
        torch.testing.assert_close(probabilities, expected_distribution.expand_as(probabilities))

    def test_hard_and_distribution_targets_share_unlabelled_row_meaning(self) -> None:
        X_cont = torch.tensor([[0.0], [0.1], [0.5], [0.9], [1.0]], dtype=DTYPE, device=DEVICE)
        hard = torch.tensor([0, 0, -1, 1, 1], dtype=torch.int64, device=X_cont.device)
        distribution = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=X_cont.dtype,
            device=X_cont.device,
        )
        recipe = classification_recipe(Input(K=2, epsilon=0.05))

        hard_network = Network.fit(recipe, X_cont, hard, max_iter=20, seed=5)
        distribution_network = Network.fit(recipe, X_cont, distribution, max_iter=20, seed=5)

        torch.testing.assert_close(
            distribution_network.predict(X_cont),
            hard_network.predict(X_cont),
        )

    @pytest.mark.parametrize(
        "target_values",
        (
            pytest.param((0, 2), id="middle-class-unobserved"),
            pytest.param((0, 1, 2), id="all-classes-observed"),
        ),
    )
    def test_declared_class_width_preserves_output_columns(
        self,
        target_values: tuple[int, ...],
    ) -> None:
        X_cont = torch.arange(len(target_values), dtype=DTYPE, device=DEVICE).unsqueeze(1)
        target = torch.tensor(target_values, dtype=torch.int64, device=X_cont.device)
        recipe = classification_recipe(Input(K=1), n_classes=3)

        network = Network.fit(recipe, X_cont, target, max_iter=1, seed=2)
        probabilities = network.predict(X_cont)

        expected_class_mass = torch.bincount(target, minlength=3).to(dtype=X_cont.dtype)
        expected_distribution = expected_class_mass / expected_class_mass.sum()
        assert network.schema.M == 3
        assert_simplex_rows(probabilities)
        torch.testing.assert_close(probabilities, expected_distribution.expand_as(probabilities))

    @pytest.mark.parametrize(
        "weight_values",
        (
            pytest.param((3.0, 1.0, 1.0), id="upweight-class-zero"),
            pytest.param((1.0, 1.0, 3.0), id="upweight-class-one"),
        ),
    )
    def test_sample_weights_change_the_fitted_class_distribution(
        self,
        weight_values: tuple[float, ...],
    ) -> None:
        X_cont = torch.tensor([[0.0], [0.5], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([0, 1, 1], dtype=torch.int64, device=X_cont.device)
        sample_weights = torch.tensor(weight_values, dtype=X_cont.dtype, device=X_cont.device)
        recipe = classification_recipe(K=1)
        initial_state = Network.initialise(recipe, X_cont, target, seed=9)

        unweighted_network = Network.fit(
            recipe,
            X_cont,
            target,
            initial_state=initial_state,
            max_iter=1,
        )
        weighted_network = Network.fit(
            recipe,
            X_cont,
            target,
            sample_weights=sample_weights,
            initial_state=initial_state,
            max_iter=1,
        )

        unweighted_probabilities = unweighted_network.predict(X_cont)
        weighted_probabilities = weighted_network.predict(X_cont)
        hard_targets = torch.nn.functional.one_hot(target, num_classes=2).to(dtype=X_cont.dtype)
        expected_class_mass = (sample_weights[:, None] * hard_targets).sum(dim=0)
        expected_distribution = expected_class_mass / expected_class_mass.sum()

        assert not torch.allclose(weighted_probabilities, unweighted_probabilities)
        torch.testing.assert_close(
            weighted_probabilities,
            expected_distribution.expand_as(weighted_probabilities),
        )
