"""M-connection initialisation and coordinate contracts."""

from __future__ import annotations

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    assert_deep_coordinates_non_increasing,
    connection_initialisation_data,
    deep_classification_recipe,
    deep_classification_session,
    hidden_connection,
    materialise,
)


class TestMConnectionCoordinates:
    @pytest.mark.parametrize("K_target", [2, 5], ids=["group", "expand"])
    def test_initial_prior_uses_the_sample_weight_count_scale(self, K_target: int) -> None:
        X_cont, y = connection_initialisation_data()
        recipe = deep_classification_recipe((3, K_target), epsilon=0.0, theta_alpha=2.0)

        unweighted = materialise(recipe, X_cont, y, seed=11)
        explicit_uniform = materialise(
            recipe,
            X_cont,
            y,
            sample_weights=torch.full((X_cont.shape[0],), 7.0, dtype=DTYPE, device=DEVICE),
            seed=11,
        )
        unweighted_state = hidden_connection(unweighted)
        explicit_state = hidden_connection(explicit_uniform)

        # The internal prior is rescaled with the counts; its ratio to total mass is fixed.
        assert unweighted_state.pseudocount == pytest.approx(1.0 / (K_target * 3))
        assert explicit_state.pseudocount == pytest.approx(1.0 / (K_target * 3))
        assert bool((unweighted_state.theta > 0).all())
        torch.testing.assert_close(unweighted_state.theta, explicit_state.theta)
        torch.testing.assert_close(
            unweighted_state.theta.sum(dim=0),
            torch.ones(3, dtype=DTYPE, device=DEVICE),
        )

    def test_grouping_uses_one_native_source_mass_per_column(self) -> None:
        X_cont, y = connection_initialisation_data()
        recipe = deep_classification_recipe((3, 2), epsilon=0.0, theta_alpha=2.0)
        session = materialise(recipe, X_cont, y, seed=5)
        state = hidden_connection(session)
        source_masses = session.graph.input.gamma.sum(dim=0)
        # Independent native-count calculation, not the implementation's rescaled prior.
        pseudocount = X_cont.shape[0] / (2 * 3)

        for column, mass in zip(state.theta.transpose(0, 1), source_masses, strict=True):
            denominator = mass + 2 * pseudocount
            expected = (
                torch.stack(((mass + pseudocount) / denominator, pseudocount / denominator))
                .sort()
                .values
            )
            torch.testing.assert_close(column.sort().values, expected)

    @pytest.mark.parametrize("K_target", [2, 5], ids=["group", "expand"])
    def test_theta_alpha_one_disables_the_initial_prior(self, K_target: int) -> None:
        X_cont, y = connection_initialisation_data()
        recipe = deep_classification_recipe((3, K_target), epsilon=0.0, theta_alpha=1.0)
        session = materialise(recipe, X_cont, y, seed=9)
        state = hidden_connection(session)

        assert state.pseudocount == 0.0
        assert bool((state.theta == 0).any())

    @pytest.mark.parametrize("widths", [(4, 3), (3, 5)], ids=["group", "expand"])
    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_each_deep_coordinate_is_non_increasing(
        self,
        widths: tuple[int, int],
        weighted: bool,
    ) -> None:
        session = deep_classification_session(widths, weighted=weighted)
        assert_deep_coordinates_non_increasing(session)
