"""Uniform and mixed S coupling through complete Network fits."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import (
    BLOB_SPREADS,
    assert_non_increasing,
    blobs,
    deep_classification_recipe,
)

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, Recipe

pytestmark = FIXED_BUDGET


class TestSCouplingFit:
    @pytest.mark.parametrize("weight", [1.0, 7.0])
    def test_equal_weights_preserve_hard_ties_through_fitting(self, weight):
        X = torch.arange(6, dtype=DTYPE, device=DEVICE)[:, None] / 8
        y = torch.tensor([0, 0, 0, 1, 1, 1], device=DEVICE)
        recipe = Recipe.chain(
            Input(K=3, epsilon=0),
            Hidden(K=2, epsilon=0),
            ClassificationHead(coupling=Coupling.S),
            coupling=Coupling.S,
            theta_alpha=1.1,
        )
        state = Network.initialise(recipe, X, y, seed=1)
        unweighted = Network.fit(recipe, X, y, initial_state=state, max_iter=1)
        weighted = Network.fit(
            recipe,
            X,
            y,
            initial_state=state,
            max_iter=1,
            sample_weights=torch.full((6,), weight, dtype=DTYPE, device=DEVICE),
        )

        torch.testing.assert_close(unweighted.predict(X), weighted.predict(X))
        assert unweighted.schema.K_active == weighted.schema.K_active
        torch.testing.assert_close(
            torch.tensor(unweighted.diagnostics.loss_history, dtype=DTYPE),
            torch.tensor(weighted.diagnostics.loss_history, dtype=DTYPE),
        )

    @pytest.mark.parametrize(
        "coupling",
        [
            Coupling.S,
            (Coupling.S, Coupling.M),
            (Coupling.M, Coupling.S),
        ],
        ids=["all-s", "s-then-m", "m-then-s"],
    )
    def test_s_coupled_chain_separates_blobs_under_the_documented_constraints(
        self,
        coupling: Coupling | tuple[Coupling, Coupling],
    ) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=7)
        recipe = deep_classification_recipe(
            (4, 5, 4), coupling=coupling, head_coupling=Coupling.S, epsilon=0.05, theta_alpha=1.5
        )

        network = Network.fit(recipe, X_cont, y, max_iter=12, seed=5)

        assert torch.equal(network.predict(X_cont).argmax(dim=1), y)
        for previous, current in pairwise(network.diagnostics.loss_history):
            assert_non_increasing(previous, current, "iteration")
        # representation.md: S rows and M columns of a transition sum to one.
        transitions = network.inspect("transition_matrices")
        for connection in recipe.connections[:-1]:
            sums = transitions[connection.name].sum(
                dim=1 if connection.coupling is Coupling.S else 0
            )
            torch.testing.assert_close(sums, torch.ones_like(sums))
