"""Uniform and mixed S coupling through complete Network fits."""

from __future__ import annotations

import math

import pytest
import torch
from network._fixtures import BLOB_SPREADS, assert_simplex_rows, blobs, deep_classification_recipe

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, Recipe


class TestSCouplingFit:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("weight", [1.0, 7.0])
    def test_equal_weights_preserve_hard_ties_through_fitting(self, dtype, weight):
        X = torch.arange(6, dtype=dtype)[:, None] / 8
        y = torch.tensor([0, 0, 0, 1, 1, 1])
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
            sample_weights=torch.full((6,), weight, dtype=dtype),
        )

        torch.testing.assert_close(unweighted.predict(X), weighted.predict(X))
        assert unweighted.schema.K_active == weighted.schema.K_active
        torch.testing.assert_close(
            torch.tensor(unweighted.diagnostics.loss_history, dtype=dtype),
            torch.tensor(weighted.diagnostics.loss_history, dtype=dtype),
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
    def test_s_coupled_chain_fits_and_predicts(
        self,
        coupling: Coupling | tuple[Coupling, Coupling],
    ) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=7)
        recipe = deep_classification_recipe(
            (4, 5, 4), coupling=coupling, head_coupling=Coupling.S, theta_alpha=1.5
        )

        network = Network.fit(recipe, X_cont, y, max_iter=12, seed=5)
        prediction = network.predict(X_cont)

        assert_simplex_rows(prediction)
        assert all(math.isfinite(loss) for loss in network.diagnostics.loss_history)
