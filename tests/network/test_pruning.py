"""Pruning and denominator safety exercised through Network fitting."""

from __future__ import annotations

import pytest
import torch
from conftest import DEVICE, FIXED_BUDGET
from network._fixtures import deep_classification_recipe, deep_regression_recipe

from entlearn import (
    Coupling,
    Hidden,
    Input,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.network import fit as fit_ops
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.primitives.normalise import _tiny

pytestmark = FIXED_BUDGET


class TestPruning:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("name", ["input", "hidden_1", "hidden_2"])
    def test_one_graph_prune_uses_tiny_and_compacts_every_owner(
        self, monkeypatch, dtype, coupling, task, name
    ):
        X = torch.linspace(0, 1, 8, dtype=dtype, device=DEVICE).unsqueeze(1)
        codes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=DEVICE)
        y = codes if task == "classification" else X.square()
        weights = torch.arange(1, 9, dtype=dtype, device=DEVICE)
        recipe = (
            deep_classification_recipe((4, 4, 4), coupling=coupling, head_coupling=coupling)
            if task == "classification"
            else deep_regression_recipe((4, 4, 4), coupling=coupling)
        )
        materialise = fit_ops._materialise
        observed = []

        def prune_boundary(*args, **kwargs):
            session = materialise(*args, **kwargs)
            graph = session.graph
            block = graph.blocks[name]
            eps, tiny = torch.finfo(dtype).eps, _tiny(dtype)
            # An InitialState holds no affiliations, so the boundary masses are written here.
            # Zero and subnormal masses are removed; the below-eps normal mass survives.
            block.gamma[:, 1:] = (
                torch.tensor([0.0, tiny / 16, eps / 2], dtype=dtype, device=DEVICE) / 8
            )
            block.gamma[:, 0] = 1 - block.gamma[:, 1:].sum(dim=1)
            expected_gamma = block.gamma[:, [0, 3]].clone()
            graph.prune_(block, session.workspace)
            torch.testing.assert_close(block.gamma, expected_gamma, rtol=0, atol=0)
            assert block.K == 2
            for connection in graph.connections:
                target = graph.blocks[connection.target]
                source_width = graph.affiliations(connection.source).shape[1]
                if isinstance(target, _HiddenBlock):
                    state = target.incoming[connection.name]
                    assert state.theta.shape == state.log_theta.shape == (target.K, source_width)
                    axis = 0 if connection.coupling is Coupling.M else 1
                    torch.testing.assert_close(
                        state.theta.sum(dim=axis),
                        torch.ones(state.theta.shape[1 - axis], dtype=dtype, device=DEVICE),
                    )
                    torch.testing.assert_close(state.log_theta, state.theta.clamp_min(tiny).log())
                elif task == "classification":
                    assert target.theta.shape == target.log_theta.shape == (2, source_width)
                    torch.testing.assert_close(target.log_theta, target.theta.clamp_min(tiny).log())
                    axis = 0 if coupling is Coupling.M else 1
                    torch.testing.assert_close(
                        target.theta.sum(dim=axis),
                        torch.ones(target.theta.shape[1 - axis], dtype=dtype, device=DEVICE),
                    )
                else:
                    assert target.C_y.shape == (1, source_width)
            observed.append(name)
            return session

        monkeypatch.setattr(fit_ops, "_materialise", prune_boundary)
        network = Network.fit(recipe, X, y, X_cat=(codes,), sample_weights=weights, max_iter=1)
        assert observed == [name]
        # The following real iteration may legitimately prune another cluster.
        active = dict(network.schema.K_active)
        assert active[name] <= 2
        # A captured group records the surviving clusters, not the declared ceiling.
        for group in network.capture_current_state().parameters:
            if isinstance(group.description, Hidden):
                assert active[group.description.name] == group.description.K
        assert torch.isfinite(network.predict(X, X_cat=(codes,))).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_regression_centroid_fallback_applies_only_at_zero_mass(self, monkeypatch, dtype):
        X = torch.zeros(4, 1, dtype=dtype, device=DEVICE)
        y = torch.tensor([2.0, 6.0, 6.0, 6.0], dtype=dtype, device=DEVICE)
        recipe = Recipe.chain(Input(K=4), RegressionHead())
        materialise = fit_ops._materialise
        update_head = _RegressionBlock.update_parameters_
        observed = []

        # An InitialState holds no affiliations, so the cluster masses are written here.
        def seed_masses(*args, **kwargs):
            session = materialise(*args, **kwargs)
            gamma = session.graph.input.gamma
            eps = torch.finfo(dtype).eps
            gamma.zero_()
            gamma[:, 3] = 1.0
            gamma[0] = torch.tensor(
                [0.0, 4 * eps, 8 * eps, 1 - 12 * eps], dtype=dtype, device=DEVICE
            )
            return session

        def observe(head, session):
            update_head(head, session)
            # Supervision gives each row weight 1/4, so the first three denominators
            # are zero, epsilon, 2*epsilon. Only the zero-mass column takes the fallback.
            observed.append(head.C_y[0, :3].clone())

        monkeypatch.setattr(fit_ops, "_materialise", seed_masses)
        monkeypatch.setattr(_RegressionBlock, "update_parameters_", observe)
        Network.fit(recipe, X, y, max_iter=1)
        torch.testing.assert_close(
            observed[0], torch.tensor([5.0, 2.0, 2.0], dtype=dtype, device=DEVICE), rtol=0, atol=0
        )
