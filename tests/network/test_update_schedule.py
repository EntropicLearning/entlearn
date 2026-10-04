"""Production fitting agrees with explicit coordinates and forward loss reduction."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from itertools import pairwise

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import connection_initialisation_data

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.network import fit as fit_ops
from entlearn.network.blocks import input as inp
from entlearn.network.blocks import manifold as mfld
from entlearn.network.blocks import regression as reg
from entlearn.network.blocks.manifold import _ManifoldInputBlock
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.connections import _ConnectionState

_FORWARD = ("input", "hidden_1", "hidden_2", "hidden_3", "output")


def _explicit_iteration(session):
    graph = session.graph
    head = graph.head
    if isinstance(head, _RegressionBlock):
        reg.update_centroids_(head, session)
        reg.update_output_weights_(head, session)
    else:
        head.update_parameters_(session)
    for name in ("hidden_3", "hidden_2", "hidden_1"):
        hidden = graph.blocks[name]
        hidden.update_affiliations_(session)
        graph.prune_(hidden, session.workspace)
        hidden.update_incoming_connections_(session)
    block = graph.input
    graph.accumulate_outgoing_cost_(block, block.live_cache.disc_cost[:, : block.K], session)
    inp.update_affiliations_(block, session)
    graph.prune_(block, session.workspace)
    inp.update_instance_weights_(block, session)
    if isinstance(block, _ManifoldInputBlock):
        mfld.update_centroids_(block, session)
        mfld.update_projectors_(block, session)
        mfld.refresh_cache_(block, session)
    else:
        inp.stage_statistics_(block, session)
        inp.update_feature_weights_(block, session)
        inp.update_centroids_(block, session)
        inp.refresh_cache_(block, session)


def _explicit_loss(session):
    # Price each contribution separately, then reduce in the specified order.
    total = torch.zeros_like(session.workspace.loss)
    for name in _FORWARD:
        session.workspace.loss.zero_()
        session.graph.blocks[name].partial_loss_(session)
        total.add_(session.workspace.loss)
    for source, target in pairwise(_FORWARD[:4]):
        session.workspace.loss.zero_()
        hidden = session.graph.blocks[target]
        hidden.incoming[f"{source}_to_{target}"].partial_loss_(
            hidden.gamma, session.graph.affiliations(source), session
        )
        total.add_(session.workspace.loss)
    return total.item()


def _tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if is_dataclass(value):
        items = ((field.name, getattr(value, field.name)) for field in fields(value))
    elif isinstance(value, dict):
        items = value.items()
    elif isinstance(value, tuple):
        items = enumerate(value)
    else:
        return {}
    return {
        path: tensor
        for name, child in items
        for path, tensor in _tensors(child, f"{prefix}/{name}").items()
    }


class TestUpdateSchedule:
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("manifold", [False, True])
    def test_production_iteration_matches_explicit_coordinates_and_loss(
        self, monkeypatch, coupling, task, weighted, manifold
    ):
        X, codes = connection_initialisation_data()
        if manifold:
            X = torch.cat((X, X.square()), dim=1)
        y = (
            torch.nn.functional.one_hot(codes, 2).to(DTYPE) * 0.8 + 0.1
            if task == "classification"
            else torch.cat([X.square(), X * 0.3], dim=1)
        )
        head = (
            ClassificationHead(coupling=coupling)
            if task == "classification"
            else RegressionHead(epsilon_M=0.2)
        )
        recipe = Recipe.chain(
            ManifoldInput(K=4, epsilon=0.2, epsilon_T=0.3)
            if manifold
            else Input(K=4, epsilon=0.2, epsilon_D=0.2, epsilon_T=0.3),
            Hidden(K=3, epsilon=0.2),
            Hidden(K=3, epsilon=0.2),
            Hidden(K=2, epsilon=0.2),
            head,
            coupling=coupling,
            theta_alpha=1.2,
        )
        # Storage order must not determine execution or loss-reduction order.
        recipe = replace(
            recipe,
            blocks=tuple(reversed(recipe.blocks)),
            connections=tuple(reversed(recipe.connections)),
        )
        weights = torch.arange(1, 9, dtype=DTYPE, device=DEVICE) if weighted else None
        categorical = None if manifold else (codes,)
        options = dict(X_cat=categorical, sample_weights=weights, max_iter=1, seed=3)
        production = Network.fit(recipe, X, y, **options)
        monkeypatch.setattr(fit_ops, "_fit_iteration_", _explicit_iteration)
        monkeypatch.setattr(fit_ops, "_loss", _explicit_loss)
        explicit = Network.fit(recipe, X, y, **options)
        actual, expected = _tensors(production._graph.blocks), _tensors(explicit._graph.blocks)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(
            torch.tensor(production.diagnostics.loss_history, dtype=DTYPE),
            torch.tensor(explicit.diagnostics.loss_history, dtype=DTYPE),
            rtol=256 * torch.finfo(DTYPE).eps,
            atol=256 * torch.finfo(DTYPE).eps,
        )
        assert len(production.diagnostics.loss_history) == 2
        torch.testing.assert_close(
            production.predict(X, X_cat=categorical), explicit.predict(X, X_cat=categorical)
        )

    def test_loss_connections_are_visited_in_forward_order_not_recipe_storage_order(
        self, monkeypatch
    ):
        X, y = connection_initialisation_data()
        recipe = Recipe.chain(
            Input(K=4, epsilon=0.2),
            Hidden(K=3, epsilon=0.2),
            Hidden(K=2, epsilon=0.2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
        )
        recipe = replace(recipe, connections=tuple(reversed(recipe.connections)))
        seen = []
        partial_loss = _ConnectionState.partial_loss_

        def observe(state, *args, **kwargs):
            seen.append(state.description.name)
            return partial_loss(state, *args, **kwargs)

        monkeypatch.setattr(_ConnectionState, "partial_loss_", observe)
        Network.fit(recipe, X, y, max_iter=1)
        assert seen == ["input_to_hidden_1", "hidden_1_to_hidden_2"] * 2
