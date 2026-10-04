"""Pruning and denominator safety exercised through Network fitting."""

from __future__ import annotations

import pytest
import torch
from conftest import DEVICE
from network._fixtures import deep_classification_recipe, deep_regression_recipe
from torch.utils._python_dispatch import TorchDispatchMode

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.network import fit as fit_ops
from entlearn.network.blocks import input as inp
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.regression import _RegressionBlock


class _FiniteDivisions(TorchDispatchMode):
    """Observe ``aten.div.out``, the overload the guarded denominators divide through.

    Watching the intermediate division rather than the subsequently masked output is
    the point: the output is repaired, the intermediate is where a zero denominator
    shows. The narrowness to one overload is deliberate rather than incidental, since
    ``normalise_`` divides an empty slice's 0/0 through ``div_`` and repairs it with
    ``nan_to_num_``, so a probe over every division overload would fire on that
    documented path instead. ``observed`` counts the watched divisions, so a test can
    price that the probe saw anything at all: an otherwise legal rewrite of the guarded
    kernels to an in-place ``div_`` would leave this mode silent and its tests green.
    """

    def __init__(self):
        super().__init__()
        self.observed = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.div.out:
            self.observed += 1
            assert torch.isfinite(result).all(), "non-finite centroid division before masking"
        return result


class TestPruning:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("name", ["input", "hidden_1", "hidden_2"])
    def test_one_graph_prune_compacts_every_owner_at_the_epsilon_boundary(
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
            eps = torch.finfo(dtype).eps
            # Unweighted masses are below, exactly at, and above epsilon. Even the
            # last has weighted mass below epsilon, so weighted pruning would fail.
            block.gamma[:, 1:] = (
                torch.tensor([eps / 2, eps, 2 * eps], dtype=dtype, device=DEVICE) / 8
            )
            block.gamma[:, 0] = 1 - block.gamma[:, 1:].sum(dim=1)
            expected_gamma = block.gamma[:, [0, 3]].clone()
            scratch_pointer = session.workspace.matrix.data_ptr()
            graph.prune_(block, session.workspace)
            torch.testing.assert_close(block.gamma, expected_gamma, rtol=0, atol=0)
            assert block.K == 2
            assert session.workspace.matrix.data_ptr() == scratch_pointer
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
                    torch.testing.assert_close(state.log_theta, state.theta.clamp_min(eps).log())
                elif task == "classification":
                    assert target.theta.shape == target.log_theta.shape == (2, source_width)
                    torch.testing.assert_close(target.log_theta, target.theta.clamp_min(eps).log())
                    axis = 0 if coupling is Coupling.M else 1
                    torch.testing.assert_close(
                        target.theta.sum(dim=axis),
                        torch.ones(target.theta.shape[1 - axis], dtype=dtype, device=DEVICE),
                    )
                else:
                    assert target.C_y.shape == (1, source_width)
            input_block = graph.input
            K = input_block.K
            assert input_block.continuous_centroids.shape == (K, 1)
            assert input_block.categorical_centroids[0].shape == (K, 2)
            assert input_block.log_C_cat[0].shape == (K, 2)
            assert input_block.live_cache.disc_cost.shape == (8, K)
            assert input_block.live_cache.sqdist.shape == (8, K)
            assert input_block.live_cache.categorical_cost.shape == (8, K)
            observed.append(name)
            return session

        monkeypatch.setattr(fit_ops, "_materialise", prune_boundary)
        network = Network.fit(recipe, X, y, X_cat=(codes,), sample_weights=weights, max_iter=1)
        assert observed == [name]
        # The following real iteration may legitimately prune another cluster.
        active = dict(network.schema.K_active)
        assert active[name] == network._graph.blocks[name].K <= 2
        # A captured group records the surviving clusters, not the declared ceiling.
        for group in network.capture_current_state().parameters:
            if isinstance(group.description, Hidden):
                assert active[group.description.name] == group.description.K
        assert torch.isfinite(network.predict(X, X_cat=(codes,))).all()

    def test_production_prunes_input_and_head_before_the_centroid_coordinate(self, monkeypatch):
        X = torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=torch.float64, device=DEVICE)
        codes = torch.tensor([0, 0, 1, 1], device=DEVICE)
        recipe = Recipe.chain(Input(K=3, epsilon=0.0), ClassificationHead(coupling=Coupling.M))
        initial = Network.initialise(
            recipe,
            X,
            codes,
            X_cat=(codes,),
            continuous_centroids=torch.zeros(3, 1, dtype=X.dtype, device=DEVICE),
            categorical_centroids=(torch.full((3, 2), 0.5, dtype=X.dtype, device=DEVICE),),
        )
        update_centroids = inp.update_centroids_
        widths_at_division = []

        def observe(block, session):
            widths_at_division.append((block.K, session.graph.head.theta.shape[1]))
            update_centroids(block, session)

        monkeypatch.setattr(inp, "update_centroids_", observe)
        network = Network.fit(recipe, X, codes, X_cat=(codes,), initial_state=initial, max_iter=1)
        assert widths_at_division == [(1, 1)]
        assert dict(network.schema.K_active)["input"] == 1

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_regression_centroid_fallback_uses_the_dtype_boundary(self, monkeypatch, dtype):
        X = torch.zeros(4, 1, dtype=dtype, device=DEVICE)
        y = torch.tensor([2.0, 6.0, 6.0, 6.0], dtype=dtype, device=DEVICE)
        recipe = Recipe.chain(Input(K=4), RegressionHead())
        materialise = fit_ops._materialise
        update_head = _RegressionBlock.update_parameters_
        observed = []

        def seed_masses(*args, **kwargs):
            session = materialise(*args, **kwargs)
            gamma = session.graph.input.gamma
            eps = torch.finfo(dtype).eps
            gamma.zero_()
            gamma[:, 3] = 1.0
            gamma[0] = torch.tensor(
                [2 * eps, 4 * eps, 8 * eps, 1 - 14 * eps], dtype=dtype, device=DEVICE
            )
            return session

        def observe(head, session):
            update_head(head, session)
            # Supervision gives each row weight 1/4, so the first three denominators
            # are epsilon/2, epsilon, 2*epsilon. Only the last uses its cluster mean.
            observed.append(head.C_y[0, :3].clone())

        monkeypatch.setattr(fit_ops, "_materialise", seed_masses)
        monkeypatch.setattr(_RegressionBlock, "update_parameters_", observe)
        with _FiniteDivisions() as divisions:
            Network.fit(recipe, X, y, max_iter=1)
        assert divisions.observed, "the probe watched no guarded division"
        torch.testing.assert_close(
            observed[0], torch.tensor([5.0, 5.0, 2.0], dtype=dtype, device=DEVICE), rtol=0, atol=0
        )
