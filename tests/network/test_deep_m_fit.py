"""Deep M fitting, ownership, pruning and lifecycle schedule through Network."""

from __future__ import annotations

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE
from network._fixtures import (
    BLOB_SPREADS,
    assert_simplex_rows,
    blobs,
    deep_classification_recipe,
    deep_regression_recipe,
    regression_blobs,
)
from network._fixtures import materialise as make_session

from entlearn import Network
from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.fit import _fit_iteration_


class TestDeepMFit:
    def test_hidden_m_chain_fits_and_predicts(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=7)
        recipe = deep_classification_recipe((4, 3))

        network = Network.fit(recipe, X_cont, y, max_iter=12, seed=5)
        prediction = network.predict(X_cont)

        assert prediction.shape == (X_cont.shape[0], 3)
        assert_simplex_rows(prediction)
        assert all(math.isfinite(loss) for loss in network.diagnostics.loss_history)
        assert network.diagnostics.n_iter > 0

    def test_hidden_m_chain_fits_and_predicts_regression(self) -> None:
        X_cont, target, _ = regression_blobs(2)
        recipe = deep_regression_recipe((3, 2), theta_alpha=1.0)

        network = Network.fit(recipe, X_cont, target, max_iter=12, seed=5)
        prediction = network.predict(X_cont)

        assert prediction.shape == target.shape
        assert bool(torch.isfinite(prediction).all())
        assert all(math.isfinite(loss) for loss in network.diagnostics.loss_history)

    def test_any_chain_depth_uses_stable_active_width_order(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=8)
        recipe = deep_classification_recipe((5, 4, 3, 3))

        network = Network.fit(recipe, X_cont, y, max_iter=4, seed=2)

        assert tuple(name for name, _ in network.schema.K_active) == recipe.stable_order()
        assert len(network._graph.order) == 5
        assert network.predict(X_cont).shape == (X_cont.shape[0], 3)

    def test_each_hidden_target_owns_its_named_incoming_state(self) -> None:
        X_cont, y = blobs(2, BLOB_SPREADS["separated"], seed=3)
        recipe = deep_classification_recipe((4, 3, 2))
        session = make_session(recipe, X_cont, y, seed=4)
        first = session.graph.blocks["hidden_1"]
        second = session.graph.blocks["hidden_2"]
        assert isinstance(first, _HiddenBlock)
        assert isinstance(second, _HiddenBlock)

        assert tuple(first.incoming) == ("input_to_hidden_1",)
        assert first.incoming["input_to_hidden_1"].theta.shape == (3, 4)
        assert tuple(second.incoming) == ("hidden_1_to_hidden_2",)
        assert second.incoming["hidden_1_to_hidden_2"].theta.shape == (2, 3)
        assert not hasattr(session.graph, "transitions")
        assert session.graph.terminal.coupling is None

    def test_hidden_pruning_compacts_both_neighbouring_transition_axes(self) -> None:
        X_cont, y = blobs(2, BLOB_SPREADS["separated"], seed=2)
        session = make_session(deep_classification_recipe((4, 3, 2)), X_cont, y, seed=6)
        first = session.graph.blocks["hidden_1"]
        second = session.graph.blocks["hidden_2"]
        assert isinstance(first, _HiddenBlock)
        assert isinstance(second, _HiddenBlock)
        first.gamma.zero_()
        first.gamma[:, 0] = 1.0

        first.prune_(session)

        assert first.K == 1
        assert first.incoming["input_to_hidden_1"].theta.shape == (1, 4)
        assert second.incoming["hidden_1_to_hidden_2"].theta.shape == (2, 1)

    def test_final_hidden_pruning_compacts_a_regression_head(self) -> None:
        X_cont, target, _ = regression_blobs(2)
        recipe = deep_regression_recipe((4, 3), theta_alpha=1.0)
        session = make_session(recipe, X_cont, target, seed=6)
        hidden = session.graph.blocks["hidden_1"]
        head = session.graph.head
        assert isinstance(hidden, _HiddenBlock)
        assert isinstance(head, _RegressionBlock)
        hidden.gamma.zero_()
        hidden.gamma[:, 0] = 1.0

        hidden.prune_(session)

        assert hidden.K == 1
        assert head.C_y.shape == (2, 1)

    def test_iteration_runs_an_explicit_head_to_input_schedule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        X_cont, y = blobs(2, BLOB_SPREADS["separated"], seed=5)
        session = make_session(deep_classification_recipe((4, 3, 2)), X_cont, y, seed=7)
        events: list[str] = []

        original_head_update = _ClassificationBlock.update_parameters_
        original_hidden_update = _HiddenBlock.update_affiliations_
        original_hidden_prune = _HiddenBlock.prune_
        original_connection_update = _HiddenBlock.update_incoming_connections_
        original_input_update = _StandardInputBlock.update_parameters_

        def head_update(block, fit_session) -> None:
            events.append("output")
            original_head_update(block, fit_session)

        def hidden_update(block, fit_session) -> None:
            events.append(f"{block.description.name}.affiliations")
            original_hidden_update(block, fit_session)

        def hidden_prune(block, fit_session) -> None:
            events.append(f"{block.description.name}.prune")
            original_hidden_prune(block, fit_session)

        def connection_update(block, fit_session) -> None:
            events.append(f"{block.description.name}.incoming")
            original_connection_update(block, fit_session)

        def input_update(block, fit_session) -> None:
            events.append("input")
            original_input_update(block, fit_session)

        monkeypatch.setattr(_ClassificationBlock, "update_parameters_", head_update)
        monkeypatch.setattr(_HiddenBlock, "update_affiliations_", hidden_update)
        monkeypatch.setattr(_HiddenBlock, "prune_", hidden_prune)
        monkeypatch.setattr(_HiddenBlock, "update_incoming_connections_", connection_update)
        monkeypatch.setattr(_StandardInputBlock, "update_parameters_", input_update)

        _fit_iteration_(session)

        assert events == [
            "output",
            "hidden_2.affiliations",
            "hidden_2.prune",
            "hidden_2.incoming",
            "hidden_1.affiliations",
            "hidden_1.prune",
            "hidden_1.incoming",
            "input",
        ]

    def test_deep_float32_fit_does_not_promote_connection_state(self) -> None:
        X_cont = torch.tensor(
            [[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]],
            dtype=torch.float32,
            device=DEVICE,
        )
        y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=DEVICE)

        network = Network.fit(deep_classification_recipe((3, 2)), X_cont, y, max_iter=2, seed=3)
        hidden = network._graph.blocks["hidden_1"]
        assert isinstance(hidden, _HiddenBlock)

        assert hidden.gamma.dtype is torch.float32
        assert hidden.incoming["input_to_hidden_1"].theta.dtype is torch.float32
        assert network.predict(X_cont).dtype is torch.float32

    def test_complete_deep_iteration_allocates_no_tensor_storage(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=4)
        session = make_session(deep_classification_recipe((4, 3, 2)), X_cont, y, seed=8, warmup=1)

        assert_zero_alloc(_fit_iteration_, session)

    def test_complete_deep_regression_iteration_allocates_no_tensor_storage(self) -> None:
        X_cont, target, _ = regression_blobs(3)
        recipe = deep_regression_recipe((4, 3), theta_alpha=1.0, epsilon_M=0.2)
        session = make_session(recipe, X_cont, target, seed=8, warmup=1)

        assert_zero_alloc(_fit_iteration_, session)
