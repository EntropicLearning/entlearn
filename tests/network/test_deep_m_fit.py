"""Deep M fitting and its fitted state through Network."""

from __future__ import annotations

import math

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, FIXED_BUDGET
from network._fixtures import (
    BLOB_SPREADS,
    assert_simplex_rows,
    blobs,
    deep_classification_recipe,
    deep_regression_recipe,
    regression_blobs,
    skip_fit_iterations,
)
from network._fixtures import materialise as make_session

from entlearn import Network
from entlearn.network.fit import _fit_iteration_

pytestmark = FIXED_BUDGET


class TestDeepMFit:
    def test_hidden_m_chain_fits_and_predicts(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=7)
        # The transition prior keeps the hidden clusters from collapsing onto two blobs.
        recipe = deep_classification_recipe((4, 3), theta_alpha=2.0)

        network = Network.fit(recipe, X_cont, y, max_iter=12, seed=5)
        prediction = network.predict(X_cont)

        assert prediction.shape == (X_cont.shape[0], 3)
        assert_simplex_rows(prediction)
        assert torch.equal(prediction.argmax(dim=1), y)
        assert all(math.isfinite(loss) for loss in network.diagnostics.loss_history)
        assert network.diagnostics.n_iter > 0

    def test_hidden_m_chain_fits_and_predicts_regression(self) -> None:
        X_cont, target, _ = regression_blobs(2)
        recipe = deep_regression_recipe((3, 2), theta_alpha=1.0)

        network = Network.fit(recipe, X_cont, target, max_iter=12, seed=5)
        prediction = network.predict(X_cont)

        # Each separated blob carries one constant target, which the fit recovers.
        torch.testing.assert_close(prediction, target)
        assert all(math.isfinite(loss) for loss in network.diagnostics.loss_history)

    def test_any_chain_depth_uses_stable_active_width_order(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=8)
        recipe = deep_classification_recipe((5, 4, 3, 3))

        network = Network.fit(recipe, X_cont, y, max_iter=4, seed=2)

        assert tuple(name for name, _ in network.schema.K_active) == recipe.stable_order()
        assert network.predict(X_cont).shape == (X_cont.shape[0], 3)

    def test_each_hidden_connection_is_inspected_under_its_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        X_cont, y = blobs(2, BLOB_SPREADS["separated"], seed=3)
        skip_fit_iterations(monkeypatch)

        network = Network.fit(deep_classification_recipe((4, 3, 2)), X_cont, y, max_iter=1, seed=4)

        matrices = network.inspect("transition_matrices")
        assert {name: tuple(theta.shape) for name, theta in matrices.items()} == {
            "input_to_hidden_1": (3, 4),
            "hidden_1_to_hidden_2": (2, 3),
        }

    def test_deep_float32_fit_does_not_promote_connection_state(self) -> None:
        X_cont = torch.tensor(
            [[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]],
            dtype=torch.float32,
            device=DEVICE,
        )
        y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=DEVICE)

        network = Network.fit(deep_classification_recipe((3, 2)), X_cont, y, max_iter=2, seed=3)

        assert network.inspect("training_affiliations")["hidden_1"].dtype == torch.float32
        theta = network.inspect("transition_matrices")["input_to_hidden_1"]
        assert theta.dtype == torch.float32
        assert network.predict(X_cont).dtype == torch.float32

    def test_complete_deep_iteration_allocates_no_tensor_storage(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["separated"], seed=4)
        session = make_session(deep_classification_recipe((4, 3, 2)), X_cont, y, seed=8, warmup=1)

        assert_zero_alloc(_fit_iteration_, session)
