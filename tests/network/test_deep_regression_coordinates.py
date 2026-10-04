"""Deep regression costs, coordinate descent and allocation contracts."""

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE
from network._fixtures import (
    assert_input_coordinates_non_increasing,
    assert_non_increasing,
    deep_regression_session,
)

from entlearn import Coupling
from entlearn.network.blocks import regression as reg
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.data import _RegressionSupervision
from entlearn.network.fit import _fit_iteration_, _loss

_OUTPUT_WEIGHT_CASES = (
    (float("inf"), None),
    (float("inf"), (1.0, 2.0, 3.0)),
    (0.0, None),
    (0.2, None),
)
_OUTPUT_WEIGHT_IDS = ("implicit-uniform", "fixed", "learned-hard", "learned-soft")


@pytest.mark.parametrize("coupling", (Coupling.M, Coupling.S))
@pytest.mark.parametrize(("epsilon_M", "W_M"), _OUTPUT_WEIGHT_CASES, ids=_OUTPUT_WEIGHT_IDS)
class TestDeepRegressionCoordinates:
    @pytest.mark.parametrize("weighted", (False, True), ids=("uniform-rows", "weighted-masked"))
    def test_head_adds_the_analytical_residual_to_the_live_hidden_cost(
        self, coupling: Coupling, epsilon_M: float, W_M: tuple[float, ...] | None, weighted: bool
    ) -> None:
        session = deep_regression_session(
            coupling=coupling, epsilon_M=epsilon_M, W_M=W_M, weighted=weighted
        )
        graph = session.graph
        head = graph.head
        assert isinstance(head, _RegressionBlock)
        head.update_parameters_(session)
        supervision = session.data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        T, M = session.data.target.shape
        K = head.C_y.shape[1]
        cost = session.workspace.matrix[:T, :K]
        initial_cost = torch.linspace(0.0, 1.0, T * K, dtype=DTYPE, device=DEVICE).reshape(T, K)
        cost.copy_(initial_cost)
        output_weights = (
            head.W_M
            if head.W_M is not None
            else torch.full((M,), 1.0 / M, dtype=DTYPE, device=DEVICE)
        )
        squared_residual = (session.data.target[:, :, None] - head.C_y[None, :, :]).square()
        expected = initial_cost + graph.connection_into(head).delta * (
            supervision.row_weights[:, None]
            * session.data.labelled[:, None]
            * (output_weights[None, :, None] * squared_residual).sum(dim=1)
        )

        assert_zero_alloc(head.accumulate_into_source_cost_, session, cost=cost)

        torch.testing.assert_close(cost, expected)

    @pytest.mark.parametrize("epsilon", (0.0, 0.2), ids=("hard", "soft"))
    @pytest.mark.parametrize("widths", ((4, 3), (3, 5), (4, 3, 2)))
    @pytest.mark.parametrize("weighted", (False, True), ids=("uniform-rows", "weighted-masked"))
    @pytest.mark.parametrize("warmup", (0, 3), ids=("initial", "warmed"))
    def test_each_coordinate_is_non_increasing(
        self,
        coupling: Coupling,
        epsilon_M: float,
        W_M: tuple[float, ...] | None,
        widths: tuple[int, ...],
        epsilon: float,
        weighted: bool,
        warmup: int,
    ) -> None:
        session = deep_regression_session(
            widths=widths,
            epsilon=epsilon,
            coupling=coupling,
            epsilon_M=epsilon_M,
            W_M=W_M,
            weighted=weighted,
            warmup=warmup,
        )
        graph = session.graph
        head = graph.head
        assert isinstance(head, _RegressionBlock)
        before = _loss(session)
        for update, label in (
            (reg.update_centroids_, "regression centroids"),
            (reg.update_output_weights_, "regression output weights"),
        ):
            update(head, session)
            after = _loss(session)
            assert_non_increasing(before, after, label)
            before = after
        for name in reversed(graph.order[1:-1]):
            block = graph.blocks[name]
            assert isinstance(block, _HiddenBlock)
            block.update_affiliations_(session)
            block.prune_(session)
            after = _loss(session)
            assert_non_increasing(before, after, f"{name} affiliations")
            before = after
            block.update_incoming_connections_(session)
            after = _loss(session)
            assert_non_increasing(before, after, f"{name} incoming connection")
            before = after

        graph.accumulate_outgoing_cost_(
            graph.input, graph.input.live_cache.disc_cost[:, : graph.input.K], session
        )
        assert_input_coordinates_non_increasing(graph.input, session, before)

    def test_complete_iteration_allocates_no_tensor_storage(
        self, coupling: Coupling, epsilon_M: float, W_M: tuple[float, ...] | None
    ) -> None:
        session = deep_regression_session(coupling=coupling, epsilon_M=epsilon_M, W_M=W_M, warmup=3)

        assert_zero_alloc(_fit_iteration_, session)
