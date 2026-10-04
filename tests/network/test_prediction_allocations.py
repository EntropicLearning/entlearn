"""Narrow allocation guards for query-local refinement coordinates."""

import pytest
import torch
from _alloc import assert_zero_alloc
from network._fixtures import prediction_model, prediction_refinement

from entlearn import Coupling, PredictConfig


class TestPredictionCoordinateAllocations:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["mixed", "manifold"])
    @pytest.mark.parametrize("epsilon", [0.0, 0.1])
    @torch.inference_mode()
    def test_updates_and_objective_reuse_allocated_scratch(self, task, kind, epsilon):
        model, X, categories = prediction_model(
            task, input_kind=kind, coupling=Coupling.S, epsilon=epsilon
        )
        config = PredictConfig(
            predict_mode="iterative", epsilon_P=0.2 if task == "classification" else None
        )
        state = prediction_refinement(model, X, categories, config)
        assert_zero_alloc(state.prediction_step_)
        for name in reversed(state.graph.order[1:-1]):
            assert_zero_alloc(state.affiliation_step_, name)
        assert_zero_alloc(state.input_step_)
        assert_zero_alloc(state.loss)
