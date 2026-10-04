"""Each prediction coordinate, and each complete prediction cycle, descends one objective."""

from functools import partial
from itertools import pairwise

import pytest
import torch
from network._fixtures import (
    prediction_coordinate_loss,
    prediction_model,
    prediction_refinement,
)

from entlearn import Coupling, PredictConfig
from entlearn.network.blocks.input_common import recover_instance_weights_
from entlearn.network.fit import loss_noise_threshold


class TestPredictionCoordinateMonotonicity:
    @pytest.mark.parametrize(
        "task,head,readout",
        [
            ("classification", Coupling.M, "geometric"),
            ("classification", Coupling.M, "arithmetic"),
            ("classification", Coupling.S, "geometric"),
            ("regression", Coupling.M, None),
        ],
    )
    @pytest.mark.parametrize(
        "kind", ["continuous", "mixed", "categorical", "distribution", "manifold"]
    )
    @pytest.mark.parametrize(
        "couplings",
        [
            (),
            (Coupling.M,),
            (Coupling.S,),
            (Coupling.M, Coupling.S),
            (Coupling.S, Coupling.M),
        ],
    )
    @pytest.mark.parametrize("epsilon", [0.0, 0.1])
    @pytest.mark.parametrize("epsilon_T", [0.5, float("inf")])
    def test_every_coordinate_and_complete_cycle_descends(
        self, task, head, readout, kind, couplings, epsilon, epsilon_T
    ):
        model, X, categories = prediction_model(
            task,
            input_kind=kind,
            depth=len(couplings),
            coupling=couplings or Coupling.M,
            head_coupling=head,
            epsilon=epsilon,
            epsilon_T=epsilon_T,
        )
        before = model.predict(X, X_cat=categories)
        policy = PredictConfig(
            predict_mode="iterative",
            output_mode=readout,
            epsilon_P=0.2 if readout == "geometric" else None,
            max_iter=6,
            tol=0,
        )
        details = ("affiliations", "diagnostics")
        if epsilon_T < float("inf"):
            details += ("instance_weights",)
        result = model.predict_with_details(
            X,
            X_cat=categories,
            predict_config=policy,
            predict_init=7,
            details=details,
        )
        assert torch.isfinite(result.prediction).all()
        for previous, current in pairwise(result.loss_history):
            assert current <= previous + 256 * torch.finfo(X.dtype).eps * max(1, abs(previous))
        torch.testing.assert_close(model.predict(X, X_cat=categories), before, rtol=0, atol=0)
        if result.instance_weights is not None:
            assert ((result.instance_weights >= 0) & (result.instance_weights <= 1)).all()

        with torch.inference_mode():
            _assert_every_coordinate_descends(
                model, X[:7], tuple(c[:7] for c in categories), readout
            )


def _assert_every_coordinate_descends(model, X, categories, readout):
    """Price each private prediction coordinate over three refinement iterations."""
    config = PredictConfig(
        predict_mode="iterative",
        output_mode=readout,
        epsilon_P=0.2 if readout == "geometric" else None,
    )
    state = prediction_refinement(model, X, categories, config)
    coordinates = [("prediction", state.prediction_step_)]
    coordinates.extend(
        (name, partial(state.affiliation_step_, name)) for name in reversed(state.graph.order[1:-1])
    )
    coordinates.append(
        ("input affiliations", partial(state.affiliation_step_, state.graph.order[0]))
    )
    if state.weights is not None:
        coordinates.append(
            (
                "recovered instance weights",
                partial(
                    recover_instance_weights_,
                    state.graph.input,
                    state.weights,
                    state.query.instance_cost,
                ),
            )
        )
    for iteration in range(3):
        for name, update in coordinates:
            before = prediction_coordinate_loss(state)
            update()
            after = prediction_coordinate_loss(state)
            assert after <= before + loss_noise_threshold(before, X.dtype), (
                f"{name}, iteration {iteration + 1}: loss increased from {before} to {after}"
            )
