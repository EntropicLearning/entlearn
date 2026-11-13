"""Common fitted policy, schema and scoring-reference finalisation."""

from dataclasses import replace

import torch

from entlearn.network.blocks.input_common import has_instance_weight_recovery
from entlearn.network.calibration import _calibrate_epsilon_P
from entlearn.network.config import PredictConfig
from entlearn.network.data import _ClassificationSupervision, _StagedData
from entlearn.network.predict import _forward, _head_logits, _read_out
from entlearn.network.session import _CompiledGraph
from entlearn.network.state import InitialState, _FittedNetwork, _Trajectory
from entlearn.network.validation import _all_finite


def finalise(
    graph: _CompiledGraph,
    data: _StagedData,
    trajectory: _Trajectory,
    predict_config: PredictConfig,
    initial_state: InitialState,
    *,
    max_iter: int,
    tol: float,
) -> _FittedNetwork:
    """Record the fitted schema and prediction policy; ``complete`` makes it ready to predict."""
    schema = replace(
        data.schema,
        K_active=tuple(
            (
                name,
                (
                    graph.head.output_width
                    if name == graph.head.description.name
                    else graph.affiliations(name).shape[1]
                ),
            )
            for name in graph.order
        ),
    )
    graph.release_caches()
    supplied = predict_config.output_mode == "geometric" and predict_config.epsilon_P is not None
    return _FittedNetwork(
        graph,
        schema,
        trajectory,
        predict_config,
        "supplied" if supplied else None,
        initial_state,
        None,
        max_iter=max_iter,
        tol=tol,
    )


def complete(
    fitted: _FittedNetwork,
    data: _StagedData,
    *,
    scoring_reference: bool = True,
    training_prediction: bool = False,
) -> tuple[_FittedNetwork, torch.Tensor | None]:
    """Calibrate a pending output temperature and build a pending scoring reference.

    One forward pass over the training rows serves both, and the training-row prediction
    that ``training_prediction`` returns beside the fit.

    Raises:
        RuntimeError: If the fitted recovery produces invalid reference values.
    """
    graph, config = fitted.graph, fitted.predict_config
    calibrating = config.output_mode == "geometric" and config.epsilon_P is None
    recovering = (
        scoring_reference and fitted.Wt_ref is None and has_instance_weight_recovery(graph.input)
    )
    if not (calibrating or recovering or training_prediction):
        return fitted, None
    query, gammas = _forward(graph, data.X_cont, X_cat_new=data.X_cat)
    if calibrating:
        supervision = data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        epsilon_P = _calibrate_epsilon_P(
            _head_logits(graph, gammas), supervision.weighted_target, graph.head_delta
        )
        config = replace(config, epsilon_P=epsilon_P)
        fitted = replace(fitted, predict_config=config, epsilon_P_source="derived")
    if not (recovering or training_prediction):
        return fitted, None
    result = _read_out(
        graph, query, gammas, config, details=("instance_weights",) if recovering else ()
    )
    if recovering:
        reference = result.instance_weights
        if reference is None or not _all_finite(reference) or bool((reference < 0).any()):
            raise RuntimeError("fitted recovery produced an invalid scoring reference")
        fitted = replace(fitted, Wt_ref=reference)
    return fitted, result.prediction if training_prediction else None
