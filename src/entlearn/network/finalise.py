"""Common fitted policy, schema and scoring-reference finalisation."""

from dataclasses import replace

from entlearn.network.calibration import _calibrate_epsilon_P
from entlearn.network.config import PredictConfig
from entlearn.network.data import _ClassificationSupervision, _StagedData
from entlearn.network.predict import _head_logits
from entlearn.network.queries import build_scoring_reference
from entlearn.network.session import _CompiledGraph
from entlearn.network.state import InitialState, _FittedNetwork, _Trajectory


def finalise(
    graph: _CompiledGraph,
    data: _StagedData,
    trajectory: _Trajectory,
    predict_config: PredictConfig,
    initial_state: InitialState,
    *,
    scoring_reference: bool,
    max_iter: int,
    tol: float,
) -> _FittedNetwork:
    """Complete the fitted schema and prediction policy before publication.

    ``scoring_reference`` builds the recovered-on-training W_T reference that
    ``score_samples`` ranks against. A fold fit that selection scores and discards
    passes ``False``, and keeps no reference.
    """
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
    epsilon_P_source = None
    if predict_config.output_mode == "geometric":
        if predict_config.epsilon_P is None:
            supervision = data.supervision
            assert isinstance(supervision, _ClassificationSupervision)
            epsilon_P = _calibrate_epsilon_P(
                _head_logits(graph, data.X_cont, data.X_cat),
                supervision.weighted_target,
                graph.connection_into(graph.head).delta,
            )
            predict_config = replace(predict_config, epsilon_P=epsilon_P)
            epsilon_P_source = "derived"
        else:
            epsilon_P_source = "supplied"
    graph.release_caches()
    return _FittedNetwork(
        graph,
        schema,
        trajectory,
        predict_config,
        epsilon_P_source,
        initial_state,
        build_scoring_reference(graph, data.X_cont, data.X_cat, predict_config)
        if scoring_reference
        else None,
        max_iter=max_iter,
        tol=tol,
    )
