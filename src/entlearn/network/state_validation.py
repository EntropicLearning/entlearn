"""Validation of provided, transferred and published state, each where it enters an operation.

- ``validate_original_state``: the original states a fitted model retains.
- ``validate_parameter_state``: fitted tensors transferred into a continuation.
- ``validate_prediction_ready_state``: a published model decoded from a snapshot.
"""

import math
from dataclasses import dataclass
from typing import assert_never

import torch

from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.blocks.input_common import has_instance_weight_recovery
from entlearn.network.blocks.manifold import _ManifoldInputBlock, stage_projectors
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.blocks.types import _ClusteringBlock, _InputBlock, _OutputBlock
from entlearn.network.initialisation.resolve_state import parameters_match, validate_initial_state
from entlearn.network.parameter_validation import validate_simplex_tensor, validate_tensor
from entlearn.network.session import _CompiledGraph
from entlearn.network.state import (
    DataSchema,
    InitialState,
    _check_cluster_match,
    _FittedNetwork,
    _input_width,
)
from entlearn.network.validation import _normalise_weights
from entlearn.recipe import Coupling, Head, Hidden, InputBlock


def validate_original_state(state: InitialState, schema: DataSchema) -> None:
    """Require valid starting groups in the fitted feature schema.

    Unlike ``validate_initial_state``, this validates groups retained by a fitted
    model, not groups provided to a new fit. Their feature meanings must already agree
    with that model, and the decoder has already read them in its dtype and onto its
    device. Intrinsic group validity is delegated to ``validate_initial_state``.
    """
    validate_initial_state(state)
    geometry = state.input_geometry
    if geometry is not None and (
        geometry.continuous_centroids.shape[1] != schema.D_cont
        or tuple(c.shape[1] for c in geometry.categorical_centroids) != schema.M_cat
    ):
        raise ValueError("original input geometry does not match the fitted feature schema")


@dataclass(frozen=True)
class _Placement:
    """The computation dtype and device that every fitted tensor must share."""

    dtype: torch.dtype
    device: torch.device

    def tensor(
        self, value: object, shape: tuple[int, ...], name: str, *, finite: bool = True
    ) -> torch.Tensor:
        """Require a tensor of ``shape`` in this placement."""
        return validate_tensor(
            value, name, shape, dtype=self.dtype, device=self.device, finite=finite
        )

    def simplex(self, value: object, shape: tuple[int, ...], name: str, *, dim: int = -1) -> None:
        """Require a tensor of ``shape`` in this placement with a stochastic ``dim`` axis."""
        validate_simplex_tensor(self.tensor(value, shape, name, finite=False), name, dim)


def validate_parameter_state(fitted: _FittedNetwork, *, require_rows: bool = False) -> None:
    """Require compatible fitted tensors and optionally row-bound coordinates.

    Raises:
        ValueError: If parameters or requested row-bound coordinates are inconsistent.
    """
    graph, schema = fitted.graph, fitted.schema
    if graph.training_rows < 1:
        raise ValueError("training row count must be positive")
    if schema.computation_dtype not in (torch.float32, torch.float64):
        raise ValueError("computation dtype must be torch.float32 or torch.float64")
    anchor = validate_tensor(graph.input.continuous_centroids, "fitted geometry", finite=False)
    place = _Placement(schema.computation_dtype, anchor.device)
    rows = graph.training_rows if require_rows else None
    widths = dict(schema.K_active)
    for name in graph.order:
        block = graph.blocks[name]
        match block:
            case _StandardInputBlock() | _ManifoldInputBlock():
                _validate_input_parameters(block, widths[name], schema, rows, place)
            case _HiddenBlock():
                _validate_hidden_parameters(block, widths, rows, place)
            case _ClassificationBlock() | _RegressionBlock():
                _validate_head_parameters(block, widths[graph.terminal.source], schema, place)
            case _:
                assert_never(block)
    if widths[graph.order[-1]] != schema.M:
        raise ValueError("number of head outputs does not match the fitted schema")


def _validate_affiliations(
    block: _ClusteringBlock, K: int, rows: int | None, place: _Placement
) -> None:
    """Require an active cluster count within the Recipe's and, with rows, affiliations."""
    name = block.description.name
    if not 1 <= K <= block.description.K:
        raise ValueError(f"invalid number of active clusters for {name!r}")
    if rows is not None:
        place.simplex(block.gamma, (rows, K), f"{name}.gamma")


def _validate_input_parameters(
    block: _InputBlock, K: int, schema: DataSchema, rows: int | None, place: _Placement
) -> None:
    """Validate one input block's geometry, normaliser and optional instance weights."""
    name = block.description.name
    _validate_affiliations(block, K, rows, place)
    place.tensor(block.continuous_centroids, (K, schema.D_cont), f"{name}.centroids")
    if rows is not None:
        place.simplex(block.instance_weights, (rows,), f"{name}.instance_weights")
    log_partition = place.tensor(block.log_partition, (), f"{name}.log_partition", finite=False)
    if bool(torch.isnan(log_partition)) or bool(torch.isneginf(log_partition)):
        raise ValueError("invalid training log normaliser")
    if isinstance(block, _StandardInputBlock):
        D = schema.D_cont + len(schema.M_cat)
        place.simplex(block.feature_weights, (D,), f"{name}.feature_weights")
        for c, log_c, M in zip(
            block.categorical_centroids, block.log_C_cat, schema.M_cat, strict=True
        ):
            place.simplex(c, (K, M), f"{name}.categorical_centroids")
            place.tensor(log_c, (K, M), f"{name}.log_C_cat")
        return
    if schema.M_cat:
        raise ValueError("manifold input cannot retain categorical features")
    d = block.description.subspace_dimension
    projectors = place.tensor(
        block.manifold_projectors, (K, schema.D_cont, d), f"{name}.manifold_projectors"
    )
    stage_projectors(projectors, block.continuous_centroids, d, allow_conversion=False)


def _validate_hidden_parameters(
    block: _HiddenBlock, widths: dict[str, int], rows: int | None, place: _Placement
) -> None:
    """Validate one hidden block's incoming transitions and their prior cluster counts."""
    K = widths[block.description.name]
    _validate_affiliations(block, K, rows, place)
    for state in block.incoming.values():
        connection = state.description
        shape = (K, widths[connection.source])
        dim = 0 if connection.coupling is Coupling.M else 1
        place.simplex(state.theta, shape, f"{connection.name}.theta", dim=dim)
        place.tensor(state.log_theta, shape, f"{connection.name}.log_theta")
        if any(
            initial < active for initial, active in zip(state.initial_widths, shape, strict=True)
        ):
            raise ValueError("invalid initial source or target cluster count")


def _validate_head_parameters(
    block: _OutputBlock, source_width: int, schema: DataSchema, place: _Placement
) -> None:
    """Validate the head's transition, or its output centroids and output weights."""
    name = block.description.name
    if isinstance(block, _RegressionBlock):
        place.tensor(block.C_y, (schema.M, source_width), f"{name}.C_y")
        if block.W_M is not None:
            place.simplex(block.W_M, (schema.M,), f"{name}.W_M")
        return
    description = block.description
    if description.n_classes is not None and description.n_classes != schema.M:
        raise ValueError("declared number of classes does not match the fitted schema")
    dim = 0 if description.coupling is Coupling.M else 1
    place.simplex(block.theta, (schema.M, source_width), f"{name}.theta", dim=dim)
    place.tensor(block.log_theta, block.theta.shape, f"{name}.log_theta")


def validate_original_bindings(
    state: InitialState, graph: _CompiledGraph, schema: DataSchema
) -> None:
    """Require retained starting groups in resolved graph order and compatible dimensions."""
    groups = {group.description.name: group for group in state.parameters}
    expected = tuple(name for name in graph.order[1:] if name in groups)
    if tuple(groups) != expected:
        raise ValueError("original parameter groups do not match the resolved graph order")
    assert state.input_geometry is not None
    source_width = _check_cluster_match(state.input_geometry, graph.input.description)
    if tuple(name for name, _ in state.connection_sub_seeds) != tuple(
        connection.name for connection in graph.connections
    ):
        raise ValueError("original state seed names disagree with the fitted graph")
    for name in graph.order[1:]:
        description = graph.blocks[name].description
        assert isinstance(description, Hidden | Head)
        (connection,) = graph.incoming[name]
        group = groups.get(name)
        if group is not None and not parameters_match(
            group, description, source_width, schema.M, coupling=connection.coupling
        ):
            raise ValueError(f"original parameter group {name!r} is incompatible with the graph")
        if isinstance(description, Hidden):
            source_width = description.K


def validate_prediction_ready_state(fitted: _FittedNetwork, *, require_rows: bool = False) -> None:
    """Require completed query references and original geometry as well as parameters.

    Continuation validates only the parameters it transfers, before new references
    and original-state placement are attached. Those intermediate graphs are not
    prediction artefacts.
    """
    validate_parameter_state(fitted, require_rows=require_rows)
    graph, schema = fitted.graph, fitted.schema
    anchor = graph.input.continuous_centroids
    geometry = fitted.initial_state.input_geometry
    assert geometry is not None
    original_width = _input_width(geometry, graph.input.description)
    # The snapshot collection validates each original once, including losing states.
    # Pruning can leave fewer active clusters than the original state contains.
    if (
        geometry.continuous_centroids.device != anchor.device
        or graph.connection_sub_seeds != fitted.initial_state.connection_sub_seeds
        or original_width < graph.input.K
    ):
        raise ValueError("original state is incompatible with the fitted graph")
    maxima = {
        block.name: block.K
        for block in graph.recipe.blocks
        if isinstance(block, InputBlock | Hidden)
    }
    maxima[graph.order[0]] = original_width
    for block in graph.blocks.values():
        if isinstance(block, _HiddenBlock):
            for connection in block.incoming.values():
                bounds = (maxima[block.description.name], maxima[connection.description.source])
                if any(
                    initial > maximum
                    for initial, maximum in zip(connection.initial_widths, bounds, strict=True)
                ):
                    raise ValueError(
                        "connection prior cluster counts exceed initial graph capacities"
                    )
    head = graph.head
    if isinstance(head, _RegressionBlock):
        _validate_regression_output_weights(head, schema, anchor.device)
    _validate_scoring_reference(fitted, anchor.device)


def _validate_scoring_reference(fitted: _FittedNetwork, device: torch.device) -> None:
    """Require a valid scoring reference exactly when the input recovers instance weights."""
    graph, reference = fitted.graph, fitted.Wt_ref
    if has_instance_weight_recovery(graph.input):
        if (
            not isinstance(reference, torch.Tensor)
            or reference.shape != (graph.training_rows,)
            or reference.dtype != fitted.schema.computation_dtype
            or reference.device != device
            or not bool(torch.isfinite(reference).all())
            or bool((reference < 0).any())
        ):
            raise ValueError("invalid scoring reference")
    elif reference is not None:
        raise ValueError("scoring reference requires instance-weight recovery")


def _validate_regression_output_weights(
    head: _RegressionBlock, schema: DataSchema, device: torch.device
) -> None:
    description = head.description
    # Absent weights mean uniform weighting only when neither learning nor fixed
    # Recipe weights require an explicit vector.
    if head.W_M is None and (math.isfinite(description.epsilon_M) or description.W_M is not None):
        raise ValueError("missing fitted regression output weights")
    if description.W_M is not None:
        if len(description.W_M) != schema.M:
            raise ValueError("fixed output weights do not match the number of regression targets")
        expected = _normalise_weights(
            torch.tensor(description.W_M, dtype=schema.computation_dtype, device=device),
            name="W_M",
        )
        # A float64 model may retain weights originally normalised in float32.
        if head.W_M is None or not torch.allclose(
            head.W_M, expected, rtol=0, atol=8 * schema.M * torch.finfo(torch.float32).eps
        ):
            raise ValueError("fitted output weights disagree with the fixed Recipe values")


def resume_error(fitted: _FittedNetwork) -> str | None:
    """Return a capability failure without treating absent payloads as fitted state."""
    try:
        validate_parameter_state(fitted, require_rows=True)
    except ValueError as error:
        return f"Network has no valid resumable row-bound state: {error}"
    return None
