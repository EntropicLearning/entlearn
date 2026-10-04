"""Tensor manifests and reconstruction of portable fitted state."""

from __future__ import annotations

from typing import Any, assert_never, cast

import torch

from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.blocks.manifold import _ManifoldInputBlock
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.blocks.types import _ClusteringBlock, _InputBlock, _RuntimeBlock
from entlearn.network.build import _build_recipe, _BuiltRecipe
from entlearn.network.config import PredictConfig, _resolve_predict_config
from entlearn.network.connections import _ConnectionState
from entlearn.network.persistence.metadata import (
    boolean,
    decode_block,
    decode_description,
    decode_recipe,
    decode_schema,
    decode_seeds,
    decode_trajectory,
    encode_block,
    encode_description,
    encode_recipe,
    encode_schema,
    encode_trajectory,
    integer,
    number,
    record,
    sequence,
    text,
)
from entlearn.network.session import _CompiledGraph
from entlearn.network.state import (
    DataSchema,
    InitialState,
    InputGeometry,
    _ClassificationParameters,
    _DownstreamParameters,
    _FittedNetwork,
    _HiddenParameters,
    _RegressionParameters,
)
from entlearn.network.state_validation import (
    validate_original_state,
    validate_prediction_ready_state,
)
from entlearn.primitives.normalise import _is_soft, floored_log_
from entlearn.recipe import (
    Block,
    ClassificationHead,
    Connection,
    Coupling,
    Hidden,
    Input,
    InputBlock,
    ManifoldInput,
    RegressionHead,
)


class TensorReader:
    """Consume the requested tensors by key, retaining their saved precision."""

    def __init__(self, tensors: dict[str, torch.Tensor], device: torch.device) -> None:
        """Own the unconsumed tensor map and explicit destination device."""
        self.remaining = tensors
        self.device = device

    def take(self, key: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Require a named tensor of the declared shape and dtype."""
        value = self.remaining.pop(key, None)
        if value is None or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"missing or incompatible saved tensor {key!r}")
        return value.detach().to(device=self.device, copy=True)

    def finish(self) -> None:
        """Reject unknown, duplicate-purpose or forbidden row-bound tensors."""
        if self.remaining:
            raise ValueError(f"unexpected saved tensors: {sorted(self.remaining)}")


def safe_log(value: torch.Tensor) -> torch.Tensor:
    """Rebuild a safe-log cache directly in the computation dtype."""
    result = torch.empty_like(value)
    floored_log_(result, value)
    return result


def encode_initial(
    state: InitialState, index: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Record complete original groups independently of current fitted parameters.

    Feature dimensions are not recorded: they equal the fitted schema's.
    """
    prefix = f"states.{index}"
    geometry = state.input_geometry
    if geometry is None:
        raise ValueError(f"original state {index} has no input geometry")
    tensors = {f"{prefix}.continuous_centroids": geometry.continuous_centroids}
    for i, value in enumerate(geometry.categorical_centroids):
        tensors[f"{prefix}.categorical_centroids.{i}"] = value
    if geometry.feature_weights is not None:
        tensors[f"{prefix}.feature_weights"] = geometry.feature_weights
    if geometry.manifold_projectors is not None:
        tensors[f"{prefix}.manifold_projectors"] = geometry.manifold_projectors
    input_metadata = {
        "input": encode_block(geometry.input),
        "K_active": geometry.K_active,
        "captured": geometry.captured,
    }
    parameters = []
    for i, group in enumerate(state.parameters):
        key = f"{prefix}.parameters.{i}"
        metadata = {"description": encode_block(group.description)}
        if isinstance(group, _HiddenParameters | _ClassificationParameters):
            metadata["source_width"] = group.theta.shape[1]
            tensors[f"{key}.theta"] = group.theta
            if isinstance(group, _HiddenParameters):
                metadata["coupling"] = group.coupling.value
        else:
            metadata["source_width"] = group.C_y.shape[1]
            metadata["output_width"] = group.C_y.shape[0]
            metadata["has_output_weights"] = group.W_M is not None
            tensors[f"{key}.C_y"] = group.C_y
            if group.W_M is not None:
                tensors[f"{key}.W_M"] = group.W_M
        parameters.append(metadata)
    return {
        "input_geometry": input_metadata,
        "parameters": parameters,
        "connection_sub_seeds": state.connection_sub_seeds,
    }, tensors


def _decode_geometry(
    value: object, prefix: str, reader: TensorReader, schema: DataSchema
) -> InputGeometry:
    data = record(value, "input K_active captured")
    description = decode_block(data["input"])
    if not isinstance(description, InputBlock):
        raise ValueError("original geometry requires an input description")
    K = integer(data["K_active"], minimum=1)
    D, dtype = schema.D_cont, schema.computation_dtype
    return InputGeometry(
        input=description,
        continuous_centroids=reader.take(f"{prefix}.continuous_centroids", (K, D), dtype),
        categorical_centroids=tuple(
            reader.take(f"{prefix}.categorical_centroids.{i}", (K, M), dtype)
            for i, M in enumerate(schema.M_cat)
        ),
        feature_weights=(
            reader.take(f"{prefix}.feature_weights", (D + len(schema.M_cat),), dtype)
            if isinstance(description, Input)
            else None
        ),
        manifold_projectors=(
            reader.take(
                f"{prefix}.manifold_projectors", (K, D, description.subspace_dimension), dtype
            )
            if isinstance(description, ManifoldInput)
            else None
        ),
        captured=boolean(data["captured"]),
    )


def _decode_parameters(
    value: object, prefix: str, reader: TensorReader, dtype: torch.dtype
) -> _DownstreamParameters:
    if type(value) is not dict or "description" not in value:
        raise ValueError("starting parameters require a block description")
    description = decode_block(cast(dict[str, Any], value)["description"])
    if isinstance(description, Hidden):
        data = record(value, "description coupling source_width")
        return _HiddenParameters(
            description,
            Coupling(text(data["coupling"])),
            reader.take(
                f"{prefix}.theta", (description.K, integer(data["source_width"], minimum=1)), dtype
            ),
        )
    if isinstance(description, ClassificationHead):
        data = record(value, "description source_width")
        return _ClassificationParameters(
            description,
            reader.take(
                f"{prefix}.theta",
                (
                    integer(description.n_classes, minimum=1),
                    integer(data["source_width"], minimum=1),
                ),
                dtype,
            ),
        )
    if isinstance(description, RegressionHead):
        data = record(value, "description source_width output_width has_output_weights")
        M = integer(data["output_width"], minimum=1)
        return _RegressionParameters(
            description,
            reader.take(f"{prefix}.C_y", (M, integer(data["source_width"], minimum=1)), dtype),
            reader.take(f"{prefix}.W_M", (M,), dtype)
            if boolean(data["has_output_weights"])
            else None,
        )
    raise ValueError("starting parameter group must be hidden or output")


def decode_initial(
    value: object, index: int, reader: TensorReader, schema: DataSchema
) -> InitialState:
    """Restore complete typed starting groups, never row-bound continuation state."""
    prefix = f"states.{index}"
    data = record(value, "input_geometry parameters connection_sub_seeds")
    if data["input_geometry"] is None:
        raise ValueError(f"original state {index} has no input geometry")
    dtype = schema.computation_dtype
    state = InitialState(
        input_geometry=_decode_geometry(data["input_geometry"], prefix, reader, schema),
        parameters=tuple(
            _decode_parameters(group, f"{prefix}.parameters.{i}", reader, dtype)
            for i, group in enumerate(sequence(data["parameters"]))
        ),
        connection_sub_seeds=decode_seeds(data["connection_sub_seeds"]),
    )
    validate_original_state(state, schema)
    return state


def _encode_input_tensors(
    block: _InputBlock, key: str, *, resumable: bool
) -> dict[str, torch.Tensor]:
    """Return an input block's geometry and normaliser, with its instance weights when resumable."""
    tensors = {}
    if resumable:
        tensors[f"{key}.instance_weights"] = block.instance_weights
    tensors[f"{key}.continuous_centroids"] = block.continuous_centroids
    tensors[f"{key}.log_partition"] = block.log_partition
    if isinstance(block, _StandardInputBlock):
        tensors[f"{key}.feature_weights"] = block.feature_weights
        for j, value in enumerate(block.categorical_centroids):
            tensors[f"{key}.categorical_centroids.{j}"] = value
    else:
        tensors[f"{key}.manifold_projectors"] = block.manifold_projectors
    return tensors


def _encode_block(
    name: str, block: _RuntimeBlock, key: str, *, resumable: bool
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return one fitted block's metadata and tensors, with its affiliations when resumable."""
    metadata: dict[str, Any] = {"name": name}
    tensors = {}
    if isinstance(block, _ClusteringBlock):
        metadata["soft_assignments"] = block.soft_assignments
        if resumable:
            tensors[f"{key}.gamma"] = block.gamma
    match block:
        case _StandardInputBlock() | _ManifoldInputBlock():
            tensors.update(_encode_input_tensors(block, key, resumable=resumable))
        case _HiddenBlock():
            (connection,) = block.incoming.values()
            metadata["initial_widths"] = connection.initial_widths
            tensors[f"{key}.theta"] = connection.theta
        case _ClassificationBlock():
            tensors[f"{key}.theta"] = block.theta
        case _RegressionBlock():
            tensors[f"{key}.C_y"] = block.C_y
            metadata["has_output_weights"] = block.W_M is not None
            if block.W_M is not None:
                tensors[f"{key}.W_M"] = block.W_M
        case _:
            assert_never(block)
    return metadata, tensors


def encode_fitted(
    fitted: _FittedNetwork, prefix: str, initial: int, *, resumable: bool
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Collect fitted parameters, references and the requested continuation capability."""
    tensors = {}
    graph = fitted.graph
    blocks = []
    for i, (name, block) in enumerate(graph.blocks.items()):
        metadata, block_tensors = _encode_block(
            name, block, f"{prefix}.blocks.{i}", resumable=resumable
        )
        blocks.append(metadata)
        tensors.update(block_tensors)
    if fitted.Wt_ref is not None:
        tensors[f"{prefix}.Wt_ref"] = fitted.Wt_ref
    return {
        "recipe": encode_recipe(graph.recipe),
        "schema": encode_schema(fitted.schema),
        "predict_config": encode_description(fitted.predict_config),
        "epsilon_P_source": fitted.epsilon_P_source,
        "diagnostics": encode_trajectory(fitted.trajectory),
        "max_iter": fitted.max_iter,
        "tol": fitted.tol,
        "initial_state": initial,
        "training_rows": graph.training_rows,
        "connection_sub_seeds": graph.connection_sub_seeds,
        "blocks": blocks,
        "has_scoring_reference": fitted.Wt_ref is not None,
        "capability": "resumable" if resumable else "prediction",
    }, tensors


def fitted_metadata(value: object) -> dict[str, Any]:
    """Require the complete fitted record before reading any of its fields."""
    return record(
        value,
        "recipe schema predict_config epsilon_P_source diagnostics initial_state "
        "training_rows connection_sub_seeds blocks has_scoring_reference capability max_iter tol",
    )


def _decode_input(
    description: InputBlock,
    key: str,
    reader: TensorReader,
    schema: DataSchema,
    soft: bool,
    gamma: torch.Tensor,
) -> _InputBlock:
    K = dict(schema.K_active)[description.name]
    dtype, device = schema.computation_dtype, reader.device
    centroids = reader.take(f"{key}.continuous_centroids", (K, schema.D_cont), dtype)
    weights = (
        reader.take(f"{key}.instance_weights", (gamma.shape[0],), dtype)
        if gamma.shape[0]
        else torch.empty(0, dtype=dtype, device=device)
    )
    log_partition = reader.take(f"{key}.log_partition", (), dtype)
    if isinstance(description, Input):
        categories = tuple(
            reader.take(f"{key}.categorical_centroids.{j}", (K, M), dtype)
            for j, M in enumerate(schema.M_cat)
        )
        return _StandardInputBlock(
            description,
            gamma,
            reader.take(f"{key}.feature_weights", (schema.D_cont + len(schema.M_cat),), dtype),
            weights,
            centroids,
            categories,
            tuple(safe_log(c) for c in categories),
            log_partition,
            soft,
        )
    return _ManifoldInputBlock(
        description,
        centroids,
        reader.take(
            f"{key}.manifold_projectors",
            (K, schema.D_cont, description.subspace_dimension),
            dtype,
        ),
        gamma,
        weights,
        log_partition,
        soft,
    )


def _decode_hidden(
    description: Hidden,
    data: dict[str, Any],
    key: str,
    reader: TensorReader,
    schema: DataSchema,
    connection: Connection,
    soft: bool,
    gamma: torch.Tensor,
) -> _HiddenBlock:
    counts = dict(schema.K_active)
    K, dtype = counts[description.name], schema.computation_dtype
    theta = reader.take(f"{key}.theta", (K, counts[connection.source]), dtype)
    initial_widths = tuple(integer(count, minimum=1) for count in sequence(data["initial_widths"]))
    if len(initial_widths) != 2:
        raise ValueError("invalid saved connection prior cluster counts")
    # Priors retain their original capacities even when fitting pruned clusters.
    state = _ConnectionState(connection, theta, safe_log(theta), initial_widths)
    return _HiddenBlock(description, gamma, {connection.name: state}, soft)


def _decode_classification(
    description: ClassificationHead,
    key: str,
    reader: TensorReader,
    schema: DataSchema,
    source_clusters: int,
) -> _ClassificationBlock:
    theta = reader.take(f"{key}.theta", (schema.M, source_clusters), schema.computation_dtype)
    return _ClassificationBlock(description, theta, safe_log(theta))


def _decode_regression(
    description: RegressionHead,
    data: dict[str, Any],
    key: str,
    reader: TensorReader,
    schema: DataSchema,
    source_clusters: int,
) -> _RegressionBlock:
    dtype = schema.computation_dtype
    return _RegressionBlock(
        description,
        reader.take(f"{key}.C_y", (schema.M, source_clusters), dtype),
        reader.take(f"{key}.W_M", (schema.M,), dtype)
        if boolean(data["has_output_weights"])
        else None,
    )


def _decode_fitted_block(
    description: Block,
    value: object,
    key: str,
    reader: TensorReader,
    schema: DataSchema,
    built: _BuiltRecipe,
    rows: int,
    *,
    resumable: bool,
) -> _RuntimeBlock:
    fields = "name"
    if isinstance(description, InputBlock | Hidden):
        fields += " soft_assignments"
    if isinstance(description, Hidden):
        fields += " initial_widths"
    if isinstance(description, RegressionHead):
        fields += " has_output_weights"
    data = record(value, fields)
    if data["name"] != description.name:
        raise ValueError("saved block identity does not match the Recipe")
    soft = False
    if isinstance(description, InputBlock | Hidden):
        soft = boolean(data["soft_assignments"])
        if soft != _is_soft(description.epsilon, schema.computation_dtype):
            raise ValueError("saved affiliation regime is inconsistent with the fitted model")
        K = dict(schema.K_active)[description.name]
        # Empty coordinates represent absent row payload, not a fitting session.
        gamma = (
            reader.take(f"{key}.gamma", (rows, K), schema.computation_dtype)
            if resumable
            else torch.empty((0, K), dtype=schema.computation_dtype, device=reader.device)
        )
    if isinstance(description, InputBlock):
        return _decode_input(description, key, reader, schema, soft, gamma)
    (connection,) = built.incoming[description.name]
    if isinstance(description, Hidden):
        return _decode_hidden(description, data, key, reader, schema, connection, soft, gamma)
    source_clusters = dict(schema.K_active)[connection.source]
    match description:
        case ClassificationHead():
            return _decode_classification(description, key, reader, schema, source_clusters)
        case RegressionHead():
            return _decode_regression(description, data, key, reader, schema, source_clusters)
        case _:
            assert_never(description)


def decode_fitted(
    value: object, prefix: str, reader: TensorReader, states: tuple[InitialState, ...]
) -> _FittedNetwork:
    """Validate and reconstruct a graph without a fitting session or dummy training rows."""
    data = fitted_metadata(value)
    if data["capability"] not in ("prediction", "resumable"):
        raise ValueError("unsupported persisted capability")
    resumable = data["capability"] == "resumable"
    built = _build_recipe(decode_recipe(data["recipe"]))
    schema = decode_schema(data["schema"])
    dtype = schema.computation_dtype
    order = tuple(block.name for block in built.topological_blocks)
    if tuple(name for name, _ in schema.K_active) != order or schema.task != built.task:
        raise ValueError("saved schema does not match Recipe graph identity")
    rows = integer(data["training_rows"], minimum=1)
    max_iter = integer(data["max_iter"], minimum=1)
    tol = number(data["tol"])
    if tol < 0:
        raise ValueError("saved tolerance must be non-negative")
    seeds = decode_seeds(data["connection_sub_seeds"])
    if tuple(name for name, _ in seeds) != tuple(c.name for c in built.connections):
        raise ValueError("saved connection seeds do not match the Recipe")
    policy = decode_description(data["predict_config"], PredictConfig)
    if _resolve_predict_config(policy, built.head) != policy:
        raise ValueError("saved prediction policy is not resolved")
    source = data["epsilon_P_source"]
    geometric = policy.output_mode == "geometric"
    if (geometric and (source not in ("supplied", "derived") or policy.epsilon_P is None)) or (
        not geometric and source is not None
    ):
        raise ValueError("invalid saved temperature provenance")
    blocks: dict[str, _RuntimeBlock] = {}
    descriptions = sequence(data["blocks"])
    if len(descriptions) != len(order):
        raise ValueError("incomplete saved block metadata")
    for i, (description, item) in enumerate(
        zip(built.topological_blocks, descriptions, strict=True)
    ):
        blocks[description.name] = _decode_fitted_block(
            description,
            item,
            f"{prefix}.blocks.{i}",
            reader,
            schema,
            built,
            rows,
            resumable=resumable,
        )
    state_index = integer(data["initial_state"])
    if state_index >= len(states):
        raise ValueError("missing original state reference")
    fitted = _FittedNetwork(
        graph=_CompiledGraph(
            built.recipe,
            order,
            built.connections,
            built.incoming,
            built.outgoing,
            blocks,
            seeds,
            rows,
        ),
        schema=schema,
        trajectory=decode_trajectory(data["diagnostics"]),
        predict_config=policy,
        epsilon_P_source=source,
        initial_state=states[state_index],
        Wt_ref=reader.take(f"{prefix}.Wt_ref", (rows,), dtype)
        if boolean(data["has_scoring_reference"])
        else None,
        max_iter=max_iter,
        tol=tol,
    )
    validate_prediction_ready_state(fitted, require_rows=resumable)
    return fitted
