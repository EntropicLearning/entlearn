"""Closed JSON descriptions for portable fitted models."""

from __future__ import annotations

import math
from dataclasses import asdict, fields
from typing import Any, cast

import torch

from entlearn._seeds import _SEED_UPPER_BOUND
from entlearn.network.config import PredictConfig
from entlearn.network.state import (
    DataSchema,
    InitOutcome,
    _SelectionHistory,
    _Trajectory,
    _winning_outcome,
)
from entlearn.recipe import (
    Block,
    ClassificationHead,
    Connection,
    Hidden,
    Input,
    ManifoldInput,
    Recipe,
    RegressionHead,
)


def record(value: object, keys: str) -> dict[str, Any]:
    """Require exactly the named fields of one JSON object."""
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError(f"expected metadata fields: {keys}")
    return cast(dict[str, Any], value)


def sequence(value: object) -> list[Any]:
    """Require a JSON array, without accepting strings or mappings."""
    if type(value) is not list:
        raise ValueError("expected a metadata array")
    return value


def integer(value: object, *, minimum: int = 0) -> int:
    """Require an integer, excluding booleans."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected an integer >= {minimum}")
    return value


def seed(value: object) -> int:
    """Require a seed in the public lifecycle's integer range."""
    result = integer(value)
    if result >= _SEED_UPPER_BOUND:
        raise ValueError("saved seed exceeds the supported integer range")
    return result


def boolean(value: object) -> bool:
    """Require a JSON boolean."""
    if type(value) is not bool:
        raise ValueError("expected a metadata boolean")
    return value


def text(value: object) -> str:
    """Require a JSON string."""
    if type(value) is not str:
        raise ValueError("expected a metadata string")
    return value


def number(value: object) -> float:
    """Require a finite JSON number."""
    if type(value) is not int and type(value) is not float:
        raise ValueError("expected a finite metadata number")
    if not math.isfinite(value):
        raise ValueError("expected a finite metadata number")
    return float(value)


_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def encode_measurement(value: float) -> float | str:
    """Represent a recorded measurement, naming a non-finite one explicitly.

    JSON has no literal for a diverged loss, so the three non-finite values are
    written as their names instead of as numbers.
    """
    return value if math.isfinite(value) else repr(value)


def measurement(value: object) -> float:
    """Require a JSON number or the name of a non-finite measurement."""
    if type(value) is str:
        if value not in _NON_FINITE:
            raise ValueError("expected a metadata number or non-finite measurement name")
        return _NON_FINITE[value]
    return number(value)


def encode_description(value: Block | Connection | PredictConfig) -> dict[str, Any]:
    """Encode a known immutable description with explicit infinite temperatures."""
    result = asdict(value)
    for name, item in result.items():
        if isinstance(item, float) and math.isinf(item):
            result[name] = {"positive_infinity": True}
    return result


def decode_description[T: (Block, Connection, PredictConfig)](value: object, cls: type[T]) -> T:
    """Decode every field and construct the description, which validates its own domains.

    A JSON object is the infinite-temperature marker and a JSON array is a tuple; every
    other value passes to the description unchanged. Block names, and whether connection
    endpoints name blocks, are validated where the description is used: by ``Recipe`` or
    by the original-state validation.
    """

    def decode(item: object) -> Any:
        if type(item) is dict:
            marker = record(item, "positive_infinity")
            if not boolean(marker["positive_infinity"]):
                raise ValueError("invalid infinite temperature")
            return math.inf
        if type(item) is list:
            return tuple(decode(element) for element in item)
        return item

    data = record(value, " ".join(field.name for field in fields(cls)))
    return cls(**{name: decode(item) for name, item in data.items()})


def encode_block(value: Block) -> dict[str, Any]:
    """Identify a block description without Python import paths."""
    return {"kind": type(value).__name__, "description": encode_description(value)}


def decode_block(value: object) -> Block:
    """Dispatch the closed block vocabulary explicitly."""
    data = record(value, "kind description")
    match data["kind"]:
        case "Input":
            return decode_description(data["description"], Input)
        case "ManifoldInput":
            return decode_description(data["description"], ManifoldInput)
        case "Hidden":
            return decode_description(data["description"], Hidden)
        case "ClassificationHead":
            return decode_description(data["description"], ClassificationHead)
        case "RegressionHead":
            return decode_description(data["description"], RegressionHead)
        case _:
            raise ValueError("unknown persisted block kind")


def encode_recipe(recipe: Recipe) -> dict[str, Any]:
    """Preserve declaration order and stable graph names."""
    return {
        "blocks": [encode_block(block) for block in recipe.blocks],
        "connections": [encode_description(connection) for connection in recipe.connections],
    }


def decode_recipe(value: object) -> Recipe:
    """Validate a complete immutable Recipe."""
    data = record(value, "blocks connections")
    return Recipe(
        blocks=tuple(decode_block(block) for block in sequence(data["blocks"])),
        connections=tuple(
            decode_description(connection, Connection)
            for connection in sequence(data["connections"])
        ),
    )


def encode_schema(schema: DataSchema) -> dict[str, Any]:
    """Represent computation dtype without execution-device provenance."""
    return {
        **asdict(schema),
        "computation_dtype": str(schema.computation_dtype).removeprefix("torch."),
    }


def decode_dtype(value: object) -> torch.dtype:
    """Require an explicitly supported saved floating dtype."""
    match value:
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64
        case _:
            raise ValueError("unsupported saved floating dtype")


def decode_schema(value: object) -> DataSchema:
    """Require feature meanings, active cluster/output counts and a supported dtype."""
    data = record(value, "task D_cont M_cat M K_active computation_dtype")
    if data["task"] not in ("classification", "regression"):
        raise ValueError("unknown persisted task")
    dtype = decode_dtype(data["computation_dtype"])
    widths = []
    for item in sequence(data["K_active"]):
        pair = sequence(item)
        if len(pair) != 2:
            raise ValueError("invalid active cluster or output count entry")
        widths.append((text(pair[0]), integer(pair[1], minimum=1)))
    schema = DataSchema(
        task=data["task"],
        D_cont=integer(data["D_cont"]),
        M_cat=tuple(integer(item, minimum=2) for item in sequence(data["M_cat"])),
        M=integer(data["M"], minimum=1),
        K_active=tuple(widths),
        computation_dtype=dtype,
    )
    if schema.D_cont + len(schema.M_cat) < 1:
        raise ValueError("the persisted input must have features")
    return schema


def decode_seeds(value: object) -> tuple[tuple[str, int], ...]:
    """Require name and supported integer seed pairs; their owners check the names."""
    pairs = []
    for item in sequence(value):
        pair = sequence(item)
        if len(pair) != 2:
            raise ValueError("invalid connection seed entry")
        pairs.append((text(pair[0]), seed(pair[1])))
    return tuple(pairs)


def encode_trajectory(trajectory: _Trajectory) -> dict[str, Any]:
    """Encode one fit's record, naming a diverged loss explicitly."""
    data = asdict(trajectory)
    data["loss_history"] = [encode_measurement(loss) for loss in trajectory.loss_history]
    return data


def decode_trajectory(value: object) -> _Trajectory:
    """Require one fit's complete loss history, iteration count, convergence and warnings."""
    data = record(value, "loss_history n_iter converged warnings")
    result = _Trajectory(
        loss_history=tuple(measurement(item) for item in sequence(data["loss_history"])),
        n_iter=integer(data["n_iter"], minimum=1),
        converged=boolean(data["converged"]),
        warnings=tuple(text(item) for item in sequence(data["warnings"])),
    )
    if len(result.loss_history) != result.n_iter + 1:
        raise ValueError("incomplete fitted diagnostics")
    return result


def encode_selection(selection: _SelectionHistory) -> dict[str, Any]:
    """Encode the selection history, naming a diverged score explicitly."""
    return {
        "initialisation_outcomes": [
            {**asdict(outcome), "score": encode_measurement(outcome.score)}
            for outcome in selection.outcomes
        ],
        "selected_index": selection.selected_index,
        "warnings": selection.warnings,
        "warnings_at": selection.warnings_at,
    }


def _decode_outcome(value: object) -> InitOutcome:
    outcome = record(value, "index seed score train_score requested_backend effective_backend")
    if outcome["requested_backend"] not in ("threads", "processes") or outcome[
        "effective_backend"
    ] not in ("serial", "threads", "processes"):
        raise ValueError("invalid initialisation outcome backend")
    return InitOutcome(
        index=integer(outcome["index"]),
        seed=seed(outcome["seed"]),
        score=measurement(outcome["score"]),
        train_score=None if outcome["train_score"] is None else number(outcome["train_score"]),
        requested_backend=outcome["requested_backend"],
        effective_backend=outcome["effective_backend"],
    )


def decode_selection(value: object) -> _SelectionHistory:
    """Require candidate records in candidate order and a winner that the ranking rule chose."""
    data = record(value, "initialisation_outcomes selected_index warnings warnings_at")
    selection = _SelectionHistory(
        tuple(_decode_outcome(item) for item in sequence(data["initialisation_outcomes"])),
        integer(data["selected_index"]),
        tuple(text(item) for item in sequence(data["warnings"])),
        integer(data["warnings_at"]),
    )
    outcomes = selection.outcomes
    if len(outcomes) > 1 and tuple(outcome.index for outcome in outcomes) != tuple(
        range(len(outcomes))
    ):
        raise ValueError("selection outcomes must retain candidate order")
    # Verify the recorded selection, never rank losses from continued models.
    if outcomes and selection.selected_index != _winning_outcome(outcomes).index:
        raise ValueError("saved winner contradicts historical selection outcomes")
    return selection
