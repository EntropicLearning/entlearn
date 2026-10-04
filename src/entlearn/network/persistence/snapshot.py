"""One fitted-state schema and decoder for file and Python transport."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import torch

from entlearn.network.blocks.types import _ClusteringBlock
from entlearn.network.persistence.metadata import (
    decode_schema,
    decode_selection,
    encode_selection,
    record,
    sequence,
    text,
)
from entlearn.network.persistence.tensors import (
    TensorReader,
    decode_fitted,
    decode_initial,
    encode_fitted,
    encode_initial,
    fitted_metadata,
)
from entlearn.network.state import _NetworkRecord
from entlearn.network.state_validation import validate_original_bindings

if TYPE_CHECKING:
    from entlearn.network.state import DataSchema, InitialState, _FittedNetwork

_FORMAT = "entlearn.network"
# This versions the saved representation, independently of package releases.
_FORMAT_VERSION = 1


def _has_rows(fitted: _FittedNetwork) -> bool:
    """Detect retained coordinates, not validity, decoding validates them."""
    return bool(fitted.graph.input.instance_weights.numel()) or any(
        block.gamma.numel()
        for block in fitted.graph.blocks.values()
        if isinstance(block, _ClusteringBlock)
    )


@torch.inference_mode()
def encode_snapshot(
    record: _NetworkRecord, *, resumable: bool | None = None
) -> tuple[str, dict[str, torch.Tensor]]:
    """Collect metadata and tensor references without copying their storage.

    ``None`` preserves existing row state for transport. Explicit booleans select
    the file-export capability, and ``True`` raises for a model without retained
    rows. Encoding does not validate the published model again; decoding does.
    Callers must serialise the snapshot before mutation.
    """
    fitted, initial_states, members = record.fitted, record.states, record.members
    if resumable is None:
        resumable = _has_rows(fitted)
        if any(_has_rows(member) != resumable for member in members or ()):
            raise RuntimeError("retained members disagree on continuation capability")
    tensors: dict[str, torch.Tensor] = {}
    originals = initial_states or (fitted.initial_state,)
    winner = record.winner
    candidates = members or (fitted,)
    states, models = [], []
    for i, original in enumerate(originals):
        description, values = encode_initial(original, i)
        states.append(description)
        tensors.update(values)
    for i, candidate in enumerate(candidates):
        if resumable and not _has_rows(candidate):
            raise ValueError(
                "Network has no valid resumable row-bound state: no training rows are retained"
            )
        description, values = encode_fitted(
            candidate,
            f"models.{i}",
            i if members is not None else winner,
            resumable=resumable,
        )
        models.append(description)
        tensors.update(values)
    metadata = {
        "format": _FORMAT,
        "version": _FORMAT_VERSION,
        "package_version": version("entlearn"),
        "retention": (
            "members"
            if members is not None
            else "states"
            if initial_states is not None
            else "winner"
        ),
        "selection": encode_selection(record.selection),
        "states": states,
        "models": models,
    }
    return json.dumps(metadata, allow_nan=False), tensors


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate metadata field {name!r}")
        result[name] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r}")


def _resolve_load_device(value: str | torch.device) -> torch.device:
    """Resolve supported placement to the device identity reported by loaded tensors."""
    if not isinstance(value, str | torch.device):
        raise ValueError("load device must be a string or torch.device")
    try:
        device = torch.device(value)
    except RuntimeError as error:
        raise ValueError("invalid load device") from error
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("unsupported load device")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError("requested CUDA device is unavailable")
    if device.type == "mps" and (
        not torch.backends.mps.is_available() or device.index not in (None, 0)
    ):
        raise ValueError("requested MPS device is unavailable")
    # Tensor validation here compares device identities only.
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type == "cuda":
        return torch.device(
            "cuda", torch.cuda.current_device() if device.index is None else device.index
        )
    return torch.device("mps", 0)


def _check_retention(retention: object, models: list[Any], originals: list[Any]) -> None:
    """Require a known retention mode and the model and original-state counts it implies."""
    if not models or not originals:
        raise ValueError("missing fitted model or original state")
    if retention not in ("winner", "states", "members"):
        raise ValueError("unknown retention mode")
    if (retention == "winner" and len(originals) != 1) or (
        len(models) != (len(originals) if retention == "members" else 1)
    ):
        raise ValueError("inconsistent retention collection sizes")


def _check_members(
    members: tuple[_FittedNetwork, ...],
    models: list[Any],
    states: tuple[InitialState, ...],
    schema: DataSchema,
    *,
    capability: object,
    retention: object,
    winner: int,
) -> None:
    """Require members to share capability, Recipe, schema and rows, each bound to its own state."""
    for i, member in enumerate(members):
        if fitted_metadata(models[i])["capability"] != capability:
            raise ValueError("retained members disagree on the saved capability")
        expected = states[i if retention == "members" else winner]
        if member.initial_state is not expected:
            raise ValueError("inconsistent member original-state identity")
        if (
            member.graph.recipe != members[0].graph.recipe
            or replace(member.schema, K_active=schema.K_active) != schema
        ):
            raise ValueError("retained members disagree on the Recipe or feature schema")
        if member.graph.training_rows != members[0].graph.training_rows:
            raise ValueError("retained members disagree on the training row count")


@torch.inference_mode()
def decode_snapshot(
    metadata: str, tensors: dict[str, torch.Tensor], *, device: str | torch.device
) -> _NetworkRecord:
    """Consume a snapshot and reconstruct owned inference storage once.

    The tensor map is consumed as each value is transferred, so incoming storage
    can be released promptly. No caller-owned graph or private record is restored.
    """
    placement = _resolve_load_device(device)
    value = json.loads(metadata, object_pairs_hook=_object, parse_constant=_constant)
    data = record(value, "format version package_version retention selection states models")
    # Producer provenance is diagnostic only.
    text(data["package_version"])
    if (
        data["format"] != _FORMAT
        or type(data["version"]) is not int
        or data["version"] != _FORMAT_VERSION
    ):
        raise ValueError("unsupported Network persistence format or version")
    reader = TensorReader(tensors, placement)
    models, originals = sequence(data["models"]), sequence(data["states"])
    retention = data["retention"]
    _check_retention(retention, models, originals)
    selection = decode_selection(data["selection"])
    if retention != "winner" and len(selection.outcomes) != len(originals):
        raise ValueError("original states do not match the candidate outcomes")
    winner = 0 if retention == "winner" else selection.selected_index
    if winner >= len(originals):
        raise ValueError("missing winning original state")
    first_model = fitted_metadata(models[0])
    schema = decode_schema(first_model["schema"])
    if placement.type == "mps" and schema.computation_dtype != torch.float32:
        raise ValueError("MPS loading requires a saved float32 computation dtype")
    states = tuple(decode_initial(state, i, reader, schema) for i, state in enumerate(originals))
    members = tuple(
        decode_fitted(model, f"models.{i}", reader, states) for i, model in enumerate(models)
    )
    _check_members(
        members,
        models,
        states,
        schema,
        capability=first_model["capability"],
        retention=retention,
        winner=winner,
    )
    for state in states:
        validate_original_bindings(state, members[0].graph, schema)
    reader.finish()
    return _NetworkRecord(
        members[winner if retention == "members" else 0],
        selection,
        states if retention != "winner" else None,
        members if retention == "members" else None,
    )
