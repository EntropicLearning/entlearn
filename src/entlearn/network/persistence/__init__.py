"""File and Python envelopes around the canonical Network snapshot."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from entlearn.network.finalise import complete
from entlearn.network.persistence.snapshot import _FORMAT, decode_snapshot, encode_snapshot
from entlearn.network.state import _NetworkRecord, _SelectionHistory

if TYPE_CHECKING:
    from entlearn.network.data import _StagedData
    from entlearn.network.model import Network
    from entlearn.network.selection import _CandidateEvaluation


def _reduce_network(network: Network) -> tuple[Callable[..., Network], tuple[object, ...]]:
    """Carry a snapshot through pickle, without a file-sized bytes buffer."""
    metadata, tensors = encode_snapshot(network._record())
    return _restore_network, (metadata, tensors, str(network.device))


def _restore_network(metadata: str, tensors: dict[str, torch.Tensor], device: str) -> Network:
    """Restore Python transport through the same decoder and publication as a portable file."""
    from entlearn.network.model import Network

    return Network._publish(decode_snapshot(metadata, tensors, device=device))


@torch.inference_mode()
def _evaluate_for_transport(
    evaluate: Callable[[int], _CandidateEvaluation], data: _StagedData, seed: int
) -> _CandidateEvaluation:
    """Evaluate a candidate in a process worker and complete its fit there.

    The snapshot carries only a network that is ready to predict.
    """
    evaluation = evaluate(seed)
    if evaluation.fitted is None:
        return evaluation
    fitted, _ = complete(evaluation.fitted, data)
    return replace(evaluation, fitted=fitted)


def _reduce_candidate(
    evaluation: _CandidateEvaluation,
) -> tuple[Callable[..., _CandidateEvaluation], tuple[object, ...]]:
    """Carry a worker's evaluation, its fit as a snapshot with no selection history yet."""
    fitted = evaluation.fitted
    if fitted is None:
        return type(evaluation), (
            evaluation.initial_state,
            evaluation.trajectories,
            None,
            evaluation.score,
            evaluation.train_score,
        )
    metadata, tensors = encode_snapshot(_NetworkRecord(fitted, _SelectionHistory(())))
    device = str(fitted.graph.input.continuous_centroids.device)
    return _restore_candidate, (metadata, tensors, device, evaluation.score)


def _restore_candidate(
    metadata: str, tensors: dict[str, torch.Tensor], device: str, score: float
) -> _CandidateEvaluation:
    """Reconstruct a worker's in-sample evaluation through the common decoder."""
    from entlearn.network.selection import _CandidateEvaluation

    fitted = decode_snapshot(metadata, tensors, device=device).fitted
    return _CandidateEvaluation(fitted.initial_state, (fitted.trajectory,), fitted, score)


def _path(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    if result.suffix != ".safetensors":
        raise ValueError("Network persistence requires a .safetensors path")
    return result


def save(network: Network, path: str | os.PathLike[str], *, resumable: bool) -> None:
    """Write the canonical representation in a single safetensors container."""
    if type(resumable) is not bool:
        raise ValueError("resumable must be a boolean")
    destination = _path(path)
    metadata, tensors = encode_snapshot(network._record(), resumable=resumable)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        destination,
        metadata={_FORMAT: metadata},
    )


def load(path: str | os.PathLike[str], *, device: str | torch.device) -> _NetworkRecord:
    """Read a container and delegate all model reconstruction to the common decoder."""
    try:
        with safe_open(_path(path), framework="pt", device="cpu") as bundle:
            metadata = bundle.metadata() or {}
            if _FORMAT not in metadata:
                raise ValueError("missing Network persistence metadata")
            names = bundle.keys()
            tensors = {name: bundle.get_tensor(name) for name in names}
            return decode_snapshot(metadata[_FORMAT], tensors, device=device)
    except SafetensorError as error:
        raise ValueError("invalid safetensors bundle") from error
