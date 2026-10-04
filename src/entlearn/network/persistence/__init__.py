"""File and Python envelopes around the canonical Network snapshot."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from entlearn.network.persistence.snapshot import _FORMAT, decode_snapshot, encode_snapshot

if TYPE_CHECKING:
    from entlearn.network.model import Network
    from entlearn.network.state import _NetworkRecord


def _reduce_network(network: Network) -> tuple[Callable[..., Network], tuple[object, ...]]:
    """Carry a snapshot through pickle, without a file-sized bytes buffer."""
    metadata, tensors = encode_snapshot(network._record())
    return _restore_network, (metadata, tensors, str(network.device))


def _restore_network(metadata: str, tensors: dict[str, torch.Tensor], device: str) -> Network:
    """Restore Python transport through the same decoder and publication as a portable file."""
    from entlearn.network.model import Network

    return Network._publish(decode_snapshot(metadata, tensors, device=device))


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
