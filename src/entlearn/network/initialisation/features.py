"""Standard-input feature-weight initialisation."""

from __future__ import annotations

import torch

from entlearn.network.validation import _normalise_weights, _stage_weights


def initialise_feature_weights(
    width: int,
    W_std: float,
    supplied: torch.Tensor | None,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return copied simplex feature weights for an input block."""
    if supplied is not None:
        weights = _stage_weights(
            supplied,
            name="feature_weights",
            length=width,
            dtype=dtype,
            device=device,
        )
        return _normalise_weights(weights, name="feature_weights")
    if W_std == 0.0:
        return torch.full((width,), 1.0 / width, dtype=dtype, device=device)
    logits = W_std * torch.randn(width, generator=generator, dtype=dtype, device=device)
    return torch.softmax(logits, dim=0)
