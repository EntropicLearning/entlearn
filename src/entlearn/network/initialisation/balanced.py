"""Balanced classification seed allocation."""

from __future__ import annotations

import torch


def allocate_balanced_seeds(
    labels: torch.Tensor,
    eligible: torch.Tensor,
    sample_weights: torch.Tensor,
    K: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(class code, seed count)`` pairs in class-code order.

    ``eligible`` marks the labelled rows, and ``K`` must not exceed their number.
    """
    active_classes = sorted(set(labels[eligible].tolist()))
    if len(active_classes) > K:
        raise ValueError(
            f"balanced input K={K} is smaller than {len(active_classes)} active classes"
        )

    capacities = {code: int((eligible & (labels == code)).sum()) for code in active_classes}
    masses = {
        code: float(sample_weights[eligible & (labels == code)].sum()) for code in active_classes
    }
    counts = {code: 1 for code in active_classes}
    order = sorted(active_classes, key=lambda code: (-masses[code], code))
    remaining = K - len(active_classes)
    # That precondition makes the total capacity at least K, so a pass over `order`
    # always places at least one seed and the loop terminates.
    while remaining:
        for code in order:
            if counts[code] == capacities[code]:
                continue
            counts[code] += 1
            remaining -= 1
            if remaining == 0:
                break
    return tuple((code, counts[code]) for code in active_classes)
