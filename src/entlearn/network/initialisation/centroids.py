"""Weighted k-means++ centroid selection over both input modalities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from entlearn._warnings import _warn
from entlearn.network.initialisation.balanced import allocate_balanced_seeds
from entlearn.network.validation import _FLOAT_DTYPES, _validate_storage
from entlearn.primitives.dissimilarity import seed_dissimilarity
from entlearn.primitives.normalise import _eps


def _weighted_choice(weights: torch.Tensor, generator: torch.Generator, n: int = 1) -> torch.Tensor:
    """Draw ``n`` indices with probability proportional to ``weights``, with replacement.

    An exact inverse-CDF draw: a uniform in ``[0, 1)`` is scaled by the cumulative total
    and located by ``searchsorted``, so a zero-weight entry cannot be selected. At least
    one weight must be positive.
    """
    cumulative = torch.cumsum(weights, dim=0)
    total = cumulative[-1]
    draw = torch.rand(n, generator=generator, device=weights.device, dtype=weights.dtype)
    last_weighted = torch.searchsorted(cumulative, total)
    return torch.searchsorted(cumulative, draw * total, right=True).clamp_(max=last_weighted)


def _uniform_unchosen_pick(
    chosen: torch.Tensor,
    count: int,
    n_rows: int,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> int:
    """Pick one not-yet-chosen row uniformly at random.

    The fallback for a degenerate weighted draw: when no remaining candidate carries
    positive probability, every unchosen row becomes equally likely.
    """
    mask = torch.ones(n_rows, dtype=torch.bool, device=chosen.device)
    mask[chosen[:count]] = False
    remaining = mask.nonzero(as_tuple=True)[0]
    pick = int(
        torch.multinomial(
            torch.ones(remaining.numel(), dtype=dtype, device=chosen.device), 1, generator=generator
        ).item()
    )
    return int(remaining[pick].item())


def _greedy_candidate(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    logX_cat: list[torch.Tensor],
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    sample_weights: torch.Tensor,
    d2: torch.Tensor,
    candidates: torch.Tensor,
) -> int:
    """Return the candidate minimising the k-means++ potential, the first wins a tie."""
    best_potential = math.inf
    best = int(candidates[0].item())
    for candidate in candidates.tolist():
        d2_candidate = torch.minimum(
            d2, seed_dissimilarity(X, X_cat, logX_cat, Wd_cont, Wd_cat, candidate)
        )
        potential = float((sample_weights * d2_candidate).sum())
        if potential < best_potential:
            best_potential = potential
            best = candidate
    return best


def select_kmeanspp_seeds(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    K: int,
    generator: torch.Generator,
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    n_candidates: int = 1,
) -> torch.Tensor:
    """Return ``K`` distinct row indices chosen by weighted k-means++ over both modalities.

    The first seed is drawn proportionally to ``sample_weights``. Each later seed is drawn
    proportionally to ``sample_weights[t] * D^2(t)``, the feature-weighted dissimilarity
    to the nearest chosen seed, with the already-chosen rows masked out. When no remaining row
    carries positive probability the next seed is drawn uniformly from the unchosen rows.
    ``n_candidates`` above one is the greedy variant: each later seed draws that many
    candidates and keeps the one minimising the potential
    ``sum_t sample_weights[t] * min_c D^2(t, c)``.

    Args:
        X: ``(T, D_cont)`` continuous data.
        X_cat: Per-feature ``(T, M_i)`` categorical distributions; empty when there is
            no categorical modality.
        K: Number of seeds to select; at most ``T``.
        generator: Generator driving the draws.
        Wd_cont: ``(D_cont,)`` continuous feature weights.
        Wd_cat: ``(D_cat,)`` categorical feature weights.
        sample_weights: ``(T,)`` per-instance weights.
        n_candidates: Candidate draws per seed; ``1`` is plain k-means++.

    Returns:
        ``(K,)`` int64 row indices, distinct by construction.
    """
    log_eps = _eps(X.dtype)
    logX_cat = [rows.clamp_min(log_eps).log() for rows in X_cat]
    chosen = torch.empty(K, dtype=torch.int64, device=X.device)
    first = int(_weighted_choice(sample_weights, generator).item())
    chosen[0] = first
    d2 = seed_dissimilarity(X, X_cat, logX_cat, Wd_cont, Wd_cat, first)
    for count in range(1, K):
        probabilities = sample_weights * d2
        probabilities[chosen[:count]] = 0.0
        if bool(probabilities.sum() > 0):
            candidates = _weighted_choice(probabilities, generator, n_candidates)
            if n_candidates == 1:
                nxt = int(candidates.item())
            else:
                nxt = _greedy_candidate(
                    X, X_cat, logX_cat, Wd_cont, Wd_cat, sample_weights, d2, candidates
                )
        else:
            nxt = _uniform_unchosen_pick(chosen, count, X.shape[0], generator, X.dtype)
        chosen[count] = nxt
        d2 = torch.minimum(d2, seed_dissimilarity(X, X_cat, logX_cat, Wd_cont, Wd_cat, nxt))
    return chosen


def select_centroid_seeds(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    K: int,
    generator: torch.Generator,
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    sample_weights: torch.Tensor,
    target: torch.Tensor,
    labelled: torch.Tensor,
    *,
    balanced: bool,
    n_candidates: int | None,
) -> torch.Tensor:
    """Select distinct eligible rows, warning when the requested count is reduced.

    Balanced selection uses only labelled rows. ``n_candidates=None`` selects the greedy
    default from the feasible, rather than requested, cluster count.
    """
    eligible = labelled if balanced else torch.ones_like(labelled)
    eligible_rows = eligible.nonzero(as_tuple=True)[0]
    available = eligible_rows.numel()
    if available == 0:
        raise ValueError("input initialisation requires at least one eligible row")
    if available < K:
        _warn(
            f"input K={K} exceeds {available} eligible rows "
            f"from {X.shape[0]} samples; reducing the initial input cluster count to {available}",
            UserWarning,
        )
        K = available
    if n_candidates is None:
        n_candidates = 2 + math.floor(math.log(K))
    if balanced:
        labels = target.argmax(dim=1)
        allocation = allocate_balanced_seeds(labels, eligible, sample_weights, K)
        pools = tuple((eligible & (labels == code), count) for code, count in allocation)
    else:
        pools = ((eligible, K),)

    selected_pools = []
    for pool, count in pools:
        pool_rows = pool.nonzero(as_tuple=True)[0]
        selected = select_kmeanspp_seeds(
            X[pool],
            [feature[pool] for feature in X_cat],
            count,
            generator,
            Wd_cont,
            Wd_cat,
            sample_weights[pool],
            n_candidates=n_candidates,
        )
        selected_pools.append(pool_rows[selected])
    return torch.cat(selected_pools)


def _staged_centroids(
    values: object,
    name: str,
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    allow_conversion: bool,
) -> torch.Tensor:
    """Require a finite floating ``shape`` tensor on ``device``; convert it to ``dtype``."""
    if not isinstance(values, torch.Tensor):
        raise ValueError(f"{name} must be a floating tensor")
    _validate_storage(values, name=name)
    if values.shape != shape or values.dtype not in _FLOAT_DTYPES or values.device != device:
        raise ValueError(
            f"{name} must be a floating tensor with shape {shape} on the X_cont device"
        )
    if not allow_conversion and values.dtype is not dtype:
        raise ValueError(f"{name} must use the computation dtype when computation_dtype is omitted")
    result = values.to(dtype=dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _staged_categorical_centroids(
    categorical: object,
    *,
    K: int,
    M_cat: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    allow_conversion: bool,
) -> tuple[torch.Tensor, ...]:
    """Check one ``(K, cardinality)`` distribution per categorical feature, normalising each row."""
    if isinstance(categorical, (str, bytes)) or not isinstance(categorical, Sequence):
        raise ValueError("categorical_centroids must be a sequence of tensors")
    if len(categorical) != len(M_cat):
        raise ValueError(
            f"categorical_centroids has {len(categorical)} feature(s), "
            f"but the input has {len(M_cat)}"
        )
    resolved = []
    for position, (values, cardinality) in enumerate(zip(categorical, M_cat, strict=True)):
        name = f"categorical_centroids[{position}]"
        distributions = _staged_centroids(
            values,
            name,
            (K, cardinality),
            dtype=dtype,
            device=device,
            allow_conversion=allow_conversion,
        )
        if bool((distributions < 0).any()):
            raise ValueError(f"{name} must be non-negative")
        row_sums = distributions.sum(dim=1)
        if not bool(torch.isfinite(row_sums).all()) or bool((row_sums <= 0).any()):
            raise ValueError(f"{name} rows must have finite positive mass")
        if not torch.allclose(row_sums, torch.ones_like(row_sums)):
            _warn(f"{name} rows do not sum to one; normalising each row", UserWarning)
        resolved.append((distributions / row_sums[:, None]).detach())
    return tuple(resolved)


def stage_supplied_centroids(
    continuous: torch.Tensor | None,
    categorical: Sequence[torch.Tensor] | None,
    *,
    K: int,
    D_cont: int,
    M_cat: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    allow_conversion: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
    """Validate and copy caller-supplied centroid geometry."""
    continuous_given = continuous is not None
    categorical_given = categorical is not None
    if not continuous_given and not categorical_given:
        return None
    if continuous_given and D_cont == 0:
        raise ValueError("continuous_centroids were supplied for an input without continuous data")
    if categorical_given and not M_cat:
        raise ValueError(
            "categorical_centroids were supplied for an input without categorical data"
        )
    if D_cont > 0 and M_cat and continuous_given != categorical_given:
        raise ValueError(
            "supplied centroid geometry for mixed input requires both "
            "continuous_centroids and categorical_centroids"
        )

    if continuous_given:
        resolved_continuous = (
            _staged_centroids(
                continuous,
                "continuous_centroids",
                (K, D_cont),
                dtype=dtype,
                device=device,
                allow_conversion=allow_conversion,
            )
            .detach()
            .clone()
        )
    else:
        resolved_continuous = torch.empty(K, 0, dtype=dtype, device=device)
    if not categorical_given:
        return resolved_continuous, ()
    return resolved_continuous, _staged_categorical_centroids(
        categorical, K=K, M_cat=M_cat, dtype=dtype, device=device, allow_conversion=allow_conversion
    )
