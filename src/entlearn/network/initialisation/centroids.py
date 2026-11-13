"""Weighted k-means++ centroid selection over both input modalities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from entlearn._warnings import _warn
from entlearn.network.initialisation.balanced import allocate_balanced_seeds
from entlearn.network.validation import _FLOAT_DTYPES, _all_finite, _validate_storage
from entlearn.primitives.distance import dissimilarity_to_rows

# On CUDA, rows with at most _ROW_SCAN_LIMIT weights use a direct cumulative sum,
# avoiding extra kernel launches. Rows with more weights use the square-based scan.
_ROW_SCAN_LIMIT = 2**15


def _cumsum_rows(weights: torch.Tensor) -> torch.Tensor:
    """Return repeatable cumulative sums for non-negative ``(B, T)`` weights.

    Each row's sums never decrease and stay unchanged at zero-weight entries.
    On CPU, use the usual row-wise cumulative sum. On CUDA, scan at least two rows
    together to avoid the non-repeatable single-row kernel. For rows with more than
    ``_ROW_SCAN_LIMIT`` weights, reshape each into a padded square, scan its rows,
    then add the cumulative row totals.
    """
    B, T = weights.shape
    if weights.device.type == "cpu":
        return weights.cumsum(dim=1)
    if T <= _ROW_SCAN_LIMIT:  # CUDA sums multiple rows in a fixed order.
        return (weights.expand(2, -1) if B == 1 else weights).cumsum(dim=1)[:B]
    side = math.isqrt(T - 1) + 1
    square = weights.new_zeros(B, side * side)
    square[:, :T] = weights
    square = square.view(B, side, side)
    rows = square.cumsum(dim=2)
    rows[:, 1:] += _cumsum_rows(rows[:, :, -1])[:, :-1, None]
    # Separately computed row offsets can round below the previous row's final sum.
    # Running maxima prevent decreases within and between rows. Reset zero-weight
    # entries first so they inherit the preceding sum rather than a rounding error.
    rows = torch.where(square > 0, rows, 0).cummax(dim=2).values
    ends = rows[:, :, -1].cummax(dim=1).values  # Maxima do not depend on summation order.
    torch.maximum(rows[:, 1:], ends[:, :-1, None], out=rows[:, 1:])
    return rows.view(B, -1)[:, :T].contiguous()


def _weighted_draws(
    weights: torch.Tensor, generators: Sequence[torch.Generator], n: int
) -> torch.Tensor:
    """Draw ``n`` indices with replacement from each row of ``(B, T)`` weights.

    Selection probabilities are proportional to the weights, with at least one positive
    weight required per row. Row ``b`` uses ``generators[b]`` to draw uniform values
    and locate them in its cumulative weights. Return ``(B, n)`` indices, never
    selecting a zero-weight entry.
    """
    cumulative = _cumsum_rows(weights)
    total = cumulative[:, -1:].contiguous()
    uniform = weights.new_empty(len(generators), n)
    for row, generator in zip(uniform, generators, strict=True):
        torch.rand(n, generator=generator, out=row)
    last_weighted = torch.searchsorted(cumulative, total)
    return torch.searchsorted(cumulative, uniform.mul_(total), right=True).clamp_(max=last_weighted)


def _greedy_candidate(
    dissimilarity: Callable[[torch.Tensor], torch.Tensor],
    sample_weights: torch.Tensor,
    d2: torch.Tensor,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the candidate seed with the lowest k-means++ potential for each initialisation.

    Each of the ``B`` rows of ``candidates`` holds the proposed seeds for one
    initialisation. The first candidate wins ties. Return the selected row indices
    with shape ``(B, 1)``, the updated nearest-seed dissimilarities ``(B, T)``,
    and the dissimilarities to the winning candidate alone, also ``(B, T)``.
    """
    B, T = d2.shape
    priced = dissimilarity(candidates.view(-1)).view(T, B, -1)
    scored = torch.minimum(priced, d2.T.unsqueeze(2))
    best = (sample_weights @ scored.view(T, -1)).view(B, -1).argmin(dim=1, keepdim=True)
    index = best.view(1, B, 1).expand(T, B, 1)
    nearest = scored.gather(2, index).view(T, B)
    column = priced.gather(2, index).view(T, B)
    return candidates.gather(1, best), nearest.T.contiguous(), column.T.contiguous()


def select_kmeanspp_seeds_(
    columns: torch.Tensor | None,
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    K: int,
    generators: Sequence[torch.Generator],
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    n_candidates: int = 1,
) -> torch.Tensor:
    """Return ``K`` distinct row indices per generator, chosen by weighted k-means++ over both modalities.

    The first seed is drawn proportionally to ``sample_weights``. Each later seed is drawn
    proportionally to ``sample_weights[t] * D^2(t)``, the feature-weighted dissimilarity
    to the nearest chosen seed, with the already-chosen rows masked out. When no remaining row
    carries positive probability the next seed is drawn uniformly from the unchosen rows.
    ``n_candidates`` above one is the greedy variant: each later seed draws that many
    candidates and keeps the one minimising the potential
    ``sum_t sample_weights[t] * min_c D^2(t, c)``.

    Each generator draws a separate set of seeds. All sets share one dissimilarity
    matrix product per round. Results repeat for the same inputs and generator states,
    but changing the batch size can change floating-point rounding and hence the seeds.

    Args:
        columns: ``(len(generators), T, K)`` output (modified): column ``k`` of each
            initialisation receives every row's dissimilarity to its seed ``k``.
            ``None`` skips it.
        X: ``(T, D_cont)`` continuous data.
        X_cat: Per-feature ``(T, M_i)`` categorical distributions or ``(T,)`` integer
            codes; empty when there is no categorical modality.
        K: Number of seeds to select; at most ``T``.
        generators: Generators driving the draws, one per set of seeds.
        Wd_cont: ``(D_cont,)`` continuous feature weights.
        Wd_cat: ``(D_cat,)`` categorical feature weights.
        sample_weights: ``(T,)`` per-instance weights.
        n_candidates: Candidate draws per seed; ``1`` is plain k-means++.

    Returns:
        ``(len(generators), K)`` int64 row indices, with no repeated index within a row.
    """
    dissimilarity = dissimilarity_to_rows(X, X_cat, Wd_cont, Wd_cat)
    chosen = [_weighted_draws(sample_weights.expand(len(generators), -1), generators, 1)]
    d2 = dissimilarity(chosen[0].view(-1)).T.contiguous()
    if columns is not None:
        columns[:, :, 0] = d2
    unchosen = torch.ones_like(d2).scatter_(1, chosen[0], 0.0)
    for count in range(1, K):
        probabilities = sample_weights * d2 * unchosen
        has_mass = probabilities.any(dim=1, keepdim=True)
        candidates = _weighted_draws(
            torch.where(has_mass, probabilities, unchosen), generators, n_candidates
        )
        if n_candidates == 1:
            column = dissimilarity(candidates.view(-1)).T
            torch.minimum(d2, column, out=d2)
        else:
            candidates, d2, column = _greedy_candidate(
                dissimilarity, sample_weights, d2, candidates
            )
        if columns is not None:
            columns[:, :, count] = column
        unchosen.scatter_(1, candidates, 0.0)
        chosen.append(candidates)
    return torch.cat(chosen, dim=1)


def cap_input_clusters_to_eligible_rows(K: int, labelled: torch.Tensor, *, balanced: bool) -> int:
    """Limit the input cluster count to the number of rows available for centroid seeding.

    Each centroid needs a distinct row. All staged rows are eligible unless ``balanced``
    is true, in which case only rows marked in ``labelled`` are eligible.
    Return the smaller of ``K`` and the eligible count, warning if ``K`` is reduced.
    Raise ``ValueError`` if no row is eligible.

    Call once before seeding any initialisations so a fit emits at most one cap warning.
    """
    available = int(labelled.sum()) if balanced else labelled.numel()
    if available == 0:
        raise ValueError("input initialisation requires at least one eligible row")
    if available < K:
        _warn(
            f"input K={K} exceeds {available} eligible rows "
            f"from {labelled.numel()} samples; reducing the initial input cluster count to "
            f"{available}",
            UserWarning,
        )
    return min(K, available)


def select_centroid_seeds(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    K: int,
    generators: Sequence[torch.Generator],
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    sample_weights: torch.Tensor,
    target: torch.Tensor,
    labelled: torch.Tensor,
    *,
    balanced: bool,
    n_candidates: int | None,
) -> torch.Tensor:
    """Select ``K`` distinct rows per generator, from the labelled rows when ``balanced``.

    The caller must cap ``K`` at the eligible row count before calling.
    ``n_candidates=None`` uses the greedy default of ``2 + floor(log(K))``.
    """
    if n_candidates is None:
        n_candidates = 2 + math.floor(math.log(K))
    if not balanced:  # every row is eligible
        return select_kmeanspp_seeds_(
            None,
            X,
            X_cat,
            K,
            generators,
            Wd_cont,
            Wd_cat,
            sample_weights,
            n_candidates=n_candidates,
        )
    labels = target.argmax(dim=1)
    allocation = allocate_balanced_seeds(labels, labelled, sample_weights, K)
    selected_pools = []
    for code, count in allocation:
        pool = labelled & (labels == code)
        pool_rows = pool.nonzero(as_tuple=True)[0]
        selected = select_kmeanspp_seeds_(
            None,
            X[pool],
            [feature[pool] for feature in X_cat],
            count,
            generators,
            Wd_cont,
            Wd_cat,
            sample_weights[pool],
            n_candidates=n_candidates,
        )
        selected_pools.append(pool_rows[selected])
    return torch.cat(selected_pools, dim=1)


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
    if not _all_finite(result):
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
        if not _all_finite(row_sums) or bool((row_sums <= 0).any()):
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
