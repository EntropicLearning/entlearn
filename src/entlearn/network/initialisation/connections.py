"""Private automatic initialisation of clustering connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from entlearn._warnings import _warn
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.manifold import _ManifoldInputBlock
from entlearn.network.blocks.types import _ClusteringBlock
from entlearn.network.data import _ClassificationSupervision
from entlearn.network.initialisation.centroids import select_kmeanspp_seeds_
from entlearn.network.state import _DownstreamParameters, _HiddenParameters
from entlearn.primitives.encoding import densify_categorical, split_wd
from entlearn.primitives.geometry import _finalise_group_means_, grouped_means
from entlearn.primitives.normalise import _eps, _tiny, floored_log_, normalise_
from entlearn.primitives.statistics import entropy
from entlearn.recipe import Coupling

if TYPE_CHECKING:
    from entlearn.network.blocks.types import _InputBlock
    from entlearn.network.connections import _ConnectionState
    from entlearn.network.session import _FitSession

# Geometric draws a label-aware seed chooses among. The seed whose propagated affiliations
# give the head the lowest loss is kept.
_SEED_CANDIDATES = 8
# Weight of the merge's geometric cost against its label cost: a tie-break into the last
# hidden block, an equal partner into an earlier one, whose groups a later merge regroups.
_GEOMETRY_TIE_BREAK = 1e-3


@dataclass(frozen=True)
class _InitialisationGeometry:
    """Temporary representatives in the original input space for downstream seeding.

    Group masses accompany these representatives; they are not recomputed from the
    propagated hidden affiliations.
    """

    continuous_centroids: torch.Tensor
    categorical_centroids: tuple[torch.Tensor, ...]
    group_masses: torch.Tensor
    feature_weights: torch.Tensor


def _input_geometry(block: _InputBlock, session: _FitSession) -> _InitialisationGeometry:
    """Return the seeded input geometry and weighted cluster masses."""
    row_weights = session.data.sample_weights
    masses = (row_weights.unsqueeze(1) * block.gamma).sum(dim=0)
    if isinstance(block, _ManifoldInputBlock):
        D = block.continuous_centroids.shape[1]
        return _InitialisationGeometry(
            continuous_centroids=block.continuous_centroids,
            categorical_centroids=(),
            group_masses=masses,
            feature_weights=torch.full(
                (D,), 1 / D, dtype=block.gamma.dtype, device=block.gamma.device
            ),
        )
    return _InitialisationGeometry(
        continuous_centroids=block.continuous_centroids,
        categorical_centroids=block.categorical_centroids,
        group_masses=masses,
        feature_weights=block.feature_weights,
    )


def _spawn_generators(generator: torch.Generator, n: int) -> list[torch.Generator]:
    """Return ``n`` generators seeded from ``generator``'s next draws."""
    seeds = torch.randint(2**62, (n,), generator=generator, device=generator.device)
    return [torch.Generator(device=generator.device).manual_seed(seed) for seed in seeds.tolist()]


def _group_source(
    geometry: _InitialisationGeometry,
    K_target: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Draw one grouping of source clusters per generator.

    Select ``K_target`` representative clusters per generator in a shared k-means++ call.
    Assign each source cluster to its nearest representative, keeping each representative
    in its own group. Return group labels with shape ``(len(generators), K_source)``.
    """
    K_source, D_cont = geometry.continuous_centroids.shape
    device = geometry.continuous_centroids.device
    if K_target == K_source:
        return torch.arange(K_source, device=device).expand(len(generators), -1)
    Wd_cont, Wd_cat = split_wd(geometry.feature_weights, D_cont)
    dissimilarities = geometry.continuous_centroids.new_empty(len(generators), K_source, K_target)
    seeds = select_kmeanspp_seeds_(
        dissimilarities,
        geometry.continuous_centroids,
        list(geometry.categorical_centroids),
        K_target,
        generators,
        Wd_cont,
        Wd_cat,
        geometry.group_masses,
    )
    labels = dissimilarities.argmin(dim=2)
    return labels.scatter_(1, seeds, torch.arange(K_target, device=device).expand_as(seeds))


def _label_rows(session: _FitSession) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return each row's label sums and labelled mass, and whether the head classifies.

    Affiliations ``gamma`` give their clusters the label sums ``gamma.T @ rows`` and the
    labelled masses ``gamma.T @ masses``. Classification rows are the weighted target, so
    the sums are weighted class counts. Regression rows are the row weight times the target,
    centred, with each output weighted as the head weights it and the whole scaled to unit
    weighted variance, so a merge costs the same at any target scale. Unlabelled rows are
    zero.
    """
    data = session.data
    supervision = data.supervision
    if isinstance(supervision, _ClassificationSupervision):
        rows = supervision.weighted_target
        return rows, rows.sum(dim=1), True
    weights = supervision.row_weights
    output_weights = supervision.output_weights
    if output_weights is None:
        output_weights = torch.full_like(supervision.target_mean, 1.0 / data.target.shape[1])
    centred = data.target - supervision.target_mean
    variance = (weights @ centred.square()) @ output_weights / weights.sum()
    second_moment = (weights @ data.target.square()) @ output_weights / weights.sum()
    # Relative precision of the spread, then a range guard for all-zero targets
    floor = (_eps(variance.dtype) * second_moment).clamp_min(_tiny(variance.dtype))
    scale = (output_weights / torch.maximum(variance, floor)).sqrt()
    return weights[:, None] * centred * scale, weights, False


def _label_entropies(sums: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """Return each cluster's labelled mass times the entropy of its label distribution."""
    probabilities = sums / masses.clamp_min(_tiny(masses.dtype)).unsqueeze(-1)
    return masses * entropy(probabilities)


def _head_loss(sums: torch.Tensor, masses: torch.Tensor, classification: bool) -> torch.Tensor:
    """Return the head's loss for clusters with these label sums, up to a constant.

    Classification gives the mass-weighted entropy of each cluster's labels. Regression
    gives the within-cluster sum of squares of the target, less its total sum of squares.
    """
    if classification:
        return _label_entropies(sums, masses).sum()
    # Guard zero mass without flooring small weights at machine precision.
    return -(sums.square().sum(dim=-1) / masses.clamp_min(_tiny(masses.dtype))).sum()


def _group_by_head_loss(
    sums: torch.Tensor,
    masses: torch.Tensor,
    points: torch.Tensor,
    point_masses: torch.Tensor,
    K_target: int,
    classification: bool,
    geometry_weight: float,
) -> torch.Tensor:
    """Assign each source cluster to one of ``K_target`` groups by greedy pairwise merging.

    Each step merges the two groups whose union raises the head's loss least. The cost of a
    merge is that rise, plus ``geometry_weight`` times the rise in the mass-weighted squared
    distance of the groups' centroids to their group mean. Only the merged group's
    costs are recomputed, so a step costs ``O(K_source^2)``.

    Args:
        sums: ``(K_source, L)`` label sums of the source clusters.
        masses: ``(K_source,)`` labelled masses of the source clusters.
        points: ``(K_source, F)`` source-cluster centroids, scaled by root feature weights.
        point_masses: ``(K_source,)`` masses of the source clusters.
        K_target: Number of groups.
        classification: Whether the head classifies.
        geometry_weight: Weight of the geometric cost.

    Returns:
        ``(K_source,)`` int64 group of each source cluster, numbered in order of each group's
        first cluster.
    """
    K_source = sums.shape[0]
    # These divisions guard zero mass.
    tiny = _tiny(sums.dtype)
    # Row k holds group k's statistics while the group is live; the arguments stay as given.
    group_sums, group_masses = sums.clone(), masses.clone()
    group_points, group_point_masses = points.clone(), point_masses.clone()
    labels = torch.arange(K_source, device=sums.device)

    def merge_costs(a: int) -> torch.Tensor:
        """Return the cost of merging group ``a`` with every group."""
        if classification:
            separate = _label_entropies(group_sums, group_masses)
            joined = _label_entropies(group_sums[a] + group_sums, group_masses[a] + group_masses)
            label_cost = joined - separate[a] - separate
        else:
            means = group_sums / group_masses.clamp_min(tiny)[:, None]
            shared = (
                group_masses[a] * group_masses / (group_masses[a] + group_masses).clamp_min(tiny)
            )
            label_cost = shared * (means[a] - means).square().sum(dim=1)
        shared = (
            group_point_masses[a]
            * group_point_masses
            / (group_point_masses[a] + group_point_masses).clamp_min(tiny)
        )
        geometry_cost = shared * (group_points[a] - group_points).square().sum(dim=1)
        # Round-off may leave a zero rise slightly negative.
        return label_cost.clamp_min(0.0) + geometry_weight * geometry_cost

    cost = torch.stack([merge_costs(a) for a in range(K_source)])
    cost.fill_diagonal_(torch.inf)
    for _ in range(K_source - K_target):
        a, b = divmod(int(cost.argmin()), K_source)
        a, b = min(a, b), max(a, b)
        total = group_point_masses[a] + group_point_masses[b]
        if total > 0:
            group_points[a] = (
                group_point_masses[a] * group_points[a] + group_point_masses[b] * group_points[b]
            ) / total
        group_point_masses[a] = total
        group_sums[a] = group_sums[a] + group_sums[b]
        group_masses[a] = group_masses[a] + group_masses[b]
        labels[labels == b] = a
        cost[b, :] = torch.inf
        cost[:, b] = torch.inf
        merged = merge_costs(a)
        merged[torch.isinf(cost[a])] = torch.inf  # group a itself and the groups merged away
        cost[a, :] = merged
        cost[:, a] = merged
    return torch.unique(labels, return_inverse=True)[1]


def _representative_points(geometry: _InitialisationGeometry) -> torch.Tensor:
    """Return source-cluster centroids of both modalities, scaled by root feature weights."""
    Wd_cont, Wd_cat = split_wd(geometry.feature_weights, geometry.continuous_centroids.shape[1])
    return torch.cat(
        [
            geometry.continuous_centroids * Wd_cont.sqrt(),
            *(
                rows * weight.sqrt()
                for weight, rows in zip(Wd_cat, geometry.categorical_centroids, strict=True)
            ),
        ],
        dim=1,
    )


def _seed_grouped_theta_(
    state: _ConnectionState,
    group_labels: torch.Tensor,
    source_masses: torch.Tensor,
    session: _FitSession,
) -> None:
    """Write the M transition that puts each source cluster's mass in its own group."""
    K_source = state.theta.shape[1]
    state.theta.zero_()
    state.theta[
        group_labels,
        torch.arange(K_source, device=state.theta.device),
    ] = source_masses
    if state.pseudocount != 0.0:
        state.theta.add_(state.pseudocount)
    normalise_(state.theta, 0, session.workspace.scratch_K[:K_source])
    floored_log_(state.log_theta, state.theta)


def _propagated_head_loss_(
    state: _ConnectionState,
    gamma_source: torch.Tensor,
    session: _FitSession,
    labels: tuple[torch.Tensor, torch.Tensor, bool],
) -> float:
    """Return the head's loss on the affiliations that the seeded transition propagates.

    The target block's affiliations are overwritten with those affiliations, using the
    same probability logs and propagation rule as prediction.
    """
    block = session.graph.blocks[state.description.target]
    assert isinstance(block, _HiddenBlock)
    state.propagate_(block.gamma, gamma_source, block.effective_epsilon, session.workspace)
    rows, row_masses, classification = labels
    return float(_head_loss(block.gamma.T @ rows, block.gamma.T @ row_masses, classification))


def _group_source_by_labels_(
    state: _ConnectionState,
    gamma_source: torch.Tensor,
    geometry: _InitialisationGeometry,
    session: _FitSession,
    generator: torch.Generator,
) -> torch.Tensor:
    """Assign each source cluster to the target group that serves the head best.

    The candidates are merges of the source clusters by the rise in the head's loss and
    ``_SEED_CANDIDATES`` geometric groupings. Each is propagated as prediction propagates it,
    and the grouping with the lowest head loss is kept, the merges winning a tie.
    """
    K_target = state.theta.shape[0]
    labels = _label_rows(session)
    rows, row_masses, classification = labels
    sums, masses = gamma_source.T @ rows, gamma_source.T @ row_masses
    points = _representative_points(geometry)
    last = state.description.target == session.graph.order[-2]
    candidates = [
        _group_by_head_loss(
            sums, masses, points, geometry.group_masses, K_target, classification, weight
        )
        for weight in ((_GEOMETRY_TIE_BREAK, 1.0) if last else (1.0,))
    ]
    candidates += list(
        _group_source(geometry, K_target, _spawn_generators(generator, _SEED_CANDIDATES))
    )
    best, best_loss = candidates[0], float("inf")
    for group_labels in candidates:
        _seed_grouped_theta_(state, group_labels, geometry.group_masses, session)
        loss = _propagated_head_loss_(state, gamma_source, session, labels)
        if loss < best_loss:
            best, best_loss = group_labels, loss
    return best


def _can_split_source_clusters(state: _ConnectionState, session: _FitSession) -> bool:
    """Return whether a widening connection can split its source clusters at prediction.

    It can when its ``delta`` is below the source block's affiliation temperature. At a
    larger ``delta`` the fit sharpens the source affiliations it trains on, and the extra
    target clusters receive no rows at prediction.
    """
    source = session.graph.blocks[state.description.source]
    assert isinstance(source, _ClusteringBlock)
    return state.delta < source.effective_epsilon


def _partition_rows_by_labels_(
    state: _ConnectionState,
    gamma_source: torch.Tensor,
    geometry: _InitialisationGeometry,
    categorical_rows: list[torch.Tensor],
    session: _FitSession,
    generator: torch.Generator,
) -> torch.Tensor:
    """Seed a widening S connection from the feature-space partition that serves the head best.

    ``_SEED_CANDIDATES`` partitions of the rows are drawn by k-means++ in feature space. Each
    gives the transition its exact update, is propagated as prediction propagates it, and the
    partition with the lowest head loss is kept, the first drawn winning a tie.

    Returns:
        ``(T,)`` int64 group of each row in the kept partition, whose transition is written.
    """
    data = session.data
    block = session.graph.blocks[state.description.target]
    assert isinstance(block, _HiddenBlock)
    rows = _InitialisationGeometry(
        data.X_cont, tuple(categorical_rows), data.sample_weights, geometry.feature_weights
    )
    labels = _label_rows(session)
    row_index = torch.arange(gamma_source.shape[0], device=gamma_source.device)
    kept: tuple[torch.Tensor, torch.Tensor] | None = None
    lowest = float("inf")
    generators = _spawn_generators(generator, _SEED_CANDIDATES)
    for group_labels in _group_source(rows, state.theta.shape[0], generators):
        block.gamma.zero_()
        block.gamma[row_index, group_labels] = 1.0
        state.update_parameters_(block.gamma, gamma_source, session)
        loss = _propagated_head_loss_(state, gamma_source, session, labels)
        if kept is None or loss < lowest:
            kept, lowest = (group_labels, state.theta.clone()), loss
    assert kept is not None
    state.theta.copy_(kept[1])
    floored_log_(state.log_theta, state.theta)
    return kept[0]


def _profile_partition(
    gamma_source: torch.Tensor,
    K_target: int,
    row_weights: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select source-affiliation profiles and assign rows to their nearest selection."""
    T = gamma_source.shape[0]
    empty = torch.empty(T, 0, dtype=gamma_source.dtype, device=gamma_source.device)
    cross_entropy = gamma_source.new_empty(1, T, K_target)
    seeds = select_kmeanspp_seeds_(
        cross_entropy,
        empty,
        [gamma_source],
        K_target,
        [generator],
        torch.empty(0, dtype=gamma_source.dtype, device=gamma_source.device),
        torch.ones(1, dtype=gamma_source.dtype, device=gamma_source.device),
        row_weights,
    )[0]
    return seeds, cross_entropy[0].argmin(dim=1)


def initialise_connection_(
    state: _ConnectionState,
    gamma_source: torch.Tensor,
    source_geometry: _InitialisationGeometry,
    session: _FitSession,
    *,
    generator: torch.Generator,
) -> _InitialisationGeometry:
    """Seed one M or S connection and pass its original-space geometry onwards.

    Narrowing M connections group source representatives, by the labels. A widening S
    connection whose ``delta`` is below its source block's affiliation temperature groups
    the original rows in feature space, by the labels. Every other connection selects
    source-affiliation profiles and groups the original rows. The prior is added to counts
    before normalisation, without changing the group masses used to seed the next connection.
    """
    K_target, K_source = state.theta.shape
    pseudocount = state.pseudocount
    if state.coupling is Coupling.M and K_target <= K_source:
        if K_target == K_source:
            group_labels = _group_source(source_geometry, K_target, [generator])[0]
        else:
            group_labels = _group_source_by_labels_(
                state, gamma_source, source_geometry, session, generator
            )
        _seed_grouped_theta_(state, group_labels, source_geometry.group_masses, session)
        continuous_centroids, categorical_centroids, group_masses = grouped_means(
            group_labels,
            source_geometry.group_masses,
            source_geometry.continuous_centroids,
            list(source_geometry.categorical_centroids),
            K_target,
        )
    else:
        row_weights = session.data.sample_weights
        categorical_rows = densify_categorical(
            list(session.data.X_cat), list(session.data.schema.M_cat), gamma_source.dtype
        )
        label_aware = (
            state.coupling is Coupling.S
            and K_target > K_source
            and _can_split_source_clusters(state, session)
        )
        if label_aware:
            group_labels = _partition_rows_by_labels_(
                state, gamma_source, source_geometry, categorical_rows, session, generator
            )
        else:
            seeds, group_labels = _profile_partition(gamma_source, K_target, row_weights, generator)
        continuous_centroids, categorical_centroids, group_masses = grouped_means(
            group_labels, row_weights, session.data.X_cont, categorical_rows, K_target
        )
        if state.coupling is Coupling.M:
            # A temporary hard partition supplies the M co-occurrence counts.
            gamma_target = torch.zeros(
                gamma_source.shape[0],
                K_target,
                dtype=gamma_source.dtype,
                device=gamma_source.device,
            )
            gamma_target[
                torch.arange(gamma_source.shape[0], device=gamma_source.device), group_labels
            ] = 1.0
            state.update_parameters_(gamma_target, gamma_source, session)
        elif not label_aware:
            state.theta.copy_(gamma_source[seeds])
            state.theta[group_masses == 0.0] = 1.0 / K_source
            if pseudocount != 0.0:
                # Assigned mass sets the prior's scale.
                state.theta.mul_(group_masses.unsqueeze(1)).add_(pseudocount)
                normalise_(state.theta, 1, session.workspace.scratch_K[:K_target])
            floored_log_(state.log_theta, state.theta)
    return _InitialisationGeometry(
        continuous_centroids=continuous_centroids,
        categorical_centroids=tuple(categorical_centroids),
        group_masses=group_masses,
        feature_weights=source_geometry.feature_weights,
    )


def initialise_hidden_chain_(
    session: _FitSession, parameters: tuple[_DownstreamParameters, ...] = ()
) -> None:
    """Initialise every hidden connection and affiliation in forward order.

    A seeded block loses the clusters whose selection repeats an earlier one, so the next
    connection is seeded on a source whose clusters can all be told apart.
    """
    graph = session.graph
    geometry = _input_geometry(graph.input, session)
    by_name = {group.description.name: group for group in parameters}
    hidden_names = graph.order[1:-1]
    # Transferred transitions are in place before any seeded block is capped, so that a cap
    # compacts their source columns as any later prune would.
    for name in hidden_names:
        supplied = by_name.get(name)
        if isinstance(supplied, _HiddenParameters):
            block = graph.blocks[name]
            assert isinstance(block, _HiddenBlock)
            (transition,) = block.incoming.values()
            transition.theta.copy_(supplied.theta)
            floored_log_(transition.log_theta, transition.theta)
    for position, name in enumerate(hidden_names):
        block = graph.blocks[name]
        assert isinstance(block, _HiddenBlock)
        (connection,) = graph.incoming[name]
        supplied = by_name.get(name)
        if isinstance(supplied, _HiddenParameters):
            block.initialise_(session)
        else:
            source_gamma = graph.affiliations(connection.source)
            generator = torch.Generator(device=source_gamma.device).manual_seed(
                dict(graph.connection_sub_seeds)[connection.name]
            )
            geometry = initialise_connection_(
                block.incoming[connection.name],
                source_gamma,
                geometry,
                session,
                generator=generator,
            )
            block.initialise_(session)
            geometry = _cap_seeded_width_(block, geometry, session)
        if (
            supplied is not None
            and position + 1 < len(hidden_names)
            and hidden_names[position + 1] not in by_name
        ):
            geometry = _geometry_from_affiliations(block.gamma, geometry.feature_weights, session)


def _cap_seeded_width_(
    block: _HiddenBlock, geometry: _InitialisationGeometry, session: _FitSession
) -> _InitialisationGeometry:
    """Prune the clusters whose seeded selection repeats an earlier one, and warn.

    Returns the geometry for the next connection, without the pruned clusters.
    """
    if not bool((geometry.group_masses == 0.0).any()):
        return geometry
    K = block.K
    block.prune_(session)
    if block.K == K:
        return geometry
    message = (
        f"hidden block {block.description.name!r} K={K} exceeds its {block.K} seeded "
        "selections with distinct source affiliations; reducing the initial hidden cluster "
        f"count to {block.K}"
    )
    session.warning_messages.append(message)
    _warn(message, UserWarning)
    keep = session.workspace.non_empty[:K]  # the prune's survivors
    return _InitialisationGeometry(
        continuous_centroids=geometry.continuous_centroids[keep],
        categorical_centroids=tuple(rows[keep] for rows in geometry.categorical_centroids),
        group_masses=geometry.group_masses[keep],
        feature_weights=geometry.feature_weights,
    )


def _geometry_from_affiliations(
    gamma: torch.Tensor, feature_weights: torch.Tensor, session: _FitSession
) -> _InitialisationGeometry:
    """Reconstruct temporary representatives for a new connection after transferred parameters."""
    data = session.data
    weighted = gamma * data.sample_weights[:, None]
    masses = weighted.sum(0)
    continuous = weighted.T @ data.X_cont
    categories = [
        weighted.T @ rows
        for rows in densify_categorical(list(data.X_cat), list(data.schema.M_cat), gamma.dtype)
    ]
    _finalise_group_means_(continuous, categories, masses, data.sample_weights, data.X_cont)
    return _InitialisationGeometry(continuous, tuple(categories), masses, feature_weights)
