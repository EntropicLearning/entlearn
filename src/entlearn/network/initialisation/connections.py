"""Private automatic initialisation of clustering connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.manifold import _ManifoldInputBlock
from entlearn.network.initialisation.centroids import select_kmeanspp_seeds
from entlearn.network.state import _DownstreamParameters, _HiddenParameters
from entlearn.primitives.dissimilarity import seed_dissimilarity
from entlearn.primitives.encoding import densify_categorical, split_wd
from entlearn.primitives.geometry import _finalise_group_means_, grouped_means
from entlearn.primitives.normalise import _eps, floored_log_, normalise_
from entlearn.recipe import Coupling

if TYPE_CHECKING:
    from entlearn.network.blocks.types import _InputBlock
    from entlearn.network.connections import _ConnectionState
    from entlearn.network.session import _FitSession


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


def _group_source(
    geometry: _InitialisationGeometry,
    K_target: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Assign each source cluster to one geometrically selected target group."""
    K_source, D_cont = geometry.continuous_centroids.shape
    if K_target == K_source:
        return torch.arange(
            K_source, dtype=torch.int64, device=geometry.continuous_centroids.device
        )
    Wd_cont, Wd_cat = split_wd(geometry.feature_weights, D_cont)
    seeds = select_kmeanspp_seeds(
        geometry.continuous_centroids,
        list(geometry.categorical_centroids),
        K_target,
        generator,
        Wd_cont,
        Wd_cat,
        geometry.group_masses.clamp_min(_eps(geometry.group_masses.dtype)),
    )
    categorical_logs = [
        rows.clamp_min(_eps(rows.dtype)).log() for rows in geometry.categorical_centroids
    ]
    dissimilarities = torch.stack(
        [
            seed_dissimilarity(
                geometry.continuous_centroids,
                list(geometry.categorical_centroids),
                categorical_logs,
                Wd_cont,
                Wd_cat,
                int(seed),
            )
            for seed in seeds
        ],
        dim=1,
    )
    labels = dissimilarities.argmin(dim=1)
    labels[seeds] = torch.arange(K_target, device=labels.device)
    return labels


def _profile_partition(
    gamma_source: torch.Tensor,
    K_target: int,
    row_weights: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select source-affiliation profiles and assign rows to their nearest selection."""
    T = gamma_source.shape[0]
    if K_target > T:
        raise ValueError(
            f"hidden initialisation needs {K_target} profiles but only {T} rows are available"
        )
    empty = torch.empty(T, 0, dtype=gamma_source.dtype, device=gamma_source.device)
    seeds = select_kmeanspp_seeds(
        empty,
        [gamma_source],
        K_target,
        generator,
        torch.empty(0, dtype=gamma_source.dtype, device=gamma_source.device),
        torch.ones(1, dtype=gamma_source.dtype, device=gamma_source.device),
        row_weights.clamp_min(_eps(row_weights.dtype)),
    )
    logs = gamma_source.clamp_min(_eps(gamma_source.dtype)).log()
    cross_entropy = torch.stack(
        [-(gamma_source * logs[int(seed)]).sum(dim=1) for seed in seeds],
        dim=1,
    )
    return seeds, cross_entropy.argmin(dim=1)


def initialise_connection_(
    state: _ConnectionState,
    gamma_source: torch.Tensor,
    source_geometry: _InitialisationGeometry,
    session: _FitSession,
    *,
    generator: torch.Generator,
) -> _InitialisationGeometry:
    """Seed one M or S connection and pass its original-space geometry onwards.

    Narrowing M connections group source representatives. Widening M connections and
    every S connection select source-affiliation profiles and group the original rows.
    The prior is added to counts before normalisation, without changing the group masses
    used to seed the next connection.
    """
    K_target, K_source = state.theta.shape
    pseudocount = state.pseudocount
    if state.coupling is Coupling.M and K_target <= K_source:
        group_labels = _group_source(source_geometry, K_target, generator)
        state.theta.zero_()
        state.theta[
            group_labels,
            torch.arange(K_source, device=state.theta.device),
        ] = source_geometry.group_masses
        if pseudocount != 0.0:
            state.theta.add_(pseudocount)
        normalise_(state.theta, 0, session.workspace.scratch_K[:K_source])
        floored_log_(state.log_theta, state.theta)
        continuous_centroids, categorical_centroids, group_masses = grouped_means(
            group_labels,
            source_geometry.group_masses,
            source_geometry.continuous_centroids,
            list(source_geometry.categorical_centroids),
            K_target,
        )
    else:
        row_weights = session.data.sample_weights
        seeds, group_labels = _profile_partition(gamma_source, K_target, row_weights, generator)
        categorical_rows = densify_categorical(
            list(session.data.X_cat), list(session.data.schema.M_cat), gamma_source.dtype
        )
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
        else:
            state.theta.copy_(gamma_source[seeds])
            if pseudocount != 0.0:
                # Assigned mass sets the prior's scale. With no prior, duplicate selected
                # profiles remain intact even when their tie-broken group has no rows.
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
    """Initialise every hidden connection and affiliation in forward order."""
    graph = session.graph
    geometry = _input_geometry(graph.input, session)
    by_name = {group.description.name: group for group in parameters}
    hidden_names = graph.order[1:-1]
    for position, name in enumerate(hidden_names):
        block = graph.blocks[name]
        assert isinstance(block, _HiddenBlock)
        (connection,) = graph.incoming[name]
        source_gamma = graph.affiliations(connection.source)
        generator = torch.Generator(device=source_gamma.device).manual_seed(
            dict(graph.connection_sub_seeds)[connection.name]
        )
        supplied = by_name.get(name)
        if isinstance(supplied, _HiddenParameters):
            transition = block.incoming[connection.name]
            transition.theta.copy_(supplied.theta)
            floored_log_(transition.log_theta, transition.theta)
        else:
            geometry = initialise_connection_(
                block.incoming[connection.name],
                source_gamma,
                geometry,
                session,
                generator=generator,
            )
        block.initialise_(session)
        if (
            supplied is not None
            and position + 1 < len(hidden_names)
            and hidden_names[position + 1] not in by_name
        ):
            geometry = _geometry_from_affiliations(block.gamma, geometry.feature_weights, session)


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
