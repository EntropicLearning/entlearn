"""Validated input-geometry generation shared by initialisation and fitting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import torch

from entlearn._seeds import _connection_sub_seed
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.blocks.manifold_geometry import seed_projectors
from entlearn.network.build import _build_recipe, _BuiltRecipe
from entlearn.network.data import _stage_data, _StagedData
from entlearn.network.initialisation.capacity import validate_profile_capacity
from entlearn.network.initialisation.centroids import (
    cap_input_clusters_to_eligible_rows,
    select_centroid_seeds,
    stage_supplied_centroids,
)
from entlearn.network.initialisation.features import initialise_feature_weights
from entlearn.network.session import _CompiledGraph
from entlearn.network.state import InitialState, InputGeometry
from entlearn.network.validation import _normalise_weights
from entlearn.primitives.encoding import densify_categorical, split_wd
from entlearn.recipe import Input, ManifoldInput, Recipe


@torch.inference_mode()
def capture_input_geometry(graph: _CompiledGraph) -> InputGeometry:
    """Copy active input geometry without row-dependent values."""
    block = graph.input
    standard = isinstance(block, _StandardInputBlock)
    return InputGeometry(
        input=replace(block.description, K=block.K),
        continuous_centroids=block.continuous_centroids.detach().clone(),
        categorical_centroids=(
            tuple(value.detach().clone() for value in block.categorical_centroids)
            if standard
            else ()
        ),
        feature_weights=block.feature_weights.detach().clone() if standard else None,
        manifold_projectors=None if standard else block.manifold_projectors.detach().clone(),
        captured=True,
    )


def resolve_connection_seeds(built: _BuiltRecipe, state: InitialState, seed: int) -> InitialState:
    """Resolve named seeds for a fresh graph without modifying supplied geometry."""
    recorded = dict(state.connection_sub_seeds)
    if len(recorded) != len(state.connection_sub_seeds):
        raise ValueError("InitialState connection names must be unique")
    seeds = tuple(
        (
            connection.name,
            recorded.get(connection.name, _connection_sub_seed(seed, connection.name)),
        )
        for connection in built.connections
    )
    if seeds == state.connection_sub_seeds:
        return state
    return replace(state, connection_sub_seeds=seeds)


def _prepare_initialisation(
    recipe: Recipe,
    X_cont: torch.Tensor,
    y: torch.Tensor,
    *,
    X_cat: Sequence[torch.Tensor] | None,
    sample_weights: torch.Tensor | None,
    class_weights: torch.Tensor | None,
    task_weights: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None,
    feature_weights: torch.Tensor | None,
    continuous_centroids: torch.Tensor | None,
    categorical_centroids: Sequence[torch.Tensor] | None,
    computation_dtype: torch.dtype | None,
) -> tuple[_BuiltRecipe, _StagedData]:
    """Validate one initialisation operation and stage its common data once."""
    if type(recipe) is not Recipe:
        raise ValueError("Recipe must be an exact Recipe")
    built = _build_recipe(recipe)
    input_block = built.input
    if feature_weights is not None:
        if isinstance(input_block, ManifoldInput):
            raise ValueError("ManifoldInput has no feature_weights")
        if input_block.W_std > 0.0:
            raise ValueError("feature_weights and a positive W_std are mutually exclusive")
    centroids_supplied = continuous_centroids is not None or categorical_centroids is not None
    if centroids_supplied and (input_block.centroid_strategy != "kmeans++" or input_block.balanced):
        raise ValueError(
            "supplied centroids and non-default selection controls are mutually exclusive"
        )
    data = _stage_data(
        built,
        X_cont,
        y,
        X_cat=X_cat,
        sample_weights=sample_weights,
        class_weights=class_weights,
        task_weights=task_weights,
        computation_dtype=computation_dtype,
    )
    validate_profile_capacity(built, data.X_cont.shape[0])
    return built, data


# Seed initialisations in fixed-size batches so changing n_inits does not change
# tensor shapes, floating-point rounding, or a candidate's selected centroid rows.
_SEED_BATCH_SIZE = 8


def _select_seeds_in_fixed_batches(
    select: Callable[[list[torch.Generator]], torch.Tensor], generators: list[torch.Generator]
) -> torch.Tensor:
    """Select centroid rows in batches of ``_SEED_BATCH_SIZE`` generators.

    Pad an incomplete batch with copies of its first generator and discard their results.
    This keeps the calculations the same shape, avoiding batch-size-dependent rounding
    that could change the selected rows. Pad even a single generator so a one-start
    fit can reproduce a candidate drawn during a multi-start fit.
    """
    chosen = []
    for start in range(0, len(generators), _SEED_BATCH_SIZE):
        chunk = generators[start : start + _SEED_BATCH_SIZE]
        lead, width = chunk[0], len(chunk)
        copies = [
            torch.Generator(device=lead.device).set_state(lead.get_state())
            for _ in range(_SEED_BATCH_SIZE - width)
        ]
        chosen.append(select(chunk + copies)[:width])
    return torch.cat(chosen)


@torch.inference_mode()
def _seed_initial_states(
    built: _BuiltRecipe,
    data: _StagedData,
    *,
    allow_conversion: bool,
    seeds: Sequence[int],
    feature_weights: torch.Tensor | None = None,
    continuous_centroids: torch.Tensor | None = None,
    categorical_centroids: Sequence[torch.Tensor] | None = None,
    rows: torch.Tensor | None = None,
) -> list[InitialState]:
    """Build one initial state per random seed from already-staged data.

    Draw centroid rows in fixed-size batches when feature weights agree across
    initialisations. Otherwise, draw each initialisation separately using its own
    feature weights. Each state holds input geometry and named connection seeds.
    """
    input_block = built.input
    X_cont, X_cat = data.X_cont, data.X_cat
    sample_weights, target, labelled = data.sample_weights, data.target, data.labelled
    if rows is not None:
        X_cont = X_cont[rows]
        X_cat = tuple(feature[rows] for feature in X_cat)
        raw = data.raw_sample_weights
        sample_weights = (
            torch.full((len(rows),), 1 / len(rows), dtype=X_cont.dtype, device=X_cont.device)
            if raw is None
            else _normalise_weights(raw[rows], name="init_rows sample_weights")
        )
        target, labelled = target[rows], labelled[rows]
    _, D_cont = X_cont.shape
    D_total = D_cont + len(X_cat)
    generators = [torch.Generator(device=X_cont.device).manual_seed(seed) for seed in seeds]
    resolved_feature_weights = [
        initialise_feature_weights(
            D_total,
            input_block.W_std if isinstance(input_block, Input) else 0.0,
            feature_weights,
            generator=generator,
            dtype=data.X_cont.dtype,
            device=data.X_cont.device,
        )
        for generator in generators
    ]
    supplied_centroids = stage_supplied_centroids(
        continuous_centroids,
        categorical_centroids,
        K=input_block.K,
        D_cont=D_cont,
        M_cat=data.schema.M_cat,
        dtype=data.X_cont.dtype,
        device=data.X_cont.device,
        allow_conversion=allow_conversion,
    )
    if supplied_centroids is None:
        n_candidates = (
            1 if input_block.centroid_strategy == "kmeans++" else input_block.greedy_candidates
        )
        K = cap_input_clusters_to_eligible_rows(
            input_block.K, labelled, balanced=input_block.balanced
        )

        def select(seeded: list[torch.Generator], weights: torch.Tensor) -> torch.Tensor:
            return select_centroid_seeds(
                X_cont,
                list(X_cat),
                K,
                seeded,
                *split_wd(weights, D_cont),
                sample_weights,
                target,
                labelled,
                balanced=input_block.balanced,
                n_candidates=n_candidates,
            )

        first = resolved_feature_weights[0]
        if all(torch.equal(weights, first) for weights in resolved_feature_weights):
            chosen = _select_seeds_in_fixed_batches(lambda chunk: select(chunk, first), generators)
        else:  # Each initialisation uses its own feature weights to compute dissimilarities.
            chosen = torch.cat(
                [select([g], w) for g, w in zip(generators, resolved_feature_weights, strict=True)]
            )
        centroids = [
            (
                X_cont[selected],
                tuple(
                    densify_categorical(
                        [feature[selected] for feature in X_cat],
                        list(data.schema.M_cat),
                        X_cont.dtype,
                    )
                ),
            )
            for selected in chosen
        ]
    else:
        centroids = [supplied_centroids] * len(seeds)
    return [
        InitialState(
            input_geometry=InputGeometry(
                input=input_block,
                continuous_centroids=continuous,
                categorical_centroids=categorical,
                feature_weights=weights if isinstance(input_block, Input) else None,
                manifold_projectors=(
                    seed_projectors(X_cont, continuous, input_block.subspace_dimension)
                    if isinstance(input_block, ManifoldInput)
                    else None
                ),
                captured=False,
            ),
            connection_sub_seeds=tuple(
                (connection.name, _connection_sub_seed(seed, connection.name))
                for connection in built.connections
            ),
        )
        for seed, weights, (continuous, categorical) in zip(
            seeds, resolved_feature_weights, centroids, strict=True
        )
    ]
