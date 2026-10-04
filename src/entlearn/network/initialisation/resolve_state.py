"""Validation and local resolution of optional named starting parameters."""

from collections.abc import Sequence

import torch

from entlearn._warnings import _warn
from entlearn.network.blocks.manifold import stage_projectors
from entlearn.network.build import _BuiltRecipe
from entlearn.network.data import _StagedData
from entlearn.network.initialisation.capacity import validate_connection_capacity
from entlearn.network.initialisation.geometry import _seed_initial_state, resolve_connection_seeds
from entlearn.network.parameter_validation import validate_simplex_tensor, validate_tensor
from entlearn.network.state import (
    InitialState,
    InputGeometry,
    _check_cluster_match,
    _ClassificationParameters,
    _copy_parameter_group,
    _DownstreamParameters,
    _HiddenParameters,
    _input_width,
    _RegressionParameters,
)
from entlearn.recipe import (
    ClassificationHead,
    Coupling,
    Head,
    Hidden,
    Input,
    ManifoldInput,
    RegressionHead,
)


def validate_initial_state(
    state: InitialState, computation_dtype: torch.dtype | None = None
) -> None:
    """Reject intrinsically malformed groups before matching them to a target.

    This is the trust boundary for a user-built ``InitialState``. Provided groups need
    not share a fitted model's dtype, device or feature schema. Fresh fitting may
    convert or discard otherwise valid groups, so values are checked as they will be
    after conversion to ``computation_dtype``. Retained groups use
    ``validate_original_state`` for those additional model-specific checks.
    """
    if type(state.parameters) is not tuple or type(state.connection_sub_seeds) is not tuple:
        raise ValueError("InitialState parameters and seeds must be tuples")
    if state.input_geometry is not None:
        _validate_input_geometry(state.input_geometry, computation_dtype)
    for group in state.parameters:
        _validate_parameter_group(group, computation_dtype)
    if any(type(name) is not str or not name for name in state.block_names) or len(
        state.block_names
    ) != len(set(state.block_names)):
        raise ValueError("starting parameter groups require unique non-empty block names")
    _validate_connection_seeds(state.connection_sub_seeds)


def _converted(value: object, name: str, computation_dtype: torch.dtype | None) -> torch.Tensor:
    """Validate a floating tensor, then its finiteness after conversion to the fit dtype."""
    tensor = validate_tensor(value, name, finite=False).to(dtype=computation_dtype)
    return validate_tensor(tensor, name)


def _simplex(
    value: torch.Tensor, name: str, computation_dtype: torch.dtype | None, dim: int = -1
) -> None:
    """Validate a stochastic axis after conversion to the fit dtype."""
    validate_simplex_tensor(value.to(dtype=computation_dtype), name, dim)


def _validate_input_geometry(
    geometry: InputGeometry, computation_dtype: torch.dtype | None
) -> None:
    """Validate one input geometry group, standard or manifold."""
    if type(geometry) is not InputGeometry or type(geometry.input) not in (Input, ManifoldInput):
        raise ValueError("invalid input geometry kind")
    if type(geometry.captured) is not bool:
        raise ValueError("captured must be a bool")
    C = _converted(geometry.continuous_centroids, "continuous_centroids", computation_dtype)
    requested, captured = geometry.input.K, geometry.captured
    # Capture records the active width as input.K; generation may fall short of it.
    if C.ndim != 2 or not (requested if captured else 1) <= C.shape[0] <= requested:
        count = f"exactly {requested}" if captured else f"between 1 and {requested}"
        raise ValueError(
            f"invalid centroid count: input geometry with input.K={requested} and "
            f"captured={captured} needs {count} centroids; got continuous_centroids of "
            f"shape {tuple(C.shape)}"
        )
    if type(geometry.categorical_centroids) is not tuple:
        raise ValueError("categorical centroids must be a tuple")
    K, D = C.shape
    if isinstance(geometry.input, ManifoldInput):
        if geometry.feature_weights is not None or geometry.categorical_centroids:
            raise ValueError("ManifoldInput has no feature_weights or categorical centroids")
        stage_projectors(
            geometry.manifold_projectors,
            C,
            geometry.input.subspace_dimension,
            allow_conversion=True,
        )
        return
    if geometry.manifold_projectors is not None:
        raise ValueError("standard input cannot have manifold_projectors")
    for c in geometry.categorical_centroids:
        c = validate_tensor(c, "categorical_centroids")
        if c.ndim != 2 or c.shape[0] != K or c.shape[1] < 2:
            raise ValueError("invalid categorical centroid shape")
        _simplex(c, "categorical_centroids", computation_dtype)
    weights = validate_tensor(
        geometry.feature_weights,
        "feature_weights",
        (D + len(geometry.categorical_centroids),),
    )
    _simplex(weights, "feature_weights", computation_dtype)


def _validate_parameter_group(
    group: _DownstreamParameters, computation_dtype: torch.dtype | None
) -> None:
    """Validate one hidden, classification or regression starting group."""
    match group:
        case _HiddenParameters():
            theta = validate_tensor(group.theta, "hidden transition")
            if (
                type(group.description) is not Hidden
                or theta.ndim != 2
                or theta.shape[0] != group.description.K
                or theta.shape[1] < 1
            ):
                raise ValueError("invalid hidden transition shape")
            if type(group.coupling) is not Coupling:
                raise ValueError("invalid hidden coupling")
            dim = 0 if group.coupling is Coupling.M else 1
            _simplex(theta, "hidden transition", computation_dtype, dim)
        case _ClassificationParameters():
            theta = validate_tensor(group.theta, "classification transition")
            if (
                type(group.description) is not ClassificationHead
                or theta.ndim != 2
                or theta.shape[0] != group.description.n_classes
                or theta.shape[1] < 1
            ):
                raise ValueError("invalid classification transition shape")
            dim = 0 if group.description.coupling is Coupling.M else 1
            _simplex(theta, "classification transition", computation_dtype, dim)
        case _RegressionParameters():
            C_y = _converted(group.C_y, "regression centroids", computation_dtype)
            if type(group.description) is not RegressionHead or C_y.ndim != 2 or min(C_y.shape) < 1:
                raise ValueError("invalid regression centroid shape")
            if group.W_M is not None:
                W_M = validate_tensor(group.W_M, "W_M", (C_y.shape[0],))
                _simplex(W_M, "W_M", computation_dtype)
        case _:
            raise ValueError("unsupported starting parameter group")


def _validate_connection_seeds(seeds: tuple[tuple[str, int], ...]) -> None:
    """Require unique named connection seeds in ``[0, 2**63)``."""
    if any(type(entry) is not tuple or len(entry) != 2 for entry in seeds):
        raise ValueError("named connection seeds require name and seed pairs")
    if any(
        type(name) is not str or not name or type(seed) is not int or not 0 <= seed < 2**63
        for name, seed in seeds
    ):
        raise ValueError("invalid named connection seed")
    if len(seeds) != len(dict(seeds)):
        raise ValueError("duplicate connection seed names")


def staging_hints(
    built: _BuiltRecipe,
    state: InitialState | None,
    X: object,
    cats: object,
    y: object,
    computation_dtype: torch.dtype | None = None,
) -> tuple[tuple[int, ...] | None, int | None]:
    """Validate a provided state once and preserve its category and class counts.

    This is the one operation-level validation of a provided state, including groups
    the target will not use. Counts are preserved only where provided groups describe
    the new data.
    """
    if state is None:
        return None, None
    dtype = computation_dtype or (X.dtype if isinstance(X, torch.Tensor) else None)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("computation dtype must be float32 or float64")
    validate_initial_state(state, dtype)
    geometry = state.input_geometry
    cards = None
    if (
        geometry is not None
        and geometry.input.name == built.input.name
        and type(geometry.input) is type(built.input)
    ):
        columns = geometry.categorical_centroids
        if (
            isinstance(X, torch.Tensor)
            and X.ndim == 2
            and X.shape[1] == geometry.continuous_centroids.shape[1]
            and isinstance(cats, Sequence)
            and not isinstance(cats, (str, bytes))
            and len(cats) == len(columns)
        ):
            compatible = all(
                isinstance(c, torch.Tensor)
                and (
                    (
                        c.ndim == 1
                        and c.dtype == torch.int64
                        and (c.numel() == 0 or int(c.max()) < stored.shape[1])
                    )
                    or (c.ndim == 2 and c.shape[1] == stored.shape[1])
                )
                for c, stored in zip(cats, columns, strict=True)
            )
            if compatible:
                cards = tuple(c.shape[1] for c in columns)
    output = next((p for p in state.parameters if p.description.name == built.head.name), None)
    known_n_classes = None
    if (
        isinstance(output, _ClassificationParameters)
        and isinstance(built.head, ClassificationHead)
        and built.head.n_classes is None
        and isinstance(y, torch.Tensor)
    ):
        M = output.theta.shape[0]
        if (y.ndim == 1 and y.dtype == torch.int64 and (y.numel() == 0 or int(y.max()) < M)) or (
            y.ndim == 2 and y.shape[1] == M
        ):
            known_n_classes = M
    return cards, known_n_classes


def _reject(messages: list[str], name: str, reason: str) -> None:
    """Warn that a provided group is incompatible and will be initialised normally."""
    _note(
        messages,
        f"InitialState block {name!r} is incompatible ({reason}); initialising normally",
    )


def _select_geometry(
    built: _BuiltRecipe,
    data: _StagedData,
    supplied: InitialState,
    messages: list[str],
    *,
    seed: int,
    allow_conversion: bool,
) -> InputGeometry:
    """Return the supplied input geometry when it fits the Recipe and data, else a generated one."""
    geometry = supplied.input_geometry
    if geometry is not None and geometry.input.name != built.input.name:
        geometry = None
    if geometry is not None:
        # A geometry was supplied and matches the input block in the recipe
        try:
            _check_cluster_match(geometry, built.input)
        except ValueError as error:
            _reject(messages, built.input.name, str(error))
            geometry = None
    if geometry is not None and (
        geometry.continuous_centroids.shape[1] != data.schema.D_cont
        or tuple(c.shape[1] for c in geometry.categorical_centroids) != data.schema.M_cat
    ):
        # As before, but shape mismatch
        _reject(messages, built.input.name, "feature layout")
        geometry = None
    if geometry is None:
        # A geometry was not provided, rejected, or does not match
        generated = _seed_initial_state(built, data, allow_conversion=allow_conversion, seed=seed)
        geometry = generated.input_geometry
    assert geometry is not None
    return geometry


def _check_kept_dtypes(
    groups: Sequence[InputGeometry | _DownstreamParameters], dtype: torch.dtype
) -> None:
    """Require every tensor of the kept groups to use the computation dtype."""
    for group in groups:
        tensors = (
            (
                group.continuous_centroids,
                *group.categorical_centroids,
                group.feature_weights,
                group.manifold_projectors,
            )
            if isinstance(group, InputGeometry)
            else (
                (group.C_y, group.W_M)
                if isinstance(group, _RegressionParameters)
                else (group.theta,)
            )
        )
        if any(t is not None and t.dtype != dtype for t in tensors):
            raise ValueError("InitialState tensors must use the computation dtype")


def resolve_state(
    built: _BuiltRecipe,
    data: _StagedData,
    state: InitialState | None,
    *,
    seed: int,
    allow_conversion: bool,
    provided: bool = False,
) -> tuple[InitialState, tuple[str, ...]]:
    """Select compatible groups and generate missing input geometry without reseeding hidden blocks.

    A ``provided`` state passed ``staging_hints`` and warns of unused parts; others are internal.
    """
    messages: list[str] = []
    supplied = InitialState() if state is None else state
    geometry = _select_geometry(
        built, data, supplied, messages, seed=seed, allow_conversion=allow_conversion
    )
    K_source = _input_width(geometry, built.input)
    retained: list[_DownstreamParameters] = []
    by_name = {group.description.name: group for group in supplied.parameters}
    for description in built.topological_blocks[1:]:
        assert isinstance(description, Hidden | Head)
        group = by_name.get(description.name)
        (connection,) = built.incoming[description.name]
        compatible = parameters_match(
            group, description, K_source, data.schema.M, coupling=connection.coupling
        )
        if isinstance(description, Hidden):
            if not compatible:
                validate_connection_capacity(
                    connection, K_source, description.K, data.X_cont.shape[0]
                )
            K_source = description.K
        if group is not None:
            if compatible:
                retained.append(group)
            else:
                _reject(messages, description.name, "block kind, coupling or active dimensions")
    _note_unused(
        messages,
        built,
        supplied if provided else None,
        geometry_used=geometry is supplied.input_geometry,
        retained=bool(retained),
        width=_input_width(geometry, built.input),
    )

    # Placement conversions are authorised explicitly for dtype, and by the new input device.
    if not allow_conversion:
        _check_kept_dtypes((geometry, *retained), data.X_cont.dtype)
    resolved = InitialState(
        _copy_parameter_group(geometry, data.X_cont.dtype, data.X_cont.device),
        tuple(
            _copy_parameter_group(group, data.X_cont.dtype, data.X_cont.device)
            for group in retained
        ),
        supplied.connection_sub_seeds,
    )
    return resolve_connection_seeds(built, resolved, seed), tuple(messages)


def _note(messages: list[str], message: str) -> None:
    """Retain a fit warning and emit it on the caller's line."""
    messages.append(message)
    _warn(message, UserWarning)


def _note_unused(
    messages: list[str],
    built: _BuiltRecipe,
    provided: InitialState | None,
    *,
    geometry_used: bool,
    retained: bool,
    width: int,
) -> None:
    """Warn for each provided group or connection seed the Recipe lacks, and for a capped input.

    A state the fit uses nothing from warns once, unless an earlier warning gives the cause.
    """
    if provided is None:
        return
    geometry = provided.input_geometry
    downstream = {description.name for description in built.topological_blocks[1:]}
    unmatched = [
        group.description.name
        for group in provided.parameters
        if group.description.name not in downstream
    ]
    if geometry is not None and geometry.input.name != built.input.name:
        unmatched.insert(0, geometry.input.name)
    for name in unmatched:
        _note(
            messages,
            f"InitialState block {name!r} matches no Recipe block and is ignored; "
            "initialising normally",
        )
    connections = {connection.name for connection in built.connections}
    for name in (name for name, _ in provided.connection_sub_seeds if name not in connections):
        _note(
            messages,
            f"InitialState connection seed {name!r} matches no Recipe connection and is "
            "ignored; initialising normally",
        )
    K = built.input.K
    if geometry_used and width < K:
        _note(
            messages,
            f"InitialState block {built.input.name!r} holds {width} centroids for the "
            f"requested K={K}; fitting {width} input clusters",
        )
    seeded = any(name in connections for name, _ in provided.connection_sub_seeds)
    if not (geometry_used or retained or seeded or messages):
        _note(
            messages,
            "InitialState holds no group or connection seed of this Recipe; initialising normally",
        )


def parameters_match(
    group: _DownstreamParameters | None,
    description: Hidden | Head,
    source_width: int,
    output_width: int,
    *,
    coupling: Coupling | None,
) -> bool:
    """Check local starting-parameter compatibility without seeding or modifying state."""
    if isinstance(description, Hidden):
        return (
            isinstance(group, _HiddenParameters)
            and group.coupling is coupling
            and group.theta.shape == (description.K, source_width)
        )
    if isinstance(description, ClassificationHead):
        return (
            isinstance(group, _ClassificationParameters)
            and group.description.coupling is description.coupling
            and group.theta.shape == (output_width, source_width)
        )
    return isinstance(group, _RegressionParameters) and group.C_y.shape == (
        output_width,
        source_width,
    )
