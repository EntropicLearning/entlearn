"""Shared fixtures and helpers for Network tests.

Prefer adding a helper here over repeating a setup block. The private fit path takes four
steps to reach a session, and every test that reaches past ``Network.fit`` needs all four.
"""

import json
from dataclasses import fields, is_dataclass, replace

import torch
from conftest import DEVICE, DTYPE
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from entlearn import (
    ClassificationHead,
    Connection,
    Coupling,
    Hidden,
    InitialState,
    Input,
    ManifoldInput,
    Network,
    PredictConfig,
    Recipe,
    RegressionHead,
)
from entlearn.network.blocks import input as inp
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.blocks.input_common import recover_instance_weights_
from entlearn.network.build import _build_recipe, _BuiltRecipe
from entlearn.network.connections import _ConnectionState
from entlearn.network.data import _stage_data, _StagedData
from entlearn.network.fit import _loss, _materialise, loss_noise_threshold
from entlearn.network.predict import _forward
from entlearn.network.predict_init import seed_prediction_
from entlearn.network.refine import _Refinement
from entlearn.network.session import _FitSession

# Cluster-centre spread of each blob case, in the [0, 1] feature regime the fixtures assume.
_SEPARATED = 0.04
_OVERLAPPING = 0.28
BLOB_SPREADS = {"separated": _SEPARATED, "overlapping": _OVERLAPPING}


def read_saved_metadata(path):
    """Read the public file format without invoking its production decoder."""
    with safe_open(path, framework="pt") as bundle:
        return json.loads(bundle.metadata()["entlearn.network"])


def write_saved_metadata(path, metadata):
    """Replace embedded JSON while preserving tensors, including malformed JSON tests."""
    tensors = load_file(path)
    content = metadata if isinstance(metadata, str) else json.dumps(metadata)
    save_file(tensors, path, metadata={"entlearn.network": content})


def replace_saved_tensors(tensors, path):
    """Change tensors without accidentally testing missing metadata instead."""
    with safe_open(path, framework="pt") as bundle:
        metadata = bundle.metadata()
    save_file(tensors, path, metadata=metadata)


def round_trip_network(network, path, *, resumable):
    """Save and reload on the source device for same-environment comparisons."""
    network.save(path, resumable=resumable)
    return Network.load(path, device=network.device)


def change_metadata(path, change):
    """Apply a deliberate corruption to the portable metadata."""
    metadata = read_saved_metadata(path)
    change(metadata)
    write_saved_metadata(path, metadata)


def set_metadata(path, location, value):
    """Set the one metadata field that ``location``, a sequence of keys and indices, names."""

    def change(metadata):
        for key in location[:-1]:
            metadata = metadata[key]
        metadata[location[-1]] = value

    change_metadata(path, change)


def assert_same_checkpoint(actual, expected):
    """Compare the public parameters, row coordinates and continuation diagnostics."""
    assert actual.recipe == expected.recipe
    assert actual.schema == expected.schema
    assert actual.predict_config == expected.predict_config
    assert actual.diagnostics == expected.diagnostics
    names = [
        "continuous_centroids",
        "transition_matrices",
        "head_parameters",
        "training_affiliations",
        "training_instance_weights",
    ]
    names.append(
        "manifold_projectors"
        if isinstance(expected.recipe.blocks[0], ManifoldInput)
        else "feature_weights"
    )
    if expected.schema.M_cat:
        names.append("categorical_centroids")
    for name in names:
        torch.testing.assert_close(actual.inspect(name), expected.inspect(name), rtol=0, atol=0)


def assert_same_values(actual, expected, *, dtype=None):
    """Assert equal values held in separate storage, recursing through containers.

    ``dtype`` converts each expected tensor before the exact comparison. An empty tensor
    owns no storage, so only a non-empty pair must have distinct data pointers.
    """
    if isinstance(expected, torch.Tensor):
        wanted = expected if dtype is None else expected.to(dtype=dtype)
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert actual.data_ptr() != expected.data_ptr() or not actual.numel()
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            assert_same_values(
                getattr(actual, field.name), getattr(expected, field.name), dtype=dtype
            )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_same_values(actual[key], value, dtype=dtype)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same_values(left, right, dtype=dtype)
    else:
        assert actual == expected


def corrupt_saved_tensor(path, key, damage):
    """Replace or omit one persisted tensor without changing its metadata."""
    tensors = load_file(path)
    original = tensors.pop(key)
    if damage == "shape":
        tensors[key] = original.unsqueeze(0)
    elif damage == "dtype":
        dtype = torch.float32 if original.dtype == torch.float64 else torch.float64
        tensors[key] = original.to(dtype)
    elif damage == "nan":
        tensors[key] = torch.full_like(original, torch.nan)
    elif damage == "simplex":
        tensors[key] = torch.zeros_like(original)
    elif damage != "missing":
        raise ValueError(f"unknown tensor corruption {damage!r}")
    replace_saved_tensors(tensors, path)


def link(source: str, target: str, name: str | None = None, **fields) -> Connection:
    """Return a Connection named ``"{source}_to_{target}"``, as ``Recipe.chain`` names it.

    Every other field keeps the Connection default; the Recipe resolves a hidden-target
    link's missing ``coupling`` and ``theta_alpha``.
    """
    return Connection(name=name or f"{source}_to_{target}", source=source, target=target, **fields)


def active_widths(recipe: Recipe, network: Network) -> Recipe:
    """Return ``recipe`` with every clustering block set to its active width in ``network``.

    Pruning can narrow a block during the fit, and a target that reuses the captured
    parameters must declare the widths they carry.
    """
    widths = dict(network.schema.K_active)
    return replace(
        recipe,
        blocks=tuple(
            replace(block, K=widths[block.name]) if hasattr(block, "K") else block
            for block in recipe.blocks
        ),
    )


def classification_recipe(
    input_block: Input | None = None,
    *,
    K: int = 2,
    n_classes: int | None = None,
) -> Recipe:
    """Return a shallow classification Recipe."""
    return Recipe.chain(
        input_block or Input(K=K),
        ClassificationHead(coupling=Coupling.M, n_classes=n_classes),
    )


def regression_recipe(
    input_block: Input | None = None,
    *,
    K: int = 2,
    epsilon_M: float = float("inf"),
    W_M: tuple[float, ...] | None = None,
) -> Recipe:
    """Return a shallow regression Recipe."""
    return Recipe.chain(
        input_block or Input(K=K),
        RegressionHead(epsilon_M=epsilon_M, W_M=W_M),
    )


def validation_pairs() -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return unequal held-out partitions with a common two-row training core."""
    return tuple(
        (
            torch.tensor(train, dtype=torch.int64, device=DEVICE),
            torch.tensor(validation, dtype=torch.int64, device=DEVICE),
        )
        for train, validation in (
            ([0, 1, 2, 4, 5, 6], [3, 7]),
            ([0, 3, 4, 7], [1, 2, 5, 6]),
        )
    )


def squared_error(prediction: torch.Tensor, target: torch.Tensor, **context) -> torch.Tensor:
    """Return the unweighted mean squared error, a selection loss that ignores its context."""
    return (prediction - target).square().mean()


def deep_regression_recipe(
    widths: tuple[int, ...] = (4, 3),
    *,
    coupling: Coupling = Coupling.M,
    epsilon: float = 0.2,
    epsilon_M: float = float("inf"),
    W_M: tuple[float, ...] | None = None,
    theta_alpha: float = 1.2,
) -> Recipe:
    """Return a regression chain with the requested clustering widths."""
    return Recipe.chain(
        Input(K=widths[0], epsilon=epsilon),
        *(Hidden(K=K, epsilon=epsilon) for K in widths[1:]),
        RegressionHead(epsilon_M=epsilon_M, W_M=W_M),
        coupling=coupling,
        theta_alpha=theta_alpha,
    )


def deep_classification_recipe(
    widths: tuple[int, ...],
    *,
    coupling: Coupling | tuple[Coupling, ...] = Coupling.M,
    head_coupling: Coupling = Coupling.M,
    epsilon: float = 0.2,
    theta_alpha: float = 1.0,
) -> Recipe:
    """Return a classification chain with explicit hidden and head couplings."""
    return Recipe.chain(
        Input(K=widths[0], epsilon=epsilon),
        *(Hidden(K=K, epsilon=epsilon) for K in widths[1:]),
        ClassificationHead(coupling=head_coupling),
        coupling=coupling,
        theta_alpha=theta_alpha,
    )


def connection_initialisation_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return eight rows in two separated intervals for connection seeding checks."""
    return (
        torch.tensor(
            [[0.0], [0.1], [0.2], [0.3], [0.7], [0.8], [0.9], [1.0]],
            dtype=DTYPE,
            device=DEVICE,
        ),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64, device=DEVICE),
    )


def continuous_classification_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a square continuous input with two balanced classes."""
    return (
        torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=torch.float64,
        ),
        torch.tensor([0, 0, 1, 1]),
    )


def blobs(
    n_blobs: int,
    spread: float,
    *,
    rows_per_blob: int = 8,
    n_features: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian blobs in ``[0, 1]`` and their class codes.

    ``spread`` sets the noise around each centre, so a small value separates the blobs
    and a large one overlaps them. One class per blob, in blob order.
    """
    # Draw on the CPU stream so every device lane fits the same data.
    generator = torch.Generator().manual_seed(seed)
    centres = torch.linspace(0.15, 0.85, n_blobs, dtype=DTYPE, device=DEVICE)
    rows = []
    codes = []
    for index in range(n_blobs):
        noise = torch.randn(rows_per_blob, n_features, dtype=DTYPE, generator=generator).to(DEVICE)
        rows.append(centres[index] + spread * noise)
        codes.append(torch.full((rows_per_blob,), index, dtype=torch.int64, device=DEVICE))
    return torch.cat(rows).clamp_(0.0, 1.0), torch.cat(codes)


def resume_case(
    task: str = "classification", kind: str = "standard", coupling: Coupling = Coupling.M
) -> tuple[Recipe, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return a one-hidden-layer Recipe with its data for continuation and persistence tests.

    ``kind`` selects a standard, categorical or manifold input; only ``"categorical"``
    supplies a categorical feature.
    """
    X, codes = blobs(3, 0.2, rows_per_blob=10, n_features=3, seed=11)
    categories = (codes.remainder(2),) if kind == "categorical" else ()
    first = (
        ManifoldInput(K=4, subspace_dimension=1, epsilon=0.1, epsilon_T=0.3)
        if kind == "manifold"
        else Input(K=4, epsilon=0.1, epsilon_D=0.2, epsilon_T=0.3)
    )
    y = codes if task == "classification" else X[:, :2].square()
    head = (
        ClassificationHead(coupling=coupling)
        if task == "classification"
        else RegressionHead(epsilon_M=0.2)
    )
    recipe = Recipe.chain(
        first, Hidden(K=3, epsilon=0.15), head, coupling=coupling, theta_alpha=1.1
    )
    return recipe, X, y, categories


def prediction_model(
    task: str = "classification",
    *,
    input_kind: str = "continuous",
    depth: int = 1,
    coupling: Coupling = Coupling.M,
    head_coupling: Coupling = Coupling.M,
    epsilon: float = 0.1,
    epsilon_T: float = 0.5,
    epsilon_D: float = float("inf"),
    members: bool = False,
    predict_config: PredictConfig | None = None,
) -> tuple[Network, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Fit a small model for query, modality and retained-member comparisons."""
    X, codes = blobs(3, 0.28, n_features=4, seed=3)
    y = codes if task == "classification" else X[:, :2].square()
    categories = ()
    if input_kind in ("mixed", "categorical", "distribution"):
        categories = (codes.remainder(2), (X[:, 0] > 0.5).long())
        if input_kind == "distribution":
            categories = tuple(torch.nn.functional.one_hot(c, 2).to(DTYPE) for c in categories)
        if input_kind == "categorical":
            X = X[:, :0]
    first = (
        ManifoldInput(K=4, subspace_dimension=2, epsilon=epsilon, epsilon_T=epsilon_T)
        if input_kind == "manifold"
        else Input(K=4, epsilon=epsilon, epsilon_T=epsilon_T, epsilon_D=epsilon_D)
    )
    head = (
        ClassificationHead(coupling=head_coupling)
        if task == "classification"
        else RegressionHead(W_M=(0.2, 0.8))
    )
    recipe = Recipe.chain(
        first,
        *(Hidden(K=3, epsilon=epsilon) for _ in range(depth)),
        head,
        coupling=coupling if depth else None,
        theta_alpha=1.2,
    )
    network = Network.fit(
        recipe,
        X,
        y,
        X_cat=categories,
        max_iter=15,
        tol=0,
        seed=7,
        predict_config=predict_config
        or PredictConfig(epsilon_P=0.2 if task == "classification" else None),
        n_inits=2 if members else 1,
        retain="members" if members else "winner",
    )
    return network, X, categories


def two_row_input_model(
    epsilon: float, *, sample_weights: torch.Tensor | None = None
) -> tuple[Network, torch.Tensor]:
    """Fit a two-row, two-cluster input model and return it with the midpoint of its rows.

    The rows sit at 0 and 1, so every fitted quantity is exact: the centroids are the rows
    and the head is a 0/1 permutation. The midpoint's two costs therefore tie bitwise on any
    machine, and only the regime decides whether its affiliations split. Keep the tie exact:
    iterative refinement amplifies a one-ulp cost difference into a one-hot split.
    """
    X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
    target = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    network = Network.fit(
        Recipe.chain(Input(K=2, epsilon=epsilon), ClassificationHead(coupling=Coupling.M)),
        X,
        target,
        sample_weights=sample_weights,
        max_iter=1,
        tol=0,
        seed=2,
    )
    return network, X.mean(dim=0, keepdim=True)


def prediction_refinement(
    model: Network, X: torch.Tensor, categories: tuple[torch.Tensor, ...], config: PredictConfig
) -> _Refinement:
    """Create query coordinates for private per-step mathematical diagnostics."""
    graph = model._graph
    query, gammas = _forward(graph, X, X_cat_new=categories)
    prediction = model.predict(
        X, X_cat=categories, predict_config=replace(config, predict_mode="single")
    )
    query.refresh_instance_cost_()
    weights = None
    if bool(torch.isfinite(graph.input.log_partition)):
        weights = torch.empty_like(query.instance_cost)
        recover_instance_weights_(graph.input, weights, query.instance_cost)
    seed_prediction_(prediction, graph, gammas[graph.terminal.source], 7, query.workspace)
    state = _Refinement.allocate(graph, query, gammas, prediction, weights, config)
    state.refresh_target_()
    return state


def prediction_coordinate_loss(state: _Refinement) -> float:
    """Price live query coordinates after refreshing only derived input quantities."""
    state.query.refresh_instance_cost_()
    if state.weights is not None:
        torch.mul(
            state.weights,
            state.graph.training_rows / state.prediction.shape[0],
            out=state.input_scale,
        )
    return state.loss()


def manifold_line(*, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Return four unit-range collinear points with a known principal direction."""
    return torch.tensor(
        [[0.0, 0.0], [0.25, 0.0], [0.75, 0.0], [1.0, 0.0]], dtype=dtype, device=DEVICE
    )


def affine_subspace_blobs(
    *,
    n_subspaces: int = 2,
    rows_per_subspace: int = 12,
    n_features: int = 3,
    subspace_dimension: int = 1,
    orthogonal_noise: float = 0.02,
    dtype: torch.dtype = DTYPE,
    seed: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-range rotated/translated subspaces and cluster codes.

    Leading coordinates have distinct scales. Zero orthogonal noise gives exact
    alignment; positive noise populates the complementary directions. Each cluster
    has its own rotation, so multiple subspaces need not be parallel.
    """
    assert 1 <= subspace_dimension < n_features
    # Draw on the CPU stream so every device lane fits the same data.
    generator = torch.Generator().manual_seed(seed)
    rows, codes = [], []
    scales = torch.linspace(1.0, 0.4, subspace_dimension, dtype=dtype, device=DEVICE)
    for index in range(n_subspaces):
        rotation = torch.linalg.qr(
            torch.randn(n_features, n_features, dtype=dtype, generator=generator).to(DEVICE)
        ).Q
        coordinates = torch.randn(
            rows_per_subspace, n_features, dtype=dtype, generator=generator
        ).to(DEVICE)
        coordinates[:, :subspace_dimension] *= scales
        coordinates[:, subspace_dimension:] *= orthogonal_noise
        centre = 0.5 + (2 * index - n_subspaces + 1) * torch.linspace(
            1.0, 2.0, n_features, dtype=dtype, device=DEVICE
        )
        rows.append(coordinates @ rotation.T + centre)
        codes.append(torch.full((rows_per_subspace,), index, dtype=torch.int64, device=DEVICE))
    X = torch.cat(rows)
    # One scalar affine transform preserves angles, rank and singular-value ratios.
    X = (X - X.amin()) / (X.amax() - X.amin())
    return X, torch.cat(codes)


def regression_blobs(output_width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return two separated blobs and their piecewise-constant regression target."""
    X_cont, codes = blobs(
        2,
        BLOB_SPREADS["separated"],
        rows_per_blob=4,
        n_features=1,
    )
    values = torch.tensor(
        [[1.0, -1.0, 2.0], [-3.0, 4.0, 0.5]],
        dtype=DTYPE,
        device=DEVICE,
    )[:, :output_width]
    return X_cont, values[codes], values


def noisy_regression_blobs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return separated inputs with unequal residual scales across three outputs."""
    X_cont, target, _ = regression_blobs(3)
    target = target.clone()
    target[:, 1] += torch.tensor(
        [0.0, 0.2, -0.1, 0.1, 1.0, -1.0, 0.5, -0.5],
        dtype=DTYPE,
        device=DEVICE,
    )
    target[:, 2] += torch.tensor(
        [0.0, 4.0, -3.0, 2.0, -2.0, 3.0, -4.0, 1.0],
        dtype=DTYPE,
        device=DEVICE,
    )
    return X_cont, target


def diverging_regression() -> tuple[Recipe, torch.Tensor, torch.Tensor]:
    """Return a regression case whose loss overflows, so its fit warns of a loss increase."""
    X = torch.rand(30, 3, dtype=DTYPE, device=DEVICE)
    y = torch.rand(30, dtype=DTYPE, device=DEVICE) * (torch.finfo(DTYPE).max ** 0.5 * 1e6)
    recipe = Recipe.chain(Input(K=3, epsilon=0.1, epsilon_T=0.3), RegressionHead(epsilon_M=0.2))
    return recipe, X, y


def collapsed_reference_model() -> Network:
    """Fit one cluster to constant rows, so the scoring reference is a single value."""
    X = torch.zeros(4, 1, dtype=DTYPE, device=DEVICE)
    return Network.fit(
        Recipe.chain(Input(K=1, epsilon_T=0.5), RegressionHead()), X, X[:, 0], max_iter=1
    )


def stage(
    recipe: Recipe,
    X_cont: torch.Tensor,
    y: torch.Tensor,
    *,
    X_cat: tuple[torch.Tensor, ...] | None = None,
    sample_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    task_weights: torch.Tensor | None = None,
) -> tuple[_BuiltRecipe, _StagedData]:
    """Return the built Recipe and the staged data for one operation."""
    built = _build_recipe(recipe)
    data = _stage_data(
        built,
        X_cont,
        y,
        X_cat=X_cat,
        sample_weights=sample_weights,
        class_weights=class_weights,
        task_weights=task_weights,
        computation_dtype=None,
    )
    return built, data


def materialise(
    recipe: Recipe,
    X_cont: torch.Tensor,
    y: torch.Tensor,
    *,
    X_cat: tuple[torch.Tensor, ...] | None = None,
    sample_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    task_weights: torch.Tensor | None = None,
    initial_state: InitialState | None = None,
    seed: int = 0,
    warmup: int = 0,
) -> _FitSession:
    """Return a fit session for ``recipe``, advanced by ``warmup`` complete iterations.

    A warmed session starts near a fixed point, where the per-step loss changes are small
    enough for round-off to dominate.
    """
    from entlearn.network.fit import _fit_iteration_

    built, data = stage(
        recipe,
        X_cont,
        y,
        X_cat=X_cat,
        sample_weights=sample_weights,
        class_weights=class_weights,
        task_weights=task_weights,
    )
    state = initial_state or Network.initialise(
        recipe,
        X_cont,
        y,
        X_cat=X_cat,
        sample_weights=sample_weights,
        class_weights=class_weights,
        task_weights=task_weights,
        seed=seed,
    )
    session = _materialise(built, data, state)
    for _ in range(warmup):
        _fit_iteration_(session)
    return session


def assert_simplex_rows(values: torch.Tensor) -> None:
    """Assert that every row is a finite probability distribution."""
    assert values.ndim == 2
    assert bool(torch.isfinite(values).all())
    assert bool((values >= 0).all())
    torch.testing.assert_close(
        values.sum(dim=1),
        torch.ones(values.shape[0], dtype=values.dtype, device=values.device),
    )


def deep_regression_session(
    widths: tuple[int, ...] = (4, 3),
    *,
    coupling: Coupling = Coupling.M,
    epsilon: float = 0.2,
    epsilon_M: float = float("inf"),
    W_M: tuple[float, ...] | None = None,
    weighted: bool = False,
    warmup: int = 0,
) -> _FitSession:
    """Return a noisy regression chain, optionally with masked and weighted targets."""
    X_cont, target = noisy_regression_blobs()
    sample_weights = task_weights = None
    if weighted:
        target = target.clone()
        target[-1] = torch.nan
        sample_weights = torch.arange(1, X_cont.shape[0] + 1, dtype=DTYPE, device=DEVICE)
        task_weights = torch.linspace(0.5, 1.5, X_cont.shape[0], dtype=DTYPE, device=DEVICE)
    return materialise(
        deep_regression_recipe(
            widths,
            coupling=coupling,
            epsilon=epsilon,
            epsilon_M=epsilon_M,
            W_M=W_M,
        ),
        X_cont,
        target,
        sample_weights=sample_weights,
        task_weights=task_weights,
        seed=4,
        warmup=warmup,
    )


def deep_classification_session(
    widths: tuple[int, ...],
    *,
    coupling: Coupling = Coupling.M,
    weighted: bool = False,
) -> _FitSession:
    """Return a warmed classification chain for weighted or unweighted coordinate checks."""
    X_cont, target = blobs(3, BLOB_SPREADS["overlapping"], seed=3)
    sample_weights = (
        torch.linspace(1.0, 2.0, X_cont.shape[0], dtype=DTYPE, device=DEVICE) if weighted else None
    )
    return materialise(
        deep_classification_recipe(
            widths, coupling=coupling, head_coupling=coupling, theta_alpha=1.2
        ),
        X_cont,
        target,
        sample_weights=sample_weights,
        seed=3,
        warmup=2,
    )


def assert_non_increasing(before: float, after: float, label: str = "") -> None:
    """Assert one coordinate update did not raise the loss beyond round-off."""
    threshold = loss_noise_threshold(before, DTYPE)
    assert after <= before + threshold, f"{label}: {before} -> {after} (threshold {threshold})"


def hidden_connection(session: _FitSession) -> _ConnectionState:
    """Return the connection from the input to the first hidden block of a chain."""
    hidden = session.graph.blocks["hidden_1"]
    assert isinstance(hidden, _HiddenBlock)
    return hidden.incoming["input_to_hidden_1"]


def assert_deep_coordinates_non_increasing(session: _FitSession) -> None:
    """Price the head, each hidden affiliation and connection, and the complete input update."""
    before = _loss(session)
    session.graph.head.update_parameters_(session)
    after = _loss(session)
    assert_non_increasing(before, after, "head")

    for name in reversed(session.graph.order[1:-1]):
        hidden = session.graph.blocks[name]
        assert isinstance(hidden, _HiddenBlock)
        before = after
        hidden.update_affiliations_(session)
        hidden.prune_(session)
        after = _loss(session)
        assert_non_increasing(before, after, f"{name} affiliations")

        before = after
        hidden.update_incoming_connections_(session)
        after = _loss(session)
        assert_non_increasing(before, after, f"{name} incoming connection")

    before = after
    session.graph.input.update_parameters_(session)
    after = _loss(session)
    assert_non_increasing(before, after, "input")


def assert_input_coordinates_non_increasing(
    block: _StandardInputBlock,
    session: _FitSession,
    before: float,
) -> None:
    """Check every input-block coordinate from affiliations through centroids."""
    inp.update_affiliations_(block, session)
    inp.prune_(block, session)
    inp.refresh_cache_(block, session)
    after = _loss(session)
    assert_non_increasing(before, after, "affiliations")

    before = after
    inp.update_instance_weights_(block, session)
    inp.refresh_cache_(block, session)
    after = _loss(session)
    assert_non_increasing(before, after, "instance weights")

    inp.stage_statistics_(block, session)
    before = after
    inp.update_feature_weights_(block, session)
    inp.refresh_cache_(block, session)
    after = _loss(session)
    assert_non_increasing(before, after, "feature weights")

    inp.stage_statistics_(block, session)
    before = after
    inp.update_centroids_(block, session)
    inp.refresh_cache_(block, session)
    after = _loss(session)
    assert_non_increasing(before, after, "centroids")


def assert_head_then_input_non_increasing(session: _FitSession) -> None:
    """Price the head update, then every input-block coordinate, on a shallow chain."""
    block = session.graph.input
    head = session.graph.head
    before = _loss(session)
    head.update_parameters_(session)
    after = _loss(session)
    assert_non_increasing(before, after, "head")

    head.accumulate_into_source_cost_(session, cost=block.live_cache.disc_cost[:, : block.K])
    assert_input_coordinates_non_increasing(block, session, after)
