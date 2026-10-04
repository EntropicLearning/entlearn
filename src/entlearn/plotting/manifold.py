"""The fitted manifolds of a manifold input in three dimensions, with the data and probes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.colors import qualitative

from entlearn.network.queries import _scores_available
from entlearn.plotting.basic import CLASS_COLOURS, _blend, _numpy, _target

if TYPE_CHECKING:
    from entlearn import Network

_DIRECTIONS, _STEPS, _RINGS = 72, 60, 15


def _grid_faces(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two triangles of every cell of a ``rows`` by ``cols`` grid of vertices."""
    corner = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)).reshape(-1)
    return (
        np.concatenate([corner, corner + 1]),
        np.concatenate([corner + 1, corner + cols + 1]),
        np.concatenate([corner + cols, corner + cols]),
    )


def _on_grid(
    X: torch.Tensor, box: torch.Tensor, resolution: int, features: list[int]
) -> torch.Tensor:
    """Points on a regular grid over ``box`` in ``features``, the other features at the median."""
    axes = [
        torch.linspace(float(box[0, i]), float(box[1, i]), resolution, dtype=X.dtype)
        for i in range(3)
    ]
    grid = torch.stack([g.flatten() for g in torch.meshgrid(*axes, indexing="ij")], dim=1)
    points = X.median(dim=0).values.repeat(len(grid), 1)
    points[:, features] = grid.to(X.device)
    return points


def plot_manifold(
    network: Network,
    X: torch.Tensor,
    y: Any = None,
    *,
    probes: torch.Tensor | None = None,
    features: tuple[int, int, int] = (0, 1, 2),
    output: int = 0,
    block: str = "input",
    volume_resolution: int = 30,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot a manifold input's local manifolds, what they carry, and where new points land.

    The figure is a 3-D scene over three features, with layers you switch on and off
    from the legend:

    - **Manifolds**: each cluster's local plane, or line when ``subspace_dimension`` is 1.
      With instance weights, a manifold also stops where ``network.score_samples`` falls
      below the lowest score among ``X``, and it fades as the score drops.
    - **Manifolds (prediction)** (hidden at first): the same manifolds, but coloured by
      what the network predicts at each point on them. A regression network uses its
      output ``output`` on the colour scale of ``y``, a classifier mixes the class colours of
      ``plot_decision`` in proportion to the predicted probabilities.
    - **Observations**: the observations ``X``, coloured by ``y``. When ``X`` is the
      training data, they are drawn larger the larger their fitted instance weight.
    - **Probes**: with ``probes``, each probe's end, where the network reconstructs it,
      sized and coloured by its score; hidden layers show each probe's origin and its
      movement from origin to end.
    - **Inlier fade** (hidden by default, with instance weights): the score over the
      whole box, drawn as a cloud.

    Args:
        network: A fitted network whose input block is a ``ManifoldInput``.
        X: The observations to draw, usually the training data, on the network's device
            and dtype.
        y: The values that colour them, such as their targets or predictions.
        probes: Further points to send through the network.
        features: The three continuous feature axes.
        output: For a regression network, the output that colours the prediction manifolds,
            without ``y``, its predictions over ``X`` also set their colour scale.
        block: The name of the manifold input block.
        volume_resolution: Grid points per axis for the inlier fade.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The 3-D subplot's row in ``fig``.
        col: The 3-D subplot's column in ``fig``.

    Raises:
        ValueError: If the input is not a manifold input, its ``subspace_dimension`` is
            above 2, ``features`` does not name three different continuous features, or
            ``output`` is not an output of the network.
    """
    C = network.inspect("continuous_centroids")[block]
    bases = network.inspect("manifold_projectors")[block]
    axes = list(features)
    if len(axes) != 3 or len(set(axes)) != 3 or not all(0 <= i < C.shape[1] for i in axes):
        raise ValueError(
            f"features must name three different continuous features, from 0 to {C.shape[1] - 1}"
        )
    if bases.shape[2] > 2:
        raise ValueError(
            f"plot_manifold draws lines and planes only; block {block!r} has "
            f"subspace_dimension={bases.shape[2]}, which cannot be drawn in three dimensions"
        )
    figure, at = _target(fig, row, col)
    scored = _scores_available(network)
    floor = float(network.score_samples(X).min()) if scored else 0.0
    low, high = X[:, axes].amin(0), X[:, axes].amax(0)
    box = torch.stack([low - 0.25 * (high - low), high + 0.25 * (high - low)])
    classification = network.schema.task == "classification"
    if classification:
        classes = network.predict(X[:1]).shape[1]
        shade = {}
    else:
        outputs = network.predict(X[:1]).shape[1]
        if not 0 <= output < outputs:
            raise ValueError(f"output must be from 0 to {outputs - 1}")
        values = _numpy(network.predict(X)[:, output] if y is None else y).reshape(-1)
        shade = dict(cmin=float(values.min()), cmax=float(values.max()))

    # Manifolds: rays from each centroid along its plane, each running out to where the plane
    # stops being feasible (another cluster wins, or the score falls below the floor).
    d = bases.shape[2]
    reach = float(torch.linalg.vector_norm(box[1] - box[0]))
    if d == 2:
        angle = torch.linspace(0, 2 * torch.pi, _DIRECTIONS, dtype=X.dtype, device=X.device)
        rays = torch.stack([angle.cos(), angle.sin()], dim=1)
    else:
        rays = torch.tensor([[1.0], [-1.0]], dtype=X.dtype, device=X.device)
    fraction = torch.linspace(0, 1, _STEPS, dtype=X.dtype, device=X.device)

    def last_feasible(k: int, on_plane: torch.Tensor, start: torch.Tensor, stop: torch.Tensor):
        """Per ray, the farthest sampled distance in [start, stop] before the plane ends."""
        t = start[:, None] + (stop - start)[:, None] * fraction  # (R, steps)
        points = C[k] + (t[..., None] * on_plane[:, None, :]).reshape(-1, C.shape[1])
        gamma = network.predict_with_details(points, details=("affiliations",)).affiliations
        assert gamma is not None
        feasible = gamma[block].argmax(dim=1) == k
        feasible &= ((points[:, axes] >= box[0]) & (points[:, axes] <= box[1])).all(dim=1)
        if scored:
            feasible &= network.score_samples(points) >= floor
        feasible = feasible.reshape(t.shape)
        first_out = torch.where(feasible.all(dim=1), _STEPS, (~feasible).int().argmax(dim=1))
        return t.gather(1, (first_out - 1).clamp(min=0)[:, None]).squeeze(1)

    step = reach / (_STEPS - 1)
    for k in range(len(C)):
        on_plane = rays @ bases[k][:, :d].T  # (R, D) unit directions in the plane
        coarse = last_feasible(
            k, on_plane, torch.zeros_like(rays[:, 0]), torch.full_like(rays[:, 0], reach)
        )
        radius = last_feasible(k, on_plane, coarse, coarse + step)
        rings = torch.linspace(0, 1, _RINGS, dtype=X.dtype, device=X.device)
        surface = C[k] + rings[:, None, None] * (radius[:, None] * on_plane)[None]  # (rings, R, D)
        score = network.score_samples(surface.reshape(-1, C.shape[1])) if scored else None
        xyz = _numpy(surface[..., axes])
        colour = qualitative.Dark24[k % len(qualitative.Dark24)]
        legend = dict(
            name="Manifolds", legendgroup="manifolds", showlegend=k == 0, hoverinfo="name"
        )
        predicted = dict(
            name="Manifolds (prediction)",
            legendgroup="manifolds_prediction",
            showlegend=k == 0,
            visible="legendonly",
            hoverinfo="name",
        )
        prediction = network.predict(surface.reshape(-1, C.shape[1]))
        if classification:
            tone = _blend(_numpy(prediction), CLASS_COLOURS)
            mesh_tone: dict[str, Any] = dict(vertexcolor=tone)
            line_tone: dict[str, Any] = dict()
        else:
            tone = _numpy(prediction[:, output])
            mesh_tone = dict(intensity=tone, colorscale="Viridis", showscale=False, **shade)
            line_tone = dict(colorscale="Viridis", **shade)
        vertices = xyz.reshape(-1, 3)
        if d == 2:
            figure.add_trace(
                go.Surface(
                    x=xyz[..., 0],
                    y=xyz[..., 1],
                    z=xyz[..., 2],
                    surfacecolor=None if score is None else _numpy(score).reshape(_RINGS, -1),
                    cmin=0,
                    cmax=1,
                    colorscale=[[0, colour], [1, colour]],
                    showscale=False,
                    opacity=None if scored else 0.6,
                    opacityscale=[[0, 0.1], [0.5, 0.8], [1, 0.8]] if scored else None,
                    **legend,
                ),
                **at,
            )
            first, second, third = _grid_faces(*xyz.shape[:2])
            figure.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=first,
                    j=second,
                    k=third,
                    opacity=0.8,
                    **mesh_tone,
                    **predicted,
                ),
                **at,
            )
        else:
            ends = xyz[-1]
            # One path from the far end of ray -1, through the centroid, to the end of ray +1.
            path = np.concatenate([np.arange(_RINGS)[::-1] * 2 + 1, np.arange(1, _RINGS) * 2])
            figure.add_trace(
                go.Scatter3d(
                    x=ends[:, 0],
                    y=ends[:, 1],
                    z=ends[:, 2],
                    mode="lines",
                    line=dict(color=colour, width=6),
                    **legend,
                ),
                **at,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=vertices[path, 0],
                    y=vertices[path, 1],
                    z=vertices[path, 2],
                    mode="lines",
                    line=dict(width=8, color=np.asarray(tone)[path].tolist(), **line_tone),
                    **predicted,
                ),
                **at,
            )
    centres = _numpy(C[:, axes])
    figure.add_trace(
        go.Scatter3d(
            x=centres[:, 0],
            y=centres[:, 1],
            z=centres[:, 2],
            mode="markers",
            marker=dict(size=4, color="black", symbol="diamond"),
            name="Manifolds",
            legendgroup="manifolds",
            showlegend=False,
            text=[f"cluster {k}" for k in range(len(C))],
            hoverinfo="text",
        ),
        **at,
    )

    # Observations: coloured by target, larger with larger instance weight.
    rows = _numpy(X[:, axes])
    size: Any = 4
    try:
        weights = _numpy(network.inspect("training_instance_weights")[block])
    except ValueError:
        weights = None
    if weights is not None and len(weights) == len(rows):
        size = 2 + 8 * np.sqrt(weights / max(weights.max(), np.finfo(float).tiny))
    observed: dict[str, Any] = dict(size=size)
    if y is not None and classification:
        labels = _numpy(y).astype(int).reshape(-1)
        observed.update(color=_blend(np.eye(classes)[labels], CLASS_COLOURS))
    elif y is not None:
        colorbar = dict(title="target", len=0.45, y=0.75, thickness=10)
        observed.update(color=values, colorscale="Viridis", colorbar=colorbar, **shade)
    figure.add_trace(
        go.Scatter3d(
            x=rows[:, 0],
            y=rows[:, 1],
            z=rows[:, 2],
            mode="markers",
            name="Observations",
            marker=observed,
        ),
        **at,
    )

    # Probes: where each starts, and where the network reconstructs it.
    if probes is not None:
        landing = network.predict_with_details(probes, details=("reconstruction",)).reconstruction
        assert landing is not None
        start, end = _numpy(probes[:, axes]), _numpy(landing.continuous[:, axes])
        score = _numpy(network.score_samples(probes)) if scored else np.full(len(start), 0.5)
        segments = np.stack([start, end, np.full_like(start, np.nan)], axis=1).reshape(-1, 3)
        figure.add_trace(
            go.Scatter3d(
                x=start[:, 0],
                y=start[:, 1],
                z=start[:, 2],
                mode="markers",
                name="Probes (origin)",
                visible="legendonly",
                marker=dict(size=2, color="grey", opacity=0.4),
            ),
            **at,
        )
        figure.add_trace(
            go.Scatter3d(
                x=segments[:, 0],
                y=segments[:, 1],
                z=segments[:, 2],
                mode="lines",
                name="Probes (movement)",
                visible="legendonly",
                hoverinfo="skip",
                line=dict(color="rgba(120,120,120,0.25)", width=1),
            ),
            **at,
        )
        figure.add_trace(
            go.Scatter3d(
                x=end[:, 0],
                y=end[:, 1],
                z=end[:, 2],
                mode="markers",
                name="Probes (end)",
                marker=dict(
                    size=1 + 9 * score,
                    color=score,
                    colorscale="YlOrRd",
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="inlier", len=0.45, y=0.25, thickness=10),
                    showscale=scored,
                ),
                text=[f"inlier {s:.2f}" for s in score],
                hoverinfo="text",
            ),
            **at,
        )

    # Inlier fade: the score over a grid of the whole box.
    if scored:
        cube = _on_grid(X, box, volume_resolution, axes)
        figure.add_trace(
            go.Volume(
                x=_numpy(cube[:, 0]).astype(np.float32),
                y=_numpy(cube[:, 1]).astype(np.float32),
                z=_numpy(cube[:, 2]).astype(np.float32),
                value=_numpy(network.score_samples(cube)).astype(np.float32),
                visible="legendonly",
                showlegend=True,
                showscale=False,
                hoverinfo="skip",
                name="Inlier fade",
                isomin=0.01,
                isomax=1.0,
                surface_count=12,
                colorscale=[[0, "#9ecae1"], [1, "#08306b"]],
                opacity=0.4,
                opacityscale=[[0, 0], [0.05, 0.3], [0.5, 0.8], [1, 1]],
            ),
            **at,
        )

    low_edge, high_edge = _numpy(box[0]), _numpy(box[1])
    figure.update_scenes(
        aspectmode="data",
        dragmode="turntable",
        **{
            f"{axis}axis": dict(title=f"feature {axes[i]}", range=[low_edge[i], high_edge[i]])
            for i, axis in enumerate("xyz")
        },
        **at,
    )
    figure.update_layout(legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=40, b=0))
    return figure
