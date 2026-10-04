"""Plotly figures of fitted tensors, one panel each.

Each function takes plain tensors (or arrays) and returns a figure. Pass ``fig``, ``row``
and ``col`` to draw into one subplot of an existing figure instead, ``Network.plot``
builds its block figures this way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, qualitative

CLUSTER_COLOURS = np.array([hex_to_rgb(c) for c in qualitative.Dark24], dtype=float)
CLASS_COLOURS = np.array([hex_to_rgb(c) for c in qualitative.Plotly], dtype=float)
_TINY = np.finfo(float).tiny
CENTROID_MARKER = dict(size=14, symbol="diamond", line=dict(width=1, color="black"))


def _numpy(value: Any) -> np.ndarray:
    """Return a NumPy copy of a tensor or array-like."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=float)


def _target(fig: go.Figure | None, row: int | None, col: int | None) -> tuple[go.Figure, dict]:
    """Return the figure to draw into and the subplot keywords for it."""
    return (go.Figure() if fig is None else fig), dict(row=row, col=col)


def _equal_aspect(fig: go.Figure, row: int | None, col: int | None) -> None:
    """Give one subplot equal x and y scales."""
    anchor = "x" if row is None else fig.get_subplot(row, col).yaxis.anchor
    fig.update_yaxes(scaleanchor=anchor, scaleratio=1, row=row, col=col)


def _palette(colours: np.ndarray, count: int) -> np.ndarray:
    """Return ``count`` RGB rows, repeating the palette when it is too short."""
    return colours[np.arange(count) % len(colours)]


def _blend(weights: np.ndarray, colours: np.ndarray) -> list[str]:
    """Mix one palette colour per column of ``weights``, in proportion to each row."""
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), _TINY, None)
    return [
        f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in weights @ _palette(colours, weights.shape[1])
    ]


def plot_loss(
    loss_history: Sequence[float],
    *,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot the loss history.

    Args:
        loss_history: The loss values, as in ``network.diagnostics.loss_history``.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    figure, at = _target(fig, row, col)
    figure.add_trace(
        go.Scatter(y=list(loss_history), mode="lines+markers", name="loss", showlegend=False), **at
    )
    figure.update_xaxes(title_text="iteration", **at)
    figure.update_yaxes(title_text=r"$\mathcal{L}$", **at)
    return figure


def plot_feature_importance(
    weights: Any,
    names: Sequence[str] | None = None,
    *,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot one bar per feature weight.

    Args:
        weights: One weight per feature, such as
            ``network.inspect("feature_weights")["input"]`` or an estimator's
            ``feature_importances_``.
        names: Feature names in the same order; by default the feature indices.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    values = _numpy(weights)
    figure, at = _target(fig, row, col)
    labels = list(names) if names is not None else [str(d) for d in range(len(values))]
    figure.add_trace(go.Bar(x=labels, y=values, name="W", showlegend=False), **at)
    figure.update_xaxes(title_text="feature", type="category", **at)
    figure.update_yaxes(title_text=r"$w$", **at)
    return figure


def plot_centroids(
    centroids: Any,
    X: Any = None,
    *,
    theta: Any = None,
    affiliations: Any = None,
    wt: Any = None,
    features: tuple[int, int] = (0, 1),
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot the continuous centroids over two features, optionally with the data.

    Without ``theta`` each centroid has its own colour. With ``theta``, the transition
    matrix leaving this block, a centroid takes the colours of the clusters or classes
    it leads to, mixed in proportion. Observations with ``affiliations`` are coloured the
    same way through their clusters.

    Args:
        centroids: The centroids, one row per cluster.
        X: Observations to draw behind the centroids.
        theta: Transition matrix with one column per centroid and one row per target
            cluster or class.
        affiliations: The observations' affiliations, one column per centroid.
        wt: Instance weights of the observations; larger weights draw larger markers.
        features: The two feature axes.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    C = _numpy(centroids)[:, list(features)]
    figure, at = _target(fig, row, col)
    leads_to = None if theta is None else _numpy(theta).T
    if leads_to is None:
        centroid_colours = _blend(np.eye(len(C)), CLUSTER_COLOURS)
    else:
        centroid_colours = _blend(leads_to, CLASS_COLOURS)
    if X is not None:
        points = _numpy(X)[:, list(features)]
        colour: Any = "lightgrey"
        if affiliations is not None:
            gamma = _numpy(affiliations)
            if leads_to is None:
                colour = _blend(gamma, CLUSTER_COLOURS)
            else:
                leads = leads_to / np.clip(leads_to.sum(axis=1, keepdims=True), _TINY, None)
                colour = _blend(gamma @ leads, CLASS_COLOURS)
        size: Any = 6
        if wt is not None:
            weights = _numpy(wt)
            size = 3 + 9 * np.sqrt(weights / max(weights.max(), _TINY))
        figure.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                name="observations",
                showlegend=False,
                marker=dict(color=colour, size=size, line=dict(width=0.5, color="black")),
            ),
            **at,
        )
    figure.add_trace(
        go.Scatter(
            x=C[:, 0],
            y=C[:, 1],
            mode="markers",
            name="centroids",
            showlegend=False,
            marker=dict(color=centroid_colours, **CENTROID_MARKER),
            text=[f"cluster {k}" for k in range(len(C))],
            hoverinfo="text",
        ),
        **at,
    )
    figure.update_xaxes(title_text=f"feature {features[0]}", **at)
    figure.update_yaxes(title_text=f"feature {features[1]}", **at)
    _equal_aspect(figure, row, col)
    return figure


def plot_affiliations(
    X: Any,
    affiliations: Any,
    centroids: Any = None,
    *,
    classes: bool = False,
    features: tuple[int, int] = (0, 1),
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Colour observations by their affiliations over two features.

    Each cluster has a colour, and each observation mixes the colours of its clusters in
    proportion to its affiliations.

    Args:
        X: The observations.
        affiliations: Their affiliations, one row per observation and one column per
            cluster, from any block.
        centroids: The centroids to mark, one row per cluster, which must live in the same
            space as ``X``.
        classes: Whether the columns are classes, such as the probabilities from
            ``network.predict``; they then take the class colours of ``plot_decision``.
        features: The two features on the axes.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    points = _numpy(X)[:, list(features)]
    gamma = _numpy(affiliations)
    figure, at = _target(fig, row, col)
    palette, label = (CLASS_COLOURS, "class") if classes else (CLUSTER_COLOURS, "cluster")
    figure.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            name="observations",
            showlegend=False,
            marker=dict(color=_blend(gamma, palette), size=6),
            text=[f"{label} {k}" for k in gamma.argmax(axis=1)],
            hoverinfo="text",
        ),
        **at,
    )
    if centroids is not None:
        C = _numpy(centroids)[:, list(features)]
        figure.add_trace(
            go.Scatter(
                x=C[:, 0],
                y=C[:, 1],
                mode="markers",
                name="centroids",
                showlegend=False,
                marker=dict(color=_blend(np.eye(len(C)), CLUSTER_COLOURS), **CENTROID_MARKER),
            ),
            **at,
        )
    figure.update_xaxes(title_text=f"feature {features[0]}", **at)
    figure.update_yaxes(title_text=f"feature {features[1]}", **at)
    _equal_aspect(figure, row, col)
    return figure


def plot_theta(
    theta: Any,
    *,
    x_title: str = "source cluster",
    y_title: str = "target cluster",
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot a transition matrix as a heatmap, one column per source cluster, with a colour bar.

    Args:
        theta: The matrix, one row per target cluster (or class, or output) and one column
            per source cluster, as in ``network.inspect("transition_matrices")``.
        x_title: The title of the horizontal axis.
        y_title: The title of the vertical axis.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    values = _numpy(theta)
    figure, at = _target(fig, row, col)
    colorbar: dict[str, Any] = dict(thickness=8, len=0.8)
    if row is not None:
        # In a subplot, the colour bar sits beside its own panel instead of the figure's edge.
        axes = figure.get_subplot(row, col)
        (_, right), (bottom, top) = axes.xaxis.domain, axes.yaxis.domain
        colorbar.update(
            x=right + 0.01, xanchor="left", y=(bottom + top) / 2, len=0.8 * (top - bottom)
        )
    figure.add_trace(
        go.Heatmap(
            z=values,
            colorscale="Blues",
            colorbar=colorbar,
            showlegend=False,
            hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>%{{z:.3f}}<extra></extra>",
        ),
        **at,
    )
    figure.update_xaxes(title_text=x_title, dtick=1, **at)
    figure.update_yaxes(title_text=y_title, dtick=1, autorange="reversed", **at)
    return figure
