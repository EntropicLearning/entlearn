"""The decision map of a classifier over two features, with its confidence map."""

from __future__ import annotations

import base64
import struct
import zlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from entlearn.network.queries import _scores_available
from entlearn.plotting.basic import (
    CLASS_COLOURS,
    _blend,
    _equal_aspect,
    _numpy,
    _palette,
)

if TYPE_CHECKING:
    from entlearn import Network


def _png(rgb: np.ndarray) -> str:
    """Encode an RGB image as a PNG data URI, first row first."""
    height, width, _ = rgb.shape
    pixels = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    raw = b"".join(b"\x00" + line.tobytes() for line in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw))
    return "data:image/png;base64," + base64.b64encode(png + chunk(b"IEND", b"")).decode()


def plot_decision(
    network: Network,
    X: torch.Tensor,
    y: Any = None,
    *,
    X_cat: Sequence[torch.Tensor] | None = None,
    features: tuple[int, int] = (0, 1),
    resolution: int = 40,
) -> go.Figure:
    """Plot which class a classifier predicts across two features, and how confidently.

    Each class has a colour, and the map mixes them in proportion to the predicted
    probabilities.
    The other continuous features
    are held at their medians over ``X``, and each categorical feature at its most
    frequent category. When the network learned instance weights, a second
    panel shows ``network.score_samples`` over the same grid.

    Args:
        network: A fitted classification network.
        X: Observations on the network's device and dtype; they set the plotted range and
            are drawn on top.
        y: Their class labels, which colour the observations; by default the predicted
            classes.
        X_cat: Their categorical features, one tensor of category codes per feature, for a
            network fitted with categorical features.
        features: The two features on the axes.
        resolution: Grid points per axis. The browser smooths the map between them.

    Raises:
        ValueError: If the network is not a classifier.
    """
    if network.schema.task != "classification":
        raise ValueError("plot_decision needs a classification network")
    first, second = features
    low, high = X[:, [first, second]].amin(0), X[:, [first, second]].amax(0)
    low, high = low - 0.1 * (high - low), high + 0.1 * (high - low)
    along = [
        torch.linspace(float(low[i]), float(high[i]), resolution, dtype=X.dtype) for i in (0, 1)
    ]
    gx, gy = torch.meshgrid(*along, indexing="xy")
    grid = X.median(dim=0).values.repeat(gx.numel(), 1)
    grid[:, first], grid[:, second] = gx.flatten().to(X.device), gy.flatten().to(X.device)
    grid_cat = None
    if X_cat is not None:
        grid_cat = [codes.mode().values.repeat(len(grid)) for codes in X_cat]

    probabilities = _numpy(network.predict(grid, X_cat=grid_cat))
    classes = probabilities.shape[1]
    rgb = (probabilities @ _palette(CLASS_COLOURS, classes)).reshape(resolution, resolution, 3)
    xs, ys = _numpy(along[0]), _numpy(along[1])
    scored = _scores_available(network)
    fig = make_subplots(
        rows=1,
        cols=2 if scored else 1,
        horizontal_spacing=0.12,
        subplot_titles=("Predicted class", "Confidence") if scored else ("Predicted class",),
    )
    fig.add_trace(
        go.Image(
            source=_png(rgb),
            zsmooth="fast",
            x0=xs[0],
            dx=xs[1] - xs[0],
            y0=ys[0],
            dy=ys[1] - ys[0],
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            z=probabilities.argmax(axis=1).reshape(resolution, resolution),
            x=xs,
            y=ys,
            contours=dict(coloring="none", start=0.5, end=classes - 1.5, size=1),
            line=dict(color="black", width=1),
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    if scored:
        fig.add_trace(
            go.Contour(
                z=_numpy(network.score_samples(grid, X_cat=grid_cat)).reshape(
                    resolution, resolution
                ),
                x=xs,
                y=ys,
                colorscale="YlOrRd",
                zmin=0,
                zmax=1,
                hoverinfo="skip",
                colorbar=dict(title="inlier", len=0.6, thickness=8),
            ),
            row=1,
            col=2,
        )

    points = _numpy(X)[:, [first, second]]
    if y is None:
        labels = _numpy(network.predict(X, X_cat=X_cat)).argmax(axis=1)
    else:
        labels = _numpy(y).astype(int)
    size: Any = 7 if not scored else 4 + 12 * _numpy(network.score_samples(X, X_cat=X_cat))
    for col in (1, 2) if scored else (1,):
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=_blend(np.eye(classes)[labels], CLASS_COLOURS),
                    size=size,
                    line=dict(width=1, color="black"),
                ),
                text=[f"class {c}" for c in labels],
                hoverinfo="text",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text=f"feature {first}", range=[xs[0], xs[-1]], row=1, col=col)
        fig.update_yaxes(title_text=f"feature {second}", range=[ys[0], ys[-1]], row=1, col=col)
        _equal_aspect(fig, 1, col)
    return fig
