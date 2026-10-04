"""The figures behind ``Network.plot``: one row of panels per block."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, assert_never

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from entlearn import ClassificationHead, Hidden, Input, ManifoldInput, RegressionHead
from entlearn.helpers.reporting import feature_importances
from entlearn.plotting.basic import (
    plot_affiliations,
    plot_centroids,
    plot_feature_importance,
    plot_loss,
    plot_theta,
)
from entlearn.plotting.manifold import plot_manifold

if TYPE_CHECKING:
    from entlearn import Network

_Panel = tuple[str, str, Callable[[go.Figure, int, int], object]]


def _panels(
    network: Network,
    description: Input | ManifoldInput | Hidden | ClassificationHead | RegressionHead,
    X: torch.Tensor | None,
    affiliations: Mapping[str, torch.Tensor] | None,
) -> list[_Panel]:
    """List one block's panels as (title, subplot type, draw into fig/row/col)."""
    name = description.name
    blocks = {b.name: b for b in network.recipe.blocks}
    incoming = {c.target: c.name for c in network.recipe.connections}
    outgoing = {c.source: c.target for c in network.recipe.connections}
    gamma = None if affiliations is None else affiliations.get(name)
    match description:
        case Input():
            centroids = network.inspect("continuous_centroids")[name]
            weights = feature_importances(network)
            match blocks[outgoing[name]]:
                case Hidden(name=target):
                    theta = network.inspect("transition_matrices")[incoming[target]]
                case ClassificationHead(name=target):
                    theta = network.inspect("head_parameters")[target]["theta"]
                case _:
                    theta = None
            panels: list[_Panel] = [
                (
                    "feature weights",
                    "xy",
                    lambda f, r, c: plot_feature_importance(weights, fig=f, row=r, col=c),
                ),
            ]
            if centroids.shape[1] >= 2:
                panels.append(
                    (
                        "centroids",
                        "xy",
                        lambda f, r, c: plot_centroids(
                            centroids, X, theta=theta, affiliations=gamma, fig=f, row=r, col=c
                        ),
                    )
                )
            if X is not None and gamma is not None and centroids.shape[1] >= 2:
                panels.append(
                    (
                        "affiliations",
                        "xy",
                        lambda f, r, c: plot_affiliations(X, gamma, centroids, fig=f, row=r, col=c),
                    )
                )
            return panels
        case ManifoldInput():
            if X is None:
                raise ValueError("plotting a manifold input needs X_cont")
            participation = feature_importances(network)
            return [
                (
                    "feature participation",
                    "xy",
                    lambda f, r, c: plot_feature_importance(participation, fig=f, row=r, col=c),
                ),
                (
                    "manifolds",
                    "scene",
                    lambda f, r, c: plot_manifold(network, X, fig=f, row=r, col=c),
                ),
            ]
        case Hidden():
            theta = network.inspect("transition_matrices")[incoming[name]]
            panels = [
                (
                    "incoming transition",
                    "xy",
                    lambda f, r, c: plot_theta(theta, fig=f, row=r, col=c),
                )
            ]
            if X is not None and gamma is not None and X.shape[1] >= 2:
                panels.append(
                    (
                        "affiliations",
                        "xy",
                        lambda f, r, c: plot_affiliations(X, gamma, fig=f, row=r, col=c),
                    )
                )
            return panels
        case ClassificationHead():
            theta = network.inspect("head_parameters")[name]["theta"]
            return [
                (
                    "class transition",
                    "xy",
                    lambda f, r, c: plot_theta(theta, y_title="class", fig=f, row=r, col=c),
                )
            ]
        case RegressionHead():
            C_y = network.inspect("head_parameters")[name]["C_y"]
            return [
                (
                    "output centroids",
                    "xy",
                    lambda f, r, c: plot_theta(C_y, y_title="output", fig=f, row=r, col=c),
                )
            ]
        case _:
            assert_never(description)


def plot_network(
    network: Network,
    *,
    block: str | None,
    X_cont: torch.Tensor | None,
    X_cat: Sequence[torch.Tensor] | None,
) -> go.Figure:
    """Return the figure of one block, or of the loss and every block."""
    descriptions = {b.name: b for b in network.recipe.blocks}
    if block is not None and block not in descriptions:
        raise ValueError(f"unknown block {block!r}; expected one of: {', '.join(descriptions)}")
    affiliations = None
    if X_cont is not None:
        result = network.predict_with_details(X_cont, X_cat=X_cat, details=("affiliations",))
        affiliations = result.affiliations
    names = list(descriptions) if block is None else [block]
    rows = [_panels(network, descriptions[name], X_cont, affiliations) for name in names]
    titles = [
        [f"{name} · {title}" for title, _, _ in panels]
        for name, panels in zip(names, rows, strict=True)
    ]
    if block is None:
        rows.insert(
            0,
            [
                (
                    "loss",
                    "xy",
                    lambda f, r, c: plot_loss(
                        network.diagnostics.loss_history, fig=f, row=r, col=c
                    ),
                )
            ],
        )
        titles.insert(0, ["loss"])
    cols = max(len(panels) for panels in rows)
    specs = []
    for panels in rows:
        if len(panels) == 1:
            specs.append([{"type": panels[0][1], "colspan": cols}] + [None] * (cols - 1))
        else:
            specs.append([{"type": kind} for _, kind, _ in panels] + [None] * (cols - len(panels)))
    heights = [600 if any(kind == "scene" for _, kind, _ in panels) else 380 for panels in rows]
    fig = make_subplots(
        rows=len(rows),
        cols=cols,
        specs=specs,
        row_heights=heights,
        subplot_titles=[title for row_titles in titles for title in row_titles],
        vertical_spacing=0.3 / len(rows),
        horizontal_spacing=0.08,
    )
    for r, panels in enumerate(rows, start=1):
        for c, (_, _, draw) in enumerate(panels, start=1):
            draw(fig, r, c)
    fig.update_layout(height=sum(heights))
    return fig
