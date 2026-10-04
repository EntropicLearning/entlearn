"""Every input centroid, and optionally the data, in parallel coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
import torch

from entlearn import ClassificationHead, Hidden, RegressionHead
from entlearn.helpers.reporting import feature_importances
from entlearn.plotting.basic import CLASS_COLOURS, CLUSTER_COLOURS, _numpy, _palette

if TYPE_CHECKING:
    from entlearn import Network

_DATA_ALPHA = 0.3


def _leads_to(network: Network, block: str) -> tuple[torch.Tensor, bool]:
    """Each input cluster's mix over the head's classes or outputs, through the transitions.

    Returns one column per input cluster, and whether the rows are classes.
    """
    blocks = {b.name: b for b in network.recipe.blocks}
    outgoing = {c.source: c for c in network.recipe.connections}
    K = network.inspect("continuous_centroids")[block].shape[0]
    lead = torch.eye(K, dtype=torch.float64)
    name = block
    while True:
        connection = outgoing[name]
        target = blocks[connection.target]
        match target:
            case Hidden():
                step = network.inspect("transition_matrices")[connection.name].to(lead)
            case ClassificationHead():
                step = network.inspect("head_parameters")[target.name]["theta"].to(lead)
            case RegressionHead():
                C_y = network.inspect("head_parameters")[target.name]["C_y"].to(lead)
                return C_y @ lead, False
            case _:
                raise ValueError(f"plot_parallel cannot follow block {target.name!r}")
        lead = step / step.sum(dim=0, keepdim=True).clamp_min(torch.finfo(lead.dtype).tiny) @ lead
        if isinstance(target, ClassificationHead):
            return lead, True
        name = target.name


def plot_parallel(
    network: Network,
    X: torch.Tensor,
    y: Any = None,
    *,
    X_cat: Sequence[torch.Tensor] | None = None,
    scaled: bool = False,
    show_data: bool = True,
    colour: Literal["cluster", "class"] = "cluster",
    names: Sequence[str] | None = None,
    categorical_names: Sequence[str] | None = None,
    block: str = "input",
) -> go.Figure:
    """Draw every input centroid across all features, with the data behind it.

    Each centroid is one line through one axis per continuous feature, one 0-to-1 axis per
    level of each categorical feature (grouped under a band), and then the head: the
    probability of each class, or each output. With ``y``, a last axis shows each
    observation's label or target, and for a centroid the class or output it leads to.
    The observations are drawn behind as transparent lines, and the feature weights are
    written under their axes. Drag along an axis to keep only the lines in that range.

    Args:
        network: A fitted network with a standard or manifold input.
        X: The observations' continuous features, on the network's device and dtype.
        y: Their labels or targets, drawn on a last axis.
        X_cat: Their categorical features, for a network fitted with categorical features.
        scaled: Multiply each feature axis by the square root of its weight and give all of
            them one range, so their spreads follow the distances the network uses.
        show_data: Draw the observations. Without them the axes keep the same ranges.
        colour: Colour the lines by ``"cluster"``: an observation by its most likely
            cluster; or, for a classifier, by ``"class"``: an observation by its label
            (its predicted class without ``y``) and a centroid by the class it leads to.
        names: Names of the continuous features; by default ``x0``, ``x1``, ...
        categorical_names: Names of the categorical features; by default ``c0``, ``c1``, ...
        block: The name of the input block.

    Raises:
        ValueError: If ``colour`` is not ``"cluster"`` or ``"class"``, ``"class"`` is
            asked of a regression network, or the names do not match the features.
    """
    if colour not in ("cluster", "class"):
        raise ValueError('colour must be "cluster" or "class"')
    X_cat = list(X_cat or [])
    C = network.inspect("continuous_centroids")[block]
    C_cats = network.inspect("categorical_centroids")[block] if network.schema.M_cat else ()
    K, T, D = C.shape[0], X.shape[0], C.shape[1]
    names = list(names) if names is not None else [f"x{d}" for d in range(D)]
    categorical_names = (
        list(categorical_names)
        if categorical_names is not None
        else [f"c{j}" for j in range(len(C_cats))]
    )
    if len(names) != D or len(categorical_names) != len(C_cats):
        raise ValueError(
            f"expected {D} continuous and {len(C_cats)} categorical feature names, "
            f"got {len(names)} and {len(categorical_names)}"
        )
    lead, classification = _leads_to(network, block)
    if colour == "class" and not classification:
        raise ValueError('colour="class" needs a classification network')
    result = network.predict_with_details(X, X_cat=X_cat or None, details=("affiliations",))
    assert result.affiliations is not None
    cluster = _numpy(result.affiliations[block]).argmax(axis=1)
    prediction = _numpy(result.prediction)
    weights = _numpy(feature_importances(network))

    # Each axis: its values for the T observations then the K centroids, and its weight.
    columns: dict[str, np.ndarray] = {"cluster": np.concatenate([cluster, np.arange(K)])}
    weight_of: dict[str, float] = {}
    levels: dict[str, str] = {}
    for d, name in enumerate(names):
        columns[name] = np.concatenate([_numpy(X[:, d]), _numpy(C[:, d])])
        weight_of[name] = float(weights[d])
    groups = []
    for j, (name, codes, C_cat) in enumerate(zip(categorical_names, X_cat, C_cats, strict=True)):
        one_hot = np.eye(C_cat.shape[1])[_numpy(codes).astype(int)]
        keys = []
        for level in range(C_cat.shape[1]):
            key = f"{name}={level}"
            columns[key] = np.concatenate([one_hot[:, level], _numpy(C_cat[:, level])])
            weight_of[key], levels[key] = float(weights[D + j]), str(level)
            keys.append(key)
        groups.append((name, float(weights[D + j]), keys))
    head = _numpy(lead).T  # (K, classes or outputs)
    shown = (
        range(1, head.shape[1]) if classification and head.shape[1] == 2 else range(head.shape[1])
    )
    for m in shown:
        key = f"P(class {m})" if classification else f"output {m}"
        columns[key] = np.concatenate([prediction[:, m], head[:, m]])
    if y is not None:
        labels = _numpy(y).reshape(T, -1)
        for m in range(labels.shape[1]):
            key = "label" if classification else f"target {m}"
            centroid = head.argmax(axis=1) if classification else head[:, m]
            columns[key] = np.concatenate([labels[:, m], centroid])

    # Ranges come from observations and centroids together, so hiding the data keeps them.
    keep = slice(None) if show_data else slice(T, None)
    scale = {key: np.sqrt(w) if scaled else 1.0 for key, w in weight_of.items()}
    shared = np.concatenate([columns[key] * scale[key] for key in weight_of])
    dimensions = []
    for key, values in columns.items():
        label = levels.get(key, key)
        dimension: dict[str, Any] = dict(label=label)
        if key in weight_of:
            values = values * scale[key]
            if scaled:
                dimension.update(label=f"√w·{label}", range=[shared.min(), shared.max()])
            elif key in levels:
                dimension.update(range=[0, 1])
            else:
                dimension.update(range=[values.min(), values.max()])
            dimension.update(tickformat=".2f")
        elif key == "cluster":
            dimension.update(tickvals=list(range(K)), range=[-0.5, K - 0.5])
        elif key.startswith("P("):
            dimension.update(range=[0, 1])
        elif key == "label":
            dimension.update(tickvals=list(range(head.shape[1])), range=[-0.1, head.shape[1] - 0.9])
        dimension.update(values=values[keep])
        dimensions.append(dimension)

    # One colour scale: the observations' colours made transparent, then the centroids'.
    if colour == "cluster":
        palette = _palette(CLUSTER_COLOURS, K)
        data_index, centroid_index = cluster, np.arange(K)
    else:
        palette = _palette(CLASS_COLOURS, head.shape[1])
        data_index = (
            _numpy(y).reshape(-1).astype(int) if y is not None else prediction.argmax(axis=1)
        )
        centroid_index = head.argmax(axis=1)
    n = len(palette)
    stops = [f"rgba({r:.0f},{g:.0f},{b:.0f},{_DATA_ALPHA})" for r, g, b in palette]
    stops += [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in palette]
    line_colour = np.concatenate([data_index, centroid_index + n])[keep]
    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=line_colour,
                colorscale=[[i / (2 * n - 1), c] for i, c in enumerate(stops)],
                cmin=0,
                cmax=2 * n - 1,
            ),
            dimensions=dimensions,
        )
    )

    # The weights under the feature axes; a band and a name under each categorical feature.
    order = list(columns)
    step = 1 / (len(order) - 1)
    note = dict(xref="paper", yref="paper", showarrow=False, font=dict(size=11))
    for name in names:
        fig.add_annotation(
            x=order.index(name) * step, y=-0.09, text=f"{weight_of[name]:.2f}", **note
        )
    for name, weight, keys in groups:
        first, last = order.index(keys[0]) * step, order.index(keys[-1]) * step
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=first - 0.4 * step,
            x1=last + 0.4 * step,
            y0=-0.03,
            y1=1.0,
            fillcolor="rgba(120,120,120,0.10)",
            line_width=0,
            layer="below",
        )
        fig.add_annotation(x=(first + last) / 2, y=-0.09, text=f"{weight:.2f}", **note)
        fig.add_annotation(
            x=(first + last) / 2, y=-0.15, text=f"<b>{name}</b>{' ≈' if scaled else ''}", **note
        )
    fig.add_annotation(
        x=-0.01,
        xanchor="right",
        y=-0.09,
        text="weight",
        **{**note, "font": dict(size=11, color="grey")},
    )
    fig.update_layout(height=600, margin=dict(l=70, r=70, t=60, b=90))
    return fig
