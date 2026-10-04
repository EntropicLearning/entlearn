import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", css_file="entlearn.css")


@app.cell
def _():
    import numpy as np
    import torch
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import minmax_scale

    return make_circles, minmax_scale, np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Deeper networks

    So far, we only showcased *shallow* networks with a single block connected to a head. In this notebook, we will include a hidden block and inspect what is learned.

    ## Dataset

    We will use two concentric circles with an extra noisy dimension.
    """)
    return


@app.cell
def _(make_circles, minmax_scale, np, plot_data):
    X, y = make_circles(n_samples=500, noise=0.08, factor=0.1, random_state=7)
    X = np.column_stack([X, np.random.default_rng(7).uniform(size=len(X))])
    X = minmax_scale(X)

    plot_data(X, y)
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recipe

    The recipe is similar to what we used so far, with a `Hidden` block between the input and the head. The input uses 7 clusters, and the hidden block 3.

    `coupling` and `delta` should be set for each connection. The latter represents the strength of the connection. `coupling` will be explained later.
    """)
    return


@app.cell
def _(X, torch, y):
    from entlearn import ClassificationHead, Hidden, Input, Network, Recipe

    recipe = Recipe.chain(
        Input(K=7, epsilon=0.02, epsilon_D=0.015, epsilon_T=0.2),
        Hidden(K=3, epsilon=0.1),
        ClassificationHead(),
        coupling="M",
        delta=[1, 5],
    )
    network = Network.fit(recipe, torch.tensor(X), torch.tensor(y), n_inits=10, seed=42)
    network
    return ClassificationHead, Hidden, Input, Network, Recipe, network


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As for previous notebooks, it is a good idea to check which features are taken into account by the input block.
    """)
    return


@app.cell
def _(network):
    from entlearn.plotting import plot_feature_importance

    plot_feature_importance(
        network.inspect("feature_weights")["input"], names=["x1", "x2", "noise"]
    ).update_traces(marker_color="orange")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The centroids learned by the input block are positioned along the circles.

    We can visualize the original points (colored using their labels) on a 6-dimensional simplex. It is drawn on the right as a polygon, with one corner per cluster. This constitutes the input for the hidden block, which never observes the original coordinates.
    """)
    return


@app.cell
def _(X, network, plot_convex, y):
    centroids = network.inspect("continuous_centroids")["input"]
    gamma_1 = network.inspect("training_affiliations")["input"]

    plot_convex(X, y, centroids, gamma_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next figure shows the affiliations over the layers. Each cluster in each block has a color, and the datapoints are colored using their affiliations. Soft affiliations are reflected in smooth gradients between the different centroids. The middle panel shows how the hidden block has merged all the clusters along the outer circle as a single one.
    The rightmost panel illustrates the predicted classes.
    """)
    return


@app.cell
def _(network, plot_layers):
    plot_layers(network)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can look at the decision map: each class has a colour, and the map blends them in proportion to the predicted probabilities. Since this network learned instance weights, the right panel also shows how typical each region is of the training data.
    """)
    return


@app.cell
def _(X, network, torch, y):
    from entlearn.plotting import plot_decision

    plot_decision(network, torch.tensor(X), y)
    return (plot_decision,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Coupling

    `coupling` sets which way the transition matrix sums to one. The default is `M`, so the mass of each input cluster is spread over the hidden clusters. The alternative is `S`, where hidden clusters are a mixture of the input clusters.

    Let us now fit the same data with an `S` coupling and a wider hidden block.
    """)
    return


@app.cell
def _(
    ClassificationHead,
    Hidden,
    Input,
    Network,
    Recipe,
    X,
    plot_convex,
    torch,
    y,
):
    recipe_S = Recipe.chain(
        Input(K=7, epsilon=0.02, epsilon_D=0.016, epsilon_T=0.2),
        Hidden(K=20, epsilon=0.5),
        ClassificationHead(),
        coupling="S",
        delta=[5, 20],
    )
    network_S = Network.fit(recipe_S, torch.tensor(X), torch.tensor(y), n_inits=50, seed=42)

    centroids_S = network_S.inspect("continuous_centroids")["input"]
    gamma_1_s = network_S.inspect("training_affiliations")["input"]

    plot_convex(X, y, centroids_S, gamma_1_s)
    return (network_S,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the `S` coupling, every training point is assigned to a single input cluster, so all the points sit on the corners of the simplex.
    """)
    return


@app.cell
def _(network_S, plot_layers):
    plot_layers(network_S)
    return


@app.cell
def _(X, network_S, plot_decision, torch, y):
    plot_decision(network_S, torch.tensor(X), y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, both ways fit the data well on this problem. Note that the `S` network asked for 20 hidden clusters, but only 8 of them end up being used.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The difference between the two couplings is clearest in the transition matrices themselves. With `M`, each column (an input cluster) sums to one: every input cluster spreads its mass over the hidden clusters. With `S`, each row (a hidden cluster) sums to one: every hidden cluster is a mixture of input clusters.
    """)
    return


@app.cell
def _(network, network_S, plot_transitions):
    plot_transitions(network, network_S)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utility cells
    The cells below define additional imports and plotting helpers. They are not relevant for the demonstration, but are left for the curious reader.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo


@app.cell(hide_code=True)
def _(go):
    def plot_data(X, y):
        fig = go.Figure(
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 2],
                z=X[:, 1],
                mode="markers",
                marker=dict(color=y, colorscale=[[0, "blue"], [1, "red"]], size=3),
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="x1", yaxis_title="noise", zaxis_title="x2", dragmode="turntable"
            ),
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    return (plot_data,)


@app.cell(hide_code=True)
def _(go, make_subplots, np):
    def plot_convex(X, y, centroids, gamma):
        centroids = centroids.numpy()
        K = len(centroids)
        around = np.arctan2(centroids[:, 1] - 0.5, centroids[:, 0] - 0.5)
        angles = 2 * np.pi * np.argsort(np.argsort(around)) / K
        vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        points = gamma.numpy() @ vertices
        numbers = [str(k + 1) for k in range(K)]
        colours = dict(color=y, colorscale=[[0, "blue"], [1, "red"]], size=5)
        reference = dict(color="limegreen", size=18, line=dict(width=1, color="black"))
        ring = vertices[np.argsort(angles)]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Data", f"{K - 1}-dimensional simplex"))
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=colours), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers+text",
                text=numbers,
                marker=reference,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.append(ring[:, 0], ring[0, 0]),
                y=np.append(ring[:, 1], ring[0, 1]),
                mode="lines",
                fill="toself",
                fillcolor="rgba(144, 238, 144, 0.4)",
                line=dict(color="black"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", marker=colours), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
                mode="markers+text",
                text=numbers,
                marker=reference,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_xaxes(scaleanchor="y2", scaleratio=1, visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        fig.update_layout(showlegend=False, height=400)
        return fig

    return (plot_convex,)


@app.cell(hide_code=True)
def _(X, make_subplots, torch):
    from entlearn.plotting import plot_affiliations

    def plot_layers(network):
        data = torch.tensor(X)
        training = network.inspect("training_affiliations")
        centroids = network.inspect("continuous_centroids")["input"]
        at_centroids = network.predict_with_details(centroids, details=("affiliations",))
        # The data show their fitted affiliations. The centroids have none, so each takes its own
        # cluster, then the hidden cluster and class the network gives it.
        layers = [
            (data, [training["input"], training["hidden_1"], network.predict(data)]),
            (
                centroids,
                [
                    torch.eye(len(centroids), dtype=centroids.dtype),
                    at_centroids.affiliations["hidden_1"],
                    at_centroids.prediction,
                ],
            ),
        ]
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=("Input clusters", "Hidden clusters", "Classes")
        )
        for points, panels in layers:
            for col, gamma in enumerate(panels, start=1):
                plot_affiliations(points, gamma, classes=col == 3, fig=fig, row=1, col=col)
        for trace in fig.data[3:]:
            trace.marker.update(symbol="diamond", size=12, line=dict(width=1, color="black"))
        fig.update_layout(height=380)
        return fig

    return (plot_layers,)


@app.cell(hide_code=True)
def _(make_subplots):
    from entlearn.plotting import plot_theta

    def plot_transitions(network_M, network_S):
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.12,
            subplot_titles=("M coupling: columns sum to one", "S coupling: rows sum to one"),
        )
        for col, network in enumerate([network_M, network_S], start=1):
            plot_theta(
                network.inspect("transition_matrices")["input_to_hidden_1"],
                x_title="input cluster",
                y_title="hidden cluster",
                fig=fig,
                row=1,
                col=col,
            )
        fig.update_traces(zmin=0, zmax=1)
        fig.update_traces(showscale=False, col=1)
        fig.update_layout(height=450)
        return fig

    return (plot_transitions,)


if __name__ == "__main__":
    app.run()
