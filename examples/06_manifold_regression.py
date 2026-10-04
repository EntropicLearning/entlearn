import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", css_file="entlearn.css")


@app.cell
def _():
    import numpy as np
    import torch
    from sklearn.datasets import make_s_curve

    return make_s_curve, np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Learning with manifolds

    The goal of this notebook is to understand and use a `ManifoldInput` block for a regression problem. The data lie on an S-shaped surface in 3 dimensions, while the target is the position along the S.

    A `ManifoldInput` is similar to an `Input`, but each cluster has a centre as well as a local plane. Many linear manifolds can be put together to tile a nonlinear surface like the curved one in this example. The idea behind this block is Entropy-Optimal Manifold Clustering (EOMC).

    ## Dataset

    We draw 300 training points and 300 held-out points from the S curve of scikit-learn. The colour shows the target, the position along the S.
    """)
    return


@app.cell
def _(make_s_curve, np, torch):
    train_array, train_t = make_s_curve(300, random_state=17)
    holdout_array, holdout_t = make_s_curve(300, random_state=23)
    X_train = torch.tensor(train_array, dtype=torch.float64)
    y_train = torch.tensor(train_t[:, None], dtype=torch.float64)
    X_holdout = torch.tensor(holdout_array, dtype=torch.float64)
    domain_low = np.array([-1.0, 0.0, -2.0])
    domain_high = np.array([1.0, 2.0, 2.0])
    padding = 0.5 * (domain_high - domain_low)
    probe_array = np.random.default_rng(31).uniform(
        domain_low - padding, domain_high + padding, (1000, 3)
    )
    X_probe = torch.tensor(probe_array, dtype=torch.float64)
    return X_holdout, X_probe, X_train, holdout_t, train_t, y_train


@app.cell
def _(X_train, plot_data, train_t):
    plot_data(X_train, train_t)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recipe

    The `ManifoldInput`-specific parameters are:

    |Parameter name | Meaning|
    |---------------|--------|
    |`subspace_dimension`|The dimension of each local plane (1 for a line, 2 for a plane,...)|
    |`alpha`|How much distance from the centroid matters, on top of distance from the plane|

    The other parameters have the same meaning as in the `Input` block.
    """)
    return


@app.cell
def _(X_train, y_train):
    from entlearn import ManifoldInput, Network, Recipe, RegressionHead

    recipe = Recipe.chain(
        ManifoldInput(K=12, subspace_dimension=2, alpha=0.5, epsilon=0.03, epsilon_T=0.5),
        RegressionHead(),
    )
    network = Network.fit(recipe, X_train, y_train, seed=7, max_iter=60)
    return (network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Centroids

    Let us look at the fitted clusters from the side of the S. Each training point is coloured by its affiliations, and the diamonds mark the centroids. Every cluster covers a piece of the S.
    """)
    return


@app.cell
def _(X_train, network):
    from entlearn.plotting import plot_affiliations

    plot_affiliations(
        X_train,
        network.inspect("training_affiliations")["input"],
        network.inspect("continuous_centroids")["input"],
        features=(0, 2),
    )
    return


@app.cell(hide_code=True)
def _(holdout_prediction, holdout_t, mo, np):
    _correlation = np.corrcoef(holdout_prediction, holdout_t)[0, 1]
    mo.md(rf"""
    ## Predictions

    Back in 3D, the held-out points are coloured by their predicted position along the S. The prediction follows the true position closely: on held-out data, their correlation is {_correlation:.2f}.
    """)
    return


@app.cell
def _(X_holdout, network):
    from entlearn.plotting import plot_manifold

    holdout_prediction = network.predict(X_holdout)[:, 0]

    _figure = plot_manifold(network, X_holdout, holdout_prediction)
    _figure.update_traces(visible="legendonly", selector=dict(name="Manifolds"))
    _figure
    return holdout_prediction, plot_manifold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is possible to plot the manifolds: turning on `Manifolds` in the legend shows each manifold, colored by its identity.
    """)
    return


@app.cell
def _(X_holdout, holdout_prediction, network, plot_manifold):
    plot_manifold(network, X_holdout, holdout_prediction)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, we can also color the planes by the predictions along them.
    """)
    return


@app.cell
def _(X_holdout, holdout_prediction, network, plot_manifold):
    _figure = plot_manifold(network, X_holdout, holdout_prediction)
    _figure.update_traces(visible="legendonly", selector=dict(name="Manifolds"))
    _figure.update_traces(visible=True, selector=dict(name="Manifolds (prediction)"))
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reconstruction

    What happens for points that are not on the original manifold? We draw 1000 random points in a box around it and pass them to the network.

    The network projects each point onto its local planes, performing a *reconstruction*. Each grey line joins a random point to its reconstruction.
    The cloud shows the inlier score over the whole box: it is only dense close to the S.
    The reconstructed points are coloured and sized by `score_samples`. Most random points are far from the surface and score close to zero, so they appear pale; only the few near the S turn red.
    Note that a good reconstruction tells us the point is close to the surface, not that the prediction is correct.
    """)
    return


@app.cell
def _(X_probe, X_train, network, plot_manifold):
    _figure = plot_manifold(network, X_train, probes=X_probe)
    _figure.update_traces(visible="legendonly", selector=dict(name="Manifolds"))
    _figure.update_traces(visible="legendonly", selector=dict(name="Observations"))
    for _name in ("Probes (origin)", "Probes (movement)", "Inlier fade"):
        _figure.update_traces(visible=True, selector=dict(name=_name))
    _figure
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

    return go, mo


@app.cell(hide_code=True)
def _(go):
    def plot_data(X, t):
        fig = go.Figure(
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=t,
                    colorscale="Viridis",
                    colorbar=dict(title="target", thickness=10),
                ),
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="feature 0",
                yaxis_title="feature 1",
                zaxis_title="feature 2",
                aspectmode="data",
                dragmode="turntable",
            ),
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    return (plot_data,)


if __name__ == "__main__":
    app.run()
