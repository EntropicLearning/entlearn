import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", css_file="entlearn.css")


@app.cell
def _():
    import numpy as np
    import optuna
    import pandas as pd
    from optuna.distributions import FloatDistribution, IntDistribution
    from optuna_integration import OptunaSearchCV
    from scipy.stats import loguniform
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        FloatDistribution,
        GridSearchCV,
        IntDistribution,
        OptunaSearchCV,
        RandomizedSearchCV,
        clone,
        loguniform,
        np,
        optuna,
        pd,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Selecting a model and hyperparameters

    The standard scikit-learn searches, such as `GridSearchCV` and `RandomizedSearchCV`, can be used with EON estimators, as well as automatic approaches such as `OptunaSearchCV`.
    In this notebook, we will demonstrate how to do it with a simple example by tuning two parameters only, `K` and `epsilon`.
    *note: in real life you likely need to tune more, which can be performed in the same way*.

    ## The searches
    We will compare:

    - `GridSearchCV`, with a grid of fixed values
    - `RandomizedSearchCV`, which draws the settings at random
    - `OptunaSearchCV`, which uses a Tree-structured Parzen Estimator to concentrate its trials near good results

    Every search uses 5 initializations: each estimator uses `n_inits=5`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    search_form = (
        mo.md("""
    Random and Optuna searches with {trials} trials
    {repeats}
    """)
        .batch(
            trials=mo.ui.slider(20, 200, step=20, value=100),
            repeats=mo.ui.slider(5, 10, value=5, label="Repeated experiments"),
        )
        .form(submit_button_label="Run the searches")
    )
    search_form
    return (search_form,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setting up the searches

    All three searches tune the same estimator on the same two folds, and are first run on one split of the data.
    """)
    return


@app.cell
def _(
    FloatDistribution,
    GridSearchCV,
    IntDistribution,
    OptunaSearchCV,
    RandomizedSearchCV,
    clone,
    loguniform,
    np,
    optuna,
    search_form,
):
    from entlearn import ClassificationHead, Input, Recipe
    from entlearn.scikit_adapter import EONClassifier

    settings = search_form.value or dict(trials=100, repeats=5)
    estimator_template = EONClassifier(
        Recipe.chain(
            Input(K=4, epsilon=0.015, epsilon_D=0.01, epsilon_T=0.15, name="features"),
            ClassificationHead(),
        ),
        n_inits=5,
    )
    K, epsilon = "recipe__blocks__features__K", "recipe__blocks__features__epsilon"
    options = dict(scoring="accuracy", error_score="raise", n_jobs=1)

    def make_searches(seed, folds):
        estimator = clone(estimator_template).set_params(random_state=seed)
        grid = GridSearchCV(
            estimator,
            {K: list(range(3, 13)), epsilon: np.geomspace(0.003, 0.06, 10).round(4).tolist()},
            cv=folds,
            **options,
        )
        randomized = RandomizedSearchCV(
            estimator,
            {K: list(range(3, 13)), epsilon: loguniform(0.003, 0.06)},
            n_iter=settings["trials"],
            cv=folds,
            random_state=seed,
            **options,
        )
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
        tpe = OptunaSearchCV(
            estimator,
            {K: IntDistribution(3, 12), epsilon: FloatDistribution(0.003, 0.06, log=True)},
            study=optuna.create_study(direction="maximize", sampler=sampler),
            n_trials=settings["trials"],
            cv=folds,
            random_state=seed,
            **options,
        )

        return {"Grid": grid, "Random": randomized, "Optuna": tpe}

    return (
        ClassificationHead,
        EONClassifier,
        Input,
        Recipe,
        estimator_template,
        make_searches,
        settings,
    )


@app.cell
def _(make_searches, split_experiment):
    X_train, y_train, _, _, folds = split_experiment(41)
    first_searches = {
        name: search.fit(X_train, y_train) for name, search in make_searches(41, folds).items()
    }
    return X_train, first_searches, folds, y_train


@app.cell(hide_code=True)
def _(first_searches, make_searches, mo, pd, settings, split_experiment):
    records = []
    for seed in mo.status.progress_bar(
        range(41, 41 + settings["repeats"]), title="Repeat searches"
    ):
        dev_X, dev_y, test_X, test_y, cv = split_experiment(seed)
        fitted = (
            first_searches
            if seed == 41
            else {
                name: search.fit(dev_X, dev_y) for name, search in make_searches(seed, cv).items()
            }
        )
        for name, search in fitted.items():
            for evaluation, value in [
                ("Train", search.best_estimator_.score(dev_X, dev_y)),
                ("Validation", search.best_score_),
                ("Test", search.best_estimator_.score(test_X, test_y)),
            ]:
                records.append(dict(Seed=seed, Method=name, Evaluation=evaluation, Accuracy=value))
    repeated_search_results = pd.DataFrame(records)
    return (repeated_search_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Results

    Each search is repeated on new splits of the data. *Train* is the accuracy of the selected model on the 100 rows used for the search, *Validation* its mean accuracy over the two validation folds, and *Test* its accuracy on the rows held out from the search.
    """)
    return


@app.cell
def _(mo, plot_results, repeated_search_results):
    mo.ui.plotly(plot_results(repeated_search_results))
    return


@app.cell(hide_code=True)
def _(first_searches, mo, plot_landscape):
    grid_search = first_searches["Grid"]
    selected = {key.rsplit("__", 1)[-1]: value for key, value in grid_search.best_params_.items()}
    mo.vstack(
        [
            mo.md(
                f"### Which settings worked?\nGrid selected **K={selected['K']}, "
                f"epsilon={selected['epsilon']}**; validation accuracy **{grid_search.best_score_:.3f}**."
            ),
            mo.ui.plotly(
                plot_landscape(
                    {name: first_searches[name] for name in ("Grid", "Random", "Optuna")}
                )
            ),
            mo.md(
                "Each dot is one setting tried. Grid places them evenly; random search draws them; "
                "Optuna draws its first 10 at random, then concentrates near good results."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Choosing the starting point

    The result of a fit also depends on where it starts. There are a few ways to pick a good starting point:

    - In the `Input` block, `centroid_strategy="greedy-kmeans++"` tries several candidates for each centroid, and `balanced=True` gives every class at least one centroid.
    - `n_inits` fits several starting points and keeps the best one. By default the best is the one with the lowest training loss; with `cv`, it is the one with the best validation score.
    - `initial_state` gives a fit a fixed starting point. Build a few with `Network.initialise` and add them to a grid as one more parameter: the search then picks a starting point together with `K` and `epsilon`, and every setting is compared from the same starts. Here the states are built only from the 50 rows that are in both training folds, so the validation rows do not shape them.
    """)
    return


@app.cell
def _(
    ClassificationHead,
    EONClassifier,
    Input,
    Recipe,
    X_train,
    folds,
    y_train,
):
    starting_point = EONClassifier(
        Recipe.chain(
            Input(K=4, epsilon=0.015, centroid_strategy="greedy-kmeans++", balanced=True),
            ClassificationHead(),
        ),
        n_inits=5,
        cv=folds,
        scoring="neg_log_loss",
        random_state=42,
    ).fit(X_train, y_train)

    starting_point.network_.diagnostics.initialisation_outcomes
    return


@app.cell
def _(GridSearchCV, X_train, clone, estimator_template, folds, np, y_train):
    import torch

    from entlearn import Network
    from entlearn.scikit_adapter import common_train_rows

    recipe = estimator_template.recipe
    states = [
        Network.initialise(
            recipe.replace_block("features", K=12),
            torch.as_tensor(X_train),
            torch.as_tensor(y_train),
            seed=seed,
            init_rows=torch.as_tensor(common_train_rows(folds)),
        )
        for seed in range(5)
    ]
    shared_grid = GridSearchCV(
        clone(estimator_template).set_params(n_inits=1, random_state=41),
        {
            "recipe__blocks__features__K": list(range(3, 13)),
            "recipe__blocks__features__epsilon": np.geomspace(0.003, 0.06, 10).round(4).tolist(),
            "initial_state": states,
        },
        cv=folds,
        scoring="accuracy",
        error_score="raise",
    ).fit(X_train, y_train)

    shared_grid.best_score_, states.index(shared_grid.best_params_["initial_state"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utility cells
    The cells below define additional imports, plotting helpers, the data generator and the data splits. They are not relevant for the demonstration, but are left for the curious reader.
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
    def plot_results(results):
        fig = go.Figure()
        for metric in ["Train", "Validation", "Test"]:
            rows = results[results.Evaluation == metric]
            fig.add_trace(
                go.Box(
                    x=rows.Method,
                    y=rows.Accuracy,
                    name=metric,
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.3,
                )
            )
        fig.update_layout(
            height=360,
            boxmode="group",
            yaxis=dict(title="Accuracy", range=[0, 1.03]),
            margin=dict(l=55, r=20, t=35, b=45),
            legend=dict(orientation="h", y=1.12),
        )
        return fig

    return (plot_results,)


@app.cell(hide_code=True)
def _(go, make_subplots, np, pd):
    from scipy.interpolate import griddata

    def samples_of(search):
        if hasattr(search, "study_"):
            trials = search.study_.trials_dataframe()
            return (
                trials["params_recipe__blocks__features__epsilon"].to_numpy(dtype=float),
                trials["params_recipe__blocks__features__K"].to_numpy(dtype=float),
                trials["value"].to_numpy(dtype=float),
            )
        results = pd.DataFrame(search.cv_results_)
        return (
            results["param_recipe__blocks__features__epsilon"].to_numpy(dtype=float),
            results["param_recipe__blocks__features__K"].to_numpy(dtype=float),
            results["mean_test_score"].to_numpy(dtype=float),
        )

    def plot_landscape(searches):
        epsilon_mesh = np.geomspace(0.003, 0.06, 60)
        K_mesh = np.linspace(3, 12, 60)
        log_span = np.log(0.06 / 0.003)
        fig = make_subplots(
            rows=1,
            cols=len(searches),
            shared_yaxes=True,
            horizontal_spacing=0.04,
            subplot_titles=list(searches),
        )
        for col, search in enumerate(searches.values(), start=1):
            epsilon, K, score = samples_of(search)
            accuracy_map = griddata(
                (np.log(epsilon / 0.003) / log_span, (K - 3) / 9),
                score,
                tuple(np.meshgrid(np.log(epsilon_mesh / 0.003) / log_span, (K_mesh - 3) / 9)),
                method="linear",
            )
            fig.add_trace(
                go.Contour(
                    z=accuracy_map,
                    x=epsilon_mesh,
                    y=K_mesh,
                    zmin=0.5,
                    zmax=1,
                    colorscale="YlOrRd",
                    showscale=col == 1,
                    colorbar=dict(len=0.6, thickness=6),
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=epsilon,
                    y=K,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="white", line=dict(width=2, color="black"), size=8),
                    customdata=score,
                    hovertemplate="K=%{y}, epsilon=%{x:.4f}<br>Accuracy: %{customdata:.3f}"
                    "<extra></extra>",
                ),
                row=1,
                col=col,
            )
            fig.update_xaxes(title_text="epsilon", type="log", row=1, col=col)
        fig.update_yaxes(title_text="K", row=1, col=1)
        fig.update_layout(
            height=420, title="Mean accuracy over the 2 validation folds (first split)"
        )
        return fig

    return (plot_landscape,)


@app.cell(hide_code=True)
def _(np, train_test_split):
    def split_experiment(seed):
        X, y = make_worms(noise_dims=20, seed=7)
        train_val, test = train_test_split(
            np.arange(len(y)), train_size=100, stratify=y, random_state=seed
        )
        train, validation = train_test_split(
            np.arange(100), train_size=50, stratify=y[train_val], random_state=seed
        )
        a, b = train_test_split(
            validation, test_size=0.5, stratify=y[train_val][validation], random_state=seed
        )
        folds = [(np.concatenate([train, b]), a), (np.concatenate([train, a]), b)]
        return X[train_val], y[train_val], X[test], y[test], folds

    return (split_experiment,)


@app.function(hide_code=True)
def make_worms(noise_dims: int = 4, seed: int = 7):
    """Three worms along the diagonal (middle one class 1), plus `noise_dims` noise columns."""
    import torch

    g = torch.Generator().manual_seed(seed)

    along = torch.tensor([1.0, -1.0], dtype=torch.float64) / 2**0.5
    across = torch.tensor([1.0, 1.0], dtype=torch.float64) / 2**0.5
    points, labels = [], []
    for centre, label in [(0.25, 0), (0.5, 1), (0.75, 0)]:
        length = 0.16 * torch.randn(55, 1, generator=g, dtype=torch.float64)
        width = 0.0365 * torch.randn(55, 1, generator=g, dtype=torch.float64)
        points.append(centre + length * along + width * across)
        labels.append(torch.full((55,), label))
    X, y = torch.cat(points), torch.cat(labels)

    # Always draw all 32 noise columns, so the first ones do not depend on noise_dims.
    low, high = X.min(), X.max()
    noise = low + (high - low) * torch.rand(len(X), 32, generator=g, dtype=torch.float64)
    X = torch.cat([X, noise[:, :noise_dims]], dim=1)
    return ((X + 0.3) / 1.6).numpy(), y.numpy()


if __name__ == "__main__":
    app.run()
