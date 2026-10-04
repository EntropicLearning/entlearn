# Model selection

Model selection chooses a model's settings from the data: for example, how many clusters each block has, or which temperatures it uses.
A search scores each candidate, usually on observations held out from its fit, and keeps the best.

The estimators support scikit-learn's search tools, so you can explore them using the same parameter paths introduced in [Package and Recipes](package.md#setting-parameters).

## GridSearchCV

`GridSearchCV` evaluates a parameter grid using cross-validation. It:

- Takes as input a grid of potential parameters
- Fits each combination of parameters on several training folds
- Evaluates each combination on the corresponding validation folds.

It then selects the combination with the highest mean validation score and, by default, fits it again on all the data supplied to the search.

Before searching, *set aside a test set*, or consider using [nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
The following small example illustrates the API. 

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from entlearn import ClassificationHead, Input, Recipe
from entlearn.scikit_adapter import EONClassifier

# Simple synthetic dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=5,
)

# Create a recipe
recipe = Recipe.chain(Input(K=2, epsilon=0.1), ClassificationHead())

# Define the parameter grid and the GridSearchCV object
search = GridSearchCV(
    EONClassifier(recipe, random_state=7, max_iter=100),
    {"recipe__blocks__input__epsilon": [0.1, 0.2]},
    cv=StratifiedKFold(2, shuffle=True, random_state=11),
    scoring="neg_log_loss",
)

# Fit the gridsearch
search.fit(X_train, y_train)
```

The path `recipe__blocks__input__epsilon` selects the affiliation temperature of the block named `input`.
To extend the parameter grid, you can add more entries to the dictionary. 

Scikit-learn scorers follow the convention that **larger values are better**.
`neg_log_loss` returns the negative of the log loss, so that a score closer to zero is better.
For regression, you could use `neg_mean_squared_error` or `r2` with an `EONRegressor` and a regression recipe.

```python
print(search.best_params_)
print(search.best_score_)
selected_model = search.best_estimator_
```

`best_params_` contains the selected settings, `best_score_` their mean cross-validation score, and `best_estimator_` the model refitted on all of `X_train`.
The selection score is *not a final test score*. Once you have finished choosing the model, evaluate it on the reserved test set using the same metric:

```python
from sklearn.metrics import log_loss

test_loss = log_loss(
    y_test,
    selected_model.predict_proba(X_test),
    labels=selected_model.classes_,
)
```

Here, `test_loss` is a positive loss, so smaller values are better.

??? warning "Fit preprocessing inside each training fold"
    Scaling, imputation and other learned transformations must be fitted using only the training portion of each fold.
    For ordinary grid search, put preprocessing and the estimator in a scikit-learn `Pipeline`, and pass that pipeline to `GridSearchCV`.
    Fitting preprocessing before cross-validation lets validation data influence the fitted models.

    For example, you can scale the features before fitting the classifier:

    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    pipeline = Pipeline(
        [
            ("scale", MinMaxScaler()),
            ("model", EONClassifier(recipe, random_state=7, max_iter=100)),
        ]
    )
    pipeline_search = GridSearchCV(
        pipeline,
        {"model__recipe__blocks__input__epsilon": [0.1, 0.2]},
        cv=StratifiedKFold(2, shuffle=True, random_state=11),
        scoring="neg_log_loss",
        error_score="raise",
    ).fit(X_train, y_train)
    ```

    The prefix `model__` selects the estimator step in the pipeline.
    Each fold fits its own scaler using only its training rows. The selected pipeline is then refitted on all of `X_train` and applies the fitted scaling automatically when predicting.

See the [GridSearchCV reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for more search options.

## Choosing search ranges

Which settings should you search, and over which values?

| Setting | Search it |
| --- | --- |
| `K` of each block | To vary model capacity and computation cost |
| `epsilon` of each block | To vary affiliation concentration |
| Connection `delta` | To vary the balance between connected blocks |
| `epsilon_D` | To learn feature weights in a standard `Input` |
| `epsilon_T` | To learn instance weights |
| The coupling | To compare M and S transition constraints |
| `delta_cat` | When the data mix continuous and categorical features |
| Hidden blocks and their settings | When you add depth |
| `subspace_dimension` and `alpha` | When you use a `ManifoldInput` |
| `epsilon_M` | When learned output weights suit your question |
| Prediction policy and `epsilon_P` | They can change the predictions without changing the fitted parameters |

Vary interacting parameters jointly within your search budget. Tuning them separately can miss useful combinations.

Space the temperatures and `delta` logarithmically, and the cluster counts linearly.
As a first coarse search, with continuous features scaled to [0, 1], try:

| Setting | Range |
| --- | --- |
| `epsilon` | $10^{-4}$ to $10^{-1}$ |
| `epsilon_D` | $10^{-3}$ to $1$ |
| `epsilon_T` | $10^{-2}$ to $10$ |
| `delta`, classification | $10^{-3}$ to $10$ |
| `delta`, regression | $10^{-2}$ to $10^{2}$ |

The affiliation temperature `epsilon` is measured on the scale of the [discretisation error](../concepts/representation.md#the-discretisation-error).

Whichever setting you search, if its best value sits **on a boundary** of its range, consider extending the range in that direction and search again.

To favour fewer clusters, include a complexity penalty in a custom scorer or search objective, for example accuracy - `0.001 * K`. This changes the ranking even when prediction scores differ, so choose the penalty to reflect the tradeoff you want.

The following grid crosses two cluster counts, two temperatures and both couplings of the classification head:

```python
coarse_search = GridSearchCV(
    EONClassifier(recipe, random_state=7, max_iter=100),
    {
        "recipe__blocks__input__K": [2, 4],
        "recipe__blocks__input__epsilon": np.logspace(-3, -1, 2).tolist(),
        "recipe__blocks__output__coupling": ["M", "S"],
    },
    cv=StratifiedKFold(2, shuffle=True, random_state=11),
    scoring="neg_log_loss",
).fit(X_train, y_train)
```

`Recipe.chain` names the head `output`, and a hidden connection's coupling is reached through `recipe__connections__<name>__coupling`.
Include the current or default settings in the grid, as a control.

## Select an initialisation for one recipe

An estimator can also compare several initialisations without changing the recipe.
Set `n_inits` to the number of candidates and `scoring` to the metric used to select one:

```python
initialisation_search = EONClassifier(
    recipe,
    n_inits=3,
    scoring="neg_log_loss",
    random_state=7,
    max_iter=100,
).fit(X_train, y_train)
```

With the default `cv=None`, candidates are compared on their training data (in-sample). 
With several initialisations and no explicit scorer, the classifier selects by accuracy and the regressor by $R^2$.

To select using validation scores, pass an integer, a splitter or explicit folds as the estimator's `cv`.

The default `init_rows=None` uses all rows supplied to that estimator fit to generate geometry, _including its internal validation rows_.
To exclude those rows, use `init_rows=common_train_rows` only with folds that have a non-empty common training set, as explained below. 

### Break ties with training scores

Exact ties in mean validation score go to the first candidate by default. With `return_train_score=True`, ties go to the smallest training–validation score gap.

```python
tie_break_model = EONClassifier(
    recipe,
    n_inits=3,
    cv=StratifiedKFold(2, shuffle=True, random_state=11),
    return_train_score=True,
    random_state=7,
    max_iter=100,
).fit(X_train, y_train)
```

The fitted network keeps the scores of every candidate in `network_.diagnostics.initialisation_outcomes`, and those of the selected candidate in [`network_.diagnostics.selected_outcome`](../reference/network.md#entlearn.FitDiagnostics.selected_outcome).
Its `train_score` is `None` unless training scores were recorded, so compute the difference only when you set `return_train_score=True` and a `cv`:

```python
outcome = tie_break_model.network_.diagnostics.selected_outcome
print(outcome.score, outcome.train_score)
gap = outcome.score - outcome.train_score
```

Diagnostic scores are stored as losses: `score` is the negated mean validation score and `train_score` the negated mean training score. In this accuracy example, `gap` is mean training accuracy minus mean validation accuracy. Exact validation-score ties go to the smallest gap.

Inside an outer search, this selection happens within each outer training fold, and the outer search only sees the selected model.
To report the gap of each selected candidate, add a scorer that reads it from the fitted estimator:

```python
def inner_gap(estimator, X, y):
    outcome = estimator.network_.diagnostics.selected_outcome
    return outcome.score - outcome.train_score


gap_search = GridSearchCV(
    EONClassifier(recipe, n_inits=3, cv=2, return_train_score=True, random_state=7, max_iter=100),
    {"recipe__blocks__input__epsilon": [0.1, 0.2]},
    cv=StratifiedKFold(2, shuffle=True, random_state=11),
    scoring={"accuracy": "accuracy", "inner_gap": inner_gap},
    refit="accuracy",
).fit(X_train, y_train)
print(gap_search.cv_results_["mean_test_inner_gap"])
```

## Supply initial states to a search

In an ordinary search, every fit generates its initial geometry from its own training rows, so each fold, and the final refit, starts from different geometry.
To start every fit from the same geometry, generate the states yourself with [`Network.initialise`](networks.md#initialisation) and pass them to the estimator as `initial_state`.

To keep one state fixed across the whole search, set it on the estimator, for example `EONClassifier(recipe, initial_state=state)`.
To choose the starting geometry as well, add a list of states to the grid as its `initial_state` axis.
`GridSearchCV` crosses this axis with the others, so it selects a state and the parameter settings together:

```python
import torch
from entlearn import Network

widest = recipe.replace_block("input", K=4)
states = [
    Network.initialise(widest, torch.as_tensor(X_train), torch.as_tensor(y_train), seed=seed)
    for seed in (1, 2, 3)
]
grid = {
    "recipe__blocks__input__K": [2, 4],
    "recipe__blocks__input__epsilon": [0.1, 0.2],
}
state_search = GridSearchCV(
    EONClassifier(recipe, max_iter=100),
    {**grid, "initial_state": states},
    cv=StratifiedKFold(2, shuffle=True, random_state=11),
    scoring="neg_log_loss",
    error_score="raise",
).fit(X_train, y_train)
selected_state = state_search.best_params_["initial_state"]
```

The states are generated for the largest `K` in the grid. A grid point with a smaller `K` starts from the first `K` centroids of its state, which are the centroids the same seed generates for that smaller `K`.
The selected model, `state_search.best_estimator_`, is refitted on all of `X_train` from `selected_state`.
An axis of $S$ states multiplies the number of parameter combinations, and so of fold fits, by $S$.

!!! warning "Validation rows can influence the states"
    These states are generated from all of `X_train`, including the rows each fold uses for validation.
    To exclude those rows, pass `Network.initialise` the rows that belong to *every* training fold as `init_rows`, and give the search the same folds.
    `common_train_rows` finds these rows. They exist only for some validation designs: with `KFold` or `StratifiedKFold`, every row is used for validation once, so there are none.

    An expanding-window split has a common training set. The next example illustrates this using ordered toy rows. Use `TimeSeriesSplit` for data whose order calls for forward validation.

    ```python
    from sklearn.model_selection import TimeSeriesSplit
    from entlearn.scikit_adapter import common_train_rows

    order = np.argsort(X_train[:, 0])
    X_ordered, y_ordered = X_train[order], y_train[order]
    folds = list(TimeSeriesSplit(n_splits=2).split(X_ordered))
    core_rows = torch.as_tensor(common_train_rows(folds))
    core_states = [
        Network.initialise(
            widest,
            torch.as_tensor(X_ordered),
            torch.as_tensor(y_ordered),
            seed=seed,
            init_rows=core_rows,
        )
        for seed in (1, 2, 3)
    ]
    core_search = GridSearchCV(
        EONClassifier(recipe, max_iter=100),
        {**grid, "initial_state": core_states},
        cv=folds,
        scoring="neg_log_loss",
        error_score="raise",
    ).fit(X_ordered, y_ordered)
    ```

    A state's centroids are in the units of the rows it was generated from.
    If a pipeline fits a scaler inside each fold, as [above](#gridsearchcv), each fold's rows have different units from the state, and nothing detects this.
    With supplied states, scale the data before generating the states and searching, and accept that the validation rows then influence the scaling.

### Rules for a valid grid

Each grid point is checked when fitted. Violations marked “Raises” fail the fit. Set `error_score="raise"` to stop the search on these errors. By default, scikit-learn records failed-fit scores as `nan` when other fits succeed.
Other violations replace incompatible input geometry with new geometry generated from `random_state`, as listed below.

| Rule | When a point breaks it |
| --- | --- |
| The estimator has `n_inits=1` | Raises |
| The estimator has `init_rows=None`; pass `init_rows` to `Network.initialise` instead | Raises |
| The state holds input geometry, as every state from `Network.initialise` does | New geometry; warns only if the fit uses nothing else from the state |
| The input block keeps the name it had when the state was generated | Warns, then new geometry |
| The input block keeps its kind, `Input` or `ManifoldInput` | Warns, then new geometry |
| The input block keeps its initialisation settings, `centroid_strategy`, `greedy_candidates`, `balanced` and `W_std`, and, for a `ManifoldInput`, its `subspace_dimension` | Warns, then new geometry |
| `K` is at most the `K` of the state, with the conditions below | Warns, then new geometry |
| The features keep the layout of the rows that generated the state, so `categorical_features` stays fixed | Warns, then new geometry, if the widths differ |

The warning names the block and the broken rule.
To stop the search on it as well, turn it into an error with `warnings.filterwarnings("error", message="InitialState")` before fitting the search.

A point may change the fitting temperatures: `epsilon`, `epsilon_D`, `epsilon_T` and `delta_cat` of an `Input`, and `epsilon`, `epsilon_T` and `alpha` of a `ManifoldInput`.
It may also change the hidden blocks, the connections and the head, since a state from `Network.initialise` holds only the input geometry and the seeds of the connections.
If a grid point removes or renames a seeded connection, the fit names it in a warning and ignores its seed. The input still starts from the state’s geometry.
The estimator converts the state to its own `dtype` and `device`.
With a supplied state, the estimator does not select an initialisation, so leave its `cv` and `scoring` at their defaults.

The `K` of a point has further conditions:

- The state must be generated by standard k-means++, or by greedy k-means++ with a fixed `greedy_candidates`, without balanced selection. Balanced selection and the default number of greedy candidates depend on `K`, so they need the state's own `K`.
- A state captured from a fitted network, with `Network.capture_current_state`, holds learned geometry. A point must then use exactly the number of centroids it holds, `input_geometry.K_active`, but may change its initialisation settings, `centroid_strategy`, `greedy_candidates`, `balanced` and `W_std`. Its name, its kind and, for a `ManifoldInput`, its `subspace_dimension` must still match.
- `Network.initialise` caps `K` at the number of eligible rows, and warns. The eligible rows are those in `init_rows`, or all the rows without it; balanced selection counts only the labelled ones. A point whose `K` exceeds the capped count warns and fits only the centroids the state holds, so generate the states from enough rows for the largest `K`.

With categorical columns, pass `Network.initialise` the codes the estimator uses, which number each column's levels in sorted order.
Every training fold must also contain every level of every categorical column: the estimator learns the levels from the rows it fits, so a fold that lacks one numbers the others differently from the state.

## Select an initialisation with a Network

The estimators' `scoring` and `cv` parameters are built on selection controls of `Network.fit`, which you can also use directly with tensors.
By default, `Network.fit(..., n_inits=3)` keeps the candidate with the lowest final training objective (see [Networks](networks.md#initialisation)).

### Select by a prediction loss

To select by a metric computed on the predictions instead, pass a function as `selection_loss`.
It should return one number, where **smaller values are better**.
For example, a sample-weighted Brier loss:

```python
import torch
from entlearn import Network

X_fit = torch.as_tensor(X_train)
y_fit = torch.as_tensor(y_train)


def brier_loss(prediction, target, *, sample_weights, **context):
    row_loss = ((prediction - target) ** 2).sum(dim=1)
    if sample_weights is None:
        return row_loss.mean()
    return (row_loss * sample_weights).sum() / sample_weights.sum()


brier_network = Network.fit(
    recipe, X_fit, y_fit, n_inits=3, seed=7, max_iter=100, selection_loss=brier_loss
)
```

The function receives the predictions and targets of the labelled rows: class probabilities and class distributions for classification, or two-dimensional values for regression.
It also receives the keyword arguments `sample_weights`, `class_weights` and `task_weights`, each `None` when you did not supply those weights, and `fold` and `partition`, described below.
The callback receives the supplied weights and decides how to combine and normalise them for its metric.

Here the candidates are still compared on the rows they were fitted on, with `fold=None` and `partition="training"`, so a lower loss is not evidence of better performance on new data.

### Select with validation folds

To compare candidates on held-out rows, also pass `validation_pairs`: a list of `(training_rows, validation_rows)` pairs of `torch.int64` index tensors.

```python
rows = torch.arange(len(X_fit))
pairs = [(rows[:50], rows[50:]), (rows[25:], rows[:25])]
validated_network = Network.fit(
    recipe,
    X_fit,
    y_fit,
    n_inits=3,
    seed=7,
    max_iter=100,
    selection_loss=brier_loss,
    validation_pairs=pairs,
)
```

Each candidate starts from one initial state, and is fitted separately on the training rows of every pair from that same state.
Each fold model is scored on single-pass predictions for its validation rows, with `partition="validation"` and `fold` set to the pair, and the candidate with the lowest mean validation loss wins.
Every fold has equal influence, regardless of its size. The winning candidate is then fitted once more on all the rows, starting from its initial state.

Both sides of a pair must be non-empty, without repeated indices, and must not share rows. Each side needs labelled rows with positive weight.
The folds may overlap each other, and they do not need to cover every row.
`validation_pairs` requires `selection_loss`, and is ignored when you supply an `initial_state`.

With `return_train_score=True`, each fold model is also scored on its training rows, with `partition="training"`, and the mean is recorded as the `train_score` of each outcome in `network.diagnostics.initialisation_outcomes`.
Exact ties in the mean validation loss then go to the smallest `score - train_score`, the validation loss minus the training loss, as described in [Break ties with training scores](#break-ties-with-training-scores).

!!! warning "Validation rows can still influence the initialisation"
    As with [supplied states](#supply-initial-states-to-a-search), restrict initialisation to rows shared by every training partition. Pass their indices as `init_rows`. The caller is responsible for excluding validation rows.
    Use `common_train_rows` to find the shared rows:
    It takes the pairs, as CPU tensors or NumPy arrays, and returns a NumPy array, which `Network.fit` needs as a tensor on the device of the data:

    ```python
    from entlearn.scikit_adapter import common_train_rows

    init_rows = torch.as_tensor(common_train_rows(pairs), device=X_fit.device)
    ```

    Pass the result as `init_rows` together with `validation_pairs`. If the training partitions share no rows, the result is empty and `Network.fit` rejects it.
    An estimator also accepts `common_train_rows` itself as its `init_rows`, as described [above](#select-an-initialisation-for-one-recipe).

    The validation labels also select the winner, so the validation score is not a final test score.

For $I$ candidates and $F$ folds, selection performs $IF + 1$ fits, or $IF + I$ with `retain="members"`, since each retained member is fitted on all the rows.

### Parallel initialisations

By default, candidates are fitted one after the other.
Set `n_jobs` to fit several at the same time, for example `n_jobs=-1` to use all available CPUs. This requires `joblib`, installed by the `sklearn` extra.
`parallel_backend` chooses between `"threads"` (the default) and `"processes"`:

```python
parallel_network = Network.fit(
    recipe, X_fit, y_fit, n_inits=3, seed=7, max_iter=100, n_jobs=2, parallel_backend="threads"
)
```

Threads share the data. Processes copy it to each worker, and require the `selection_loss` and any other function you pass to be picklable, so define them at module level rather than inside another function.
A process request made from inside a worker or a non-main thread emits a warning and uses threads instead.
With either backend, warnings raised inside a candidate reach your code once every candidate has finished, as described in [Troubleshooting](troubleshooting.md#warnings).
The estimators accept the same `n_jobs` and `parallel_backend` parameters.

The seeds of all the candidates are drawn before any is fitted, so the selected candidate does not depend on `n_jobs`, the backend or the order in which workers finish.
Each candidate's record in [`network.diagnostics.initialisation_outcomes`](../reference/network.md#entlearn.FitDiagnostics) holds the requested and the actual backend, as `requested_backend` and `effective_backend`.

## Optional Optuna backend

Optuna's `OptunaSearchCV` samples parameter combinations within a fixed trial budget.
Install its optional dependencies separately:

```bash
pip install optuna "optuna-integration[sklearn]"
```

Parameters can be sampled from a continuous range using `FloatDistribution`, or from a list of choices using `CategoricalDistribution`.
For example, we can vary the temperature between 0.05 and 0.5, and choose the number of clusters from `[2, 4, 8]`:

```python
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna_integration import OptunaSearchCV

optuna_search = OptunaSearchCV(
    EONClassifier(recipe, random_state=7, max_iter=100),
    param_distributions={
        "recipe__blocks__input__epsilon": FloatDistribution(0.05, 0.5, log=True),
        "recipe__blocks__input__K": CategoricalDistribution([2, 4, 8]),
    },
    n_trials=10,
    random_state=13,
    cv=StratifiedKFold(2, shuffle=True, random_state=11),
    scoring="neg_log_loss",
    error_score="raise",
).fit(X_train, y_train)

selected_model = optuna_search.best_estimator_
```

Here, `log=True` samples the temperature on a logarithmic scale.
You can inspect `best_params_` and `best_score_`, and use the selected model as in the grid search above.
See the [OptunaSearchCV reference](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.OptunaSearchCV.html) for its search options.

[Tutorial 4](../tutorials/series.md#tutorial-4) provides a visual walkthrough of parameter search.
