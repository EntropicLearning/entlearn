# Hyperparameters

Which settings can you change, and what do they control?

## Model settings

The main settings for a model are:

| Setting | Controls |
| --- | --- |
| `K` | Number of clusters in an input or hidden block |
| `epsilon` | How spread the affiliations are across clusters |
| `Hidden.epsilon_ratio` | A hidden block's `epsilon`, given as a ratio to the `delta` of its incoming connection. Optional alternative to `epsilon` |
| `epsilon_D` | Learning of feature weights in a standard `Input` |
| `epsilon_T` | Learning of instance weights in an input block |
| Connection `delta` | Strength of the coupling between two blocks |
| Connection `delta_ratio` | A connection's `delta`, given as a ratio to the `delta` of the previous connection |
| `RegressionHead.epsilon_M` | Learning of weights across output dimensions |
| Connection and `ClassificationHead` `theta_alpha` | A Dirichlet prior on a hidden transition or the head's class transition; see [Transition and head priors](#transition-and-head-priors) |
| `ManifoldInput.subspace_dimension` and `alpha` | Dimension and distance metric of each local plane, see [Manifold](manifold.md) |

!!! question "What does a temperature do?"
    For affiliations, `epsilon=0` gives hard assignments, while positive finite values allow soft assignments. Larger temperatures favour more spread-out affiliations. Infinite affiliation temperatures are not accepted.
    For feature, instance and output weights, a finite positive temperature enables learning, zero selects minimum-cost entries through a hard update.

!!! note "Infinite weight temperatures freeze the weights"
    The default `epsilon_D`, `epsilon_T` and `epsilon_M` are infinity. Feature weights stay at their initial values, instance weights stay at the supplied sample weights (or uniform weights when none are supplied). Output weights remain fixed as described below.

For example, to enable feature weighting and later change the affiliation temperature:

```python
from entlearn import ClassificationHead, Input, Recipe
from entlearn.scikit_adapter import EONClassifier

recipe = Recipe.chain(
    Input(K=3, epsilon=0.1, epsilon_D=0.5),
    ClassificationHead(),
)
model = EONClassifier(recipe, random_state=7, max_iter=100)
model.set_params(recipe__blocks__input__epsilon=0.2)
```

For a hidden block, you can set the affiliation temperature with either `epsilon` or `epsilon_ratio`.
The ratio sets `epsilon = epsilon_ratio * delta`, using the `delta` of the block's incoming connection.
For example, a ratio of 1.5 with `delta=0.1` gives `epsilon=0.15`:

```python
from entlearn import Hidden

ratio_recipe = Recipe.chain(
    Input(K=6, epsilon=0.01),
    Hidden(K=2, epsilon_ratio=1.5),
    ClassificationHead(),
    delta=0.1,  # Sets delta=0.1 on both connections.
)
explicit_recipe = ratio_recipe.resolve_ratios()
print(explicit_recipe.blocks[1].epsilon)  # 1.5 * 0.1
```

Each connection can have its own `delta` or `delta_ratio`. Sequences have one entry per connection in chain order, with `None` where the other setting supplies the strength.
The first connection has no preceding connection, so its `delta_ratio` entry must be `None`:

```python
blocks = (Input(), Hidden(), Hidden(), ClassificationHead())

# Different absolute strengths.
different_deltas = Recipe.chain(*blocks, delta=[0.1, 0.2, 0.8])

# Different ratios, resolving to strengths of 0.1, 0.2 and 0.8.
different_ratios = Recipe.chain(
    *blocks,
    delta=[0.1, None, None],
    delta_ratio=[None, 2, 4],
)

# Absolute strengths and ratios can be mixed.
mixed_recipe = Recipe.chain(
    *blocks,
    delta=[0.1, None, 0.8],
    delta_ratio=[None, 2, None],
)
```

A scalar `delta` applies to all of the connections without a ratio.
`delta_ratio` is the connection's `delta` divided by the `delta` of the preceding connection.
`Recipe.chain` applies a scalar `delta_ratio` to every connection after the first: `delta_ratio=3` makes each `delta` three times larger than the one before it.
Since `epsilon_ratio` uses `delta`, `delta_ratio` is resolved first.

```python
chain_recipe = Recipe.chain(
    Input(K=6, epsilon=0.01),
    Hidden(K=4, epsilon_ratio=1.5),
    Hidden(K=2, epsilon_ratio=1.5),
    ClassificationHead(),
    delta=0.1,
    delta_ratio=3,
)
explicit_chain = chain_recipe.resolve_ratios()
print([c.delta for c in explicit_chain.connections])  # 0.1, 0.3 and 0.9, to rounding
print(explicit_chain.blocks[2].epsilon)  # 1.5 * 0.3
```

`resolve_ratios()` returns the recipe with the resulting `epsilon` and `delta` written out, and a fitted network's display shows each ratio beside its resulting value.
Provide either `epsilon` or `epsilon_ratio`. A ratio requires the corresponding `epsilon` to be at its default value (`None`).
Similarly, provide either `delta` or `delta_ratio`. A ratio requires the `Connection`'s `delta` field to be at its default value (`None`); without a ratio, `None` gives 1.0.
`Recipe.chain`'s own `delta` argument defaults to 1.0 and combines with `delta_ratio`, as above.
To change a setting from one form to the other, give both fields at once, for example `replace_connection(name, delta=0.5, delta_ratio=None)`.
Only a hidden block has `epsilon_ratio`, since an input block has no incoming connection. A connection out of an input block has no `delta_ratio`, since no connection comes before it.

`Coupling.M` and `Coupling.S` also change the model. For a transition matrix shaped `(K_target, K_source)`, M normalises each column and S each row.
They therefore give the transition matrix different meanings, see the [m and s coupling section](../concepts/representation.md#transitions-between-clusterings).

## Fitting and prediction controls

`max_iter` limits training iterations, `tol` controls stopping, and `n_inits` sets the number of starting points to compare.
A fit stops after an iteration that changes the loss by at most `tol` of its scale. [Convergence and stopping](../concepts/learning.md#convergence-and-stopping) describes how.
Use `seed` with `Network.fit`, or `random_state` with an estimator, to control initialisation.

Device and dtype select numerical execution: estimators stage data using `device` and `dtype`, while Networks use the provided tensor placement (with the option to override the precision with the `computation_dtype` argument).

Prediction has separate iteration and tolerance settings:

```python
from entlearn import PredictConfig

policy = PredictConfig(predict_mode="iterative", max_iter=50, tol=1e-5)
```

The default `predict_mode="single"` uses a single forward pass.
A supplied policy replaces the complete fitted policy, except that a geometric read-out that leaves `epsilon_P` out keeps the fitted [output temperature](#classification-calibration).
Use `dataclasses.replace` to change selected settings while preserving the rest. See [prediction controls](networks.md#predict-new-observations).

## Classification calibration

The output temperature `epsilon_P` controls how concentrated the class probabilities are.
With the default geometric read-out, it is chosen automatically at the end of fitting by minimising weighted single-pass cross-entropy on labelled training rows with positive sample weight.
Each fold and final refit calibrates on its own training rows.

Automatic calibration can be prevented by supplying a fixed temperature:

```python
fixed_policy = PredictConfig(epsilon_P=0.5)
fixed_model = EONClassifier(recipe, predict_config=fixed_policy, random_state=7)
```

The fitted temperature is stored in `network.predict_config.epsilon_P` and not recalibrated at inference time.
A new geometric prediction policy with `epsilon_P=None` keeps the fitted temperature; supply `epsilon_P` to change it.
After an arithmetic fit, which has no fitted temperature, such a policy uses the uncalibrated fallback.
Arithmetic read-out (`output_mode="arithmetic"`) is available only for M classification heads and accepts no output temperature. Regression accepts neither classification read-outs nor output temperatures.

## Regression output weights

A finite `epsilon_M` learns relative weights across target dimensions. Infinity keeps a supplied `W_M` tuple fixed, or uses uniform weights.
A fixed `W_M` requires infinite `epsilon_M`. Inspect the weights with `target_weights(network)` from `entlearn.helpers.reporting`.

## Other settings

`W_std` controls the initial spread of feature weights, where zero starts them uniformly.
`delta_cat` scales the contribution of categorical features relative to continuous ones.
By default, fitting sets it from the training rows with [`delta_cat_guess`](../reference/helpers.md#entlearn.helpers.delta_cat_guess): the value at which an uninformative categorical feature costs as much as an uninformative continuous one.
When the data have no varying continuous feature or no categorical feature with at least two levels, it is 1.
For centroid seeding, you can choose between `"kmeans++"` or `"greedy-kmeans++"`. The latter accepts a fixed `greedy_candidates` count.
In classification tasks, `balanced=True` balances the initial centroids across the classes.

## Transition and head priors

A hidden-target connection and a `ClassificationHead` each take a `theta_alpha`, a Dirichlet prior on their transition matrix.
The default `theta_alpha=1` disables it. A value above one adds the same pseudocount to every entry, so no entry can be zero; `theta_alpha - 1` is the prior's total mass as a fraction of the sample weight.

Without a prior, an entry that no observation supports costs about 708 in float64 and 87 in float32, a price set by the dtype rather than the data.
A prior helps when many entries are zero: with hard or very cold affiliations, when float32 and float64 fits should agree, and when small clusters would otherwise give extreme class probabilities.

Start slightly above one, such as `1.1`, and tune it with a search, as for `delta`.
A large prior pulls every transition towards uniform, which weakens the coupling and flattens the predicted probabilities.

Geometry reuse (`initial_state`, `init_rows`), retention (`retain`) and estimator continuation (`warm_start`) serve different purposes, see [Networks](networks.md#initialisation) and [Estimators](estimators.md#refit-and-reuse).
Supplied sample, class and target-dependent weights are explained in [Estimator weighting](estimators.md#sample-and-class-weights).

## A recommended starting point

With so many settings, where should you start?
The rules below come in three kinds: combinations that are not valid, a recommended first experiment, and choices that depend on the problem.

### Rules for a valid model

- The head must match the task: a `ClassificationHead` for `EONClassifier`, a `RegressionHead` for `EONRegressor`.
- The recipe must be a [chain](package.md#what-shapes-can-a-model-have): one input block, zero or more hidden blocks, one head.
- An S classification head requires the geometric read-out.
- Fixed output weights `W_M` require an infinite `epsilon_M`, and fixed feature weights an infinite `epsilon_D`. With a finite `epsilon_D`, if you supply weights and do not fix the temperature, they will be used as starting conditions.

### A first experiment

- Scale the continuous features, and any regression targets, to [0, 1] with a `MinMaxScaler` or a `QuantileTransformer`, fitted on the training rows only.
- Connect a standard `Input` directly to the head.
- Use soft affiliations, with a small positive `epsilon`. By default `epsilon=0` gives hard ones.
- Learn the feature and instance weights, with finite `epsilon_D` and `epsilon_T`.
- Keep the default M coupling.
- For classification, keep the default geometric read-out, and let fitting calibrate its temperature.
- Keep the default single-pass prediction.
- Use a modest number of clusters compared with the number of observations.
- Fit several initialisations, since different starting points can give noticeably different models, and record the seed. The estimator keeps the one that scores best on the training rows.

```python
first_recipe = Recipe.chain(
    Input(K=4, epsilon=0.01, epsilon_D=0.1, epsilon_T=0.1),
    ClassificationHead(),
)
first_model = EONClassifier(first_recipe, n_inits=5, random_state=7, max_iter=100)
```

### Choices that depend on the problem

- **Scaler.** A `MinMaxScaler` keeps the shape of each feature's distribution, but a few extreme values squeeze the others into a narrow portion of the interval [0, 1]. A `QuantileTransformer` spreads each feature evenly over [0, 1] and limits the influence of outliers, but it also changes the distances between observations. Which one suits your data depends on the problem. A search over a scikit-learn `Pipeline` can compare both options.
- **Hidden blocks.** Try including hidden blocks, starting from one. Compare the results to those obtained with a shallow network.
- **Manifold input.** Use a [`ManifoldInput`](manifold.md#when-to-use-a-manifold-input) when local subspaces are part of your hypothesis about the data.
- **Coupling.** The optimal coupling depends on the problem, so treat the coupling as a hyperparameter and [compare `Coupling.M` and `Coupling.S`](model_selection.md#choosing-search-ranges).
- **Connection strength.** Consider adjusting the connection strength `delta` via search, and try having different `delta` values for different connections.
- **Affiliation temperature.** Choose the value of `epsilon` through a search, and use `epsilon=0` only when you want hard assignments.
- **Priors.** With hard or very cold affiliations, small clusters, or probabilities you score with log-loss, try a `theta_alpha` slightly above one on the connections and the head; see [Transition and head priors](#transition-and-head-priors).
- **Output weights.** Learn them with a finite `epsilon_M` only when the relative emphasis of the scaled target dimensions is meaningful for your question, and check the skill of each output dimension separately.
- **Iterative prediction.** Consider using it and check if it improves held-out results at an acceptable cost.
- **Categorical costs.** If your data include categorical features, consider tuning `delta_cat`, which scales their costs relative to the continuous features, see [Choosing search ranges](model_selection.md#choosing-search-ranges).
- **Class weights, missing targets and reconstruction.** Use them as your question requires. [Using estimators](estimators.md) describes each one.
- **Selecting the initialisation on validation folds.** Once the first experiment runs, pass a cross-validation splitter, such as `StratifiedKFold`, as the estimator's `cv`. It then compares the initialisations on held-out rows instead of the training rows, see [Model selection](model_selection.md#select-an-initialisation-for-one-recipe).
