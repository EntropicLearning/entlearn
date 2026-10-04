# Hyperparameters

Which settings can you change, and what do they control?

## Model settings

The main settings for a model are:

| Setting | Controls |
| --- | --- |
| `K` | Number of clusters in an input or hidden block |
| `epsilon` | How spread the affiliations are across clusters |
| `epsilon_D` | Learning of feature weights in a standard `Input` |
| `epsilon_T` | Learning of instance weights in an input block |
| Connection `delta` | Strength of the coupling between two blocks |
| `RegressionHead.epsilon_M` | Learning of weights across output dimensions |
| `ManifoldInput.subspace_dimension` and `alpha` | Dimension and distance metric of each local plane; see [Manifold](manifold.md) |

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

`Coupling.M` and `Coupling.S` also change the model. For a transition matrix shaped `(K_target, K_source)`, M normalises each column and S each row.
They therefore give the transition matrix different meanings, see the [m and s coupling section](../concepts/representation.md#transitions-between-clusterings).

## Fitting and prediction controls

`max_iter` limits training iterations, `tol` controls stopping, and `n_inits` sets the number of starting points to compare.
Use `seed` with `Network.fit`, or `random_state` with an estimator, to control initialisation.

Device and dtype select numerical execution: estimators stage data using `device` and `dtype`, while Networks use the provided tensor placement (with the option to override the precision with the `computation_dtype` argument).

Prediction has separate iteration and tolerance settings:

```python
from entlearn import PredictConfig

policy = PredictConfig(predict_mode="iterative", max_iter=50, tol=1e-5)
```

The default `predict_mode="single"` uses a single forward pass.
A supplied policy replaces the complete fitted policy, so you can use `dataclasses.replace` to change selected settings while preserving the rest. See [prediction controls](networks.md#predict-new-observations).

## Classification calibration

The output temperature `epsilon_P` controls how concentrated the class probabilities are.
With the default geometric read-out, it is chosen automatically at the end of fitting by minimising weighted single-pass cross-entropy on labelled training rows with positive sample weight.
Each fold and final refit calibrates on its own training rows. 

Automatic calibration can be prevented by supllying a fixed temperature:

```python
fixed_policy = PredictConfig(epsilon_P=0.5)
fixed_model = EONClassifier(recipe, predict_config=fixed_policy, random_state=7)
```

The fitted temperature is stored in `network.predict_config.epsilon_P` and not recalibrated at inference time.
A new prediction policy with `epsilon_P=None` uses the uncalibrated fallback, so use `dataclasses.replace` when preserving the fitted temperature.
Arithmetic read-out (`output_mode="arithmetic"`) is available only for M classification heads and accepts no output temperature. Regression accepts neither classification read-outs nor output temperatures.

## Regression output weights

A finite `epsilon_M` learns relative weights across target dimensions. Infinity keeps a supplied `W_M` tuple fixed, or uses uniform weights.
A fixed `W_M` requires infinite `epsilon_M`. Inspect the weights with `target_weights(network)` from `entlearn.helpers.reporting`.

## Other settings

`W_std` controls the initial spread of feature weights, where zero starts them uniformly. 
`delta_cat` scales the contribution of categorical features relative to continuous ones.
For centroid seeding, you can choose between `"kmeans++"` or `"greedy-kmeans++"`. The latter accepts a fixed `greedy_candidates` count.
`balanced=True` can be used in classification tasks to assign equal amounts of clusters to each class. 
On hidden-target connections, `theta_alpha` controls a Dirichlet prior (disabled with `theta_alpha=1`). 

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

- **Scaler.** A `MinMaxScaler` keeps the shape of each feature's distribution, but a few extreme values squeeze the others into a narrow part of [0, 1]. A `QuantileTransformer` spreads each feature evenly over [0, 1] and limits the influence of outliers, but it changes the distances between observations. Which one suits your data depends on the problem. A search over a scikit-learn `Pipeline` can compare both options.
- **Hidden blocks.** Try including hidden blocks, starting from one. Compare the results to those obtained with a shallow network.
- **Manifold input.** Use a [`ManifoldInput`](manifold.md#when-to-use-a-manifold-input) when local subspaces are part of your hypothesis about the data.
- **Coupling.** The optimal coupling depends on the problem, so treat the coupling as a hyperparameter and [compare them](model_selection.md#choosing-search-ranges).
- **Connection strength.** Consider adjusting the connection strength `delta` via search, and try having different `delta` values for different connections.
- **Affiliation temperature.** Choose the value of `epsilon` through a search, and use `epsilon=0` only when you want hard assignments.
- **Output weights.** Learn them with a finite `epsilon_M` only when the relative emphasis of the scaled target dimensions is meaningful for your question, and check the skill of each output dimension separately.
- **Iterative prediction.** Consider using it and check if it improves held-out results at an acceptable cost.
- **Categorical costs.** If your data include categorical features, consider tuning `delta_cat`, which scales their costs relative to the continuous features; see [Choosing search ranges](model_selection.md#choosing-search-ranges).
- **Class weights, missing targets and reconstruction.** Use them as your question requires; [Using estimators](estimators.md) describes each one.
- **Selecting the initialisation on validation folds.** Once the first experiment runs, pass a cross-validation splitter, such as `StratifiedKFold`, as the estimator's `cv`. It then compares the initialisations on held-out rows instead of the training rows, see [Model selection](model_selection.md#select-an-initialisation-for-one-recipe).
