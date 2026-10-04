# Using Networks

## From a recipe to a fitted network

`Network.fit` takes a [Recipe](package.md) and tensor data and returns a fitted network.

!!! note "Tensors only"
    Networks operate only on `torch.Tensor` objects.

```python
import torch
from entlearn import ClassificationHead, Input, Network, Recipe

# Some example data
X = torch.tensor([[0.0], [0.1], [0.9], [1.0]], dtype=torch.float64)
y = torch.tensor([0, 0, 1, 1])

# Create a simple Recipe
recipe = Recipe.chain(Input(K=2), ClassificationHead())

# Fit a Network using the recipe and the data
network = Network.fit(recipe, X, y, seed=7, max_iter=20)
```

`seed` controls initialisation, and `max_iter` sets the maximum number of training iterations.

!!! info "Convergence"
    Fitting stops automatically when the procedure converges, so a large `max_iter` is only an upper limit.

Predict new observations with the fitted network:

```python
query = torch.tensor([[0.05], [0.95]], dtype=X.dtype)
answer = network.predict(query)
```

For this example, `answer` is a probability matrix with shape `(2,2)`, one row per observation and one column per class.

!!! note "Classification predictions are probabilities"
      Unlike a scikit-learn classifier's `predict`, `Network.predict` returns class probabilities. For regression, it returns predicted values.

Select the most likely class with `argmax`:

```python
class_idx = answer.argmax(dim=1)
```

## Expected input shapes

Continuous inputs `X` have shape `(n_samples, n_features)`, including when there is only one feature.

The shape of `y` depends on the task. For **classification**, either class codes `(n_samples,)` with dtype `torch.int64` or label distributions `(n_samples, n_classes)` with dtype `torch.float32` or `torch.float64` are accepted.
For **regression** the targets can be either `(n_samples,)`, `(n_samples,1)`, or `(n_samples, n_targets)` (for multi-target regression) with dtype `torch.float32` or `torch.float64`.

For regression, a row containing `NaN` targets is treated as unlabelled. If only some targets are missing, `Network` warns and masks the whole row. The estimator rejects partially missing rows.

**Categorical features**, if present, can be passed separately through the `X_cat` argument, which expects a list containing one tensor per categorical feature. See the `Network` reference for the supported representations.

Make sure that inputs and targets reside on the same `device` and have compatible `dtype`.
At prediction time, preserve the feature order and use the fitted model’s device and computation dtype.


!!! warning "Feature scaling" 
    Feature scaling can affect the distances used by the model. Scaling continuous features to comparable ranges, such as [0, 1], helps prevent their units from dominating these distances.
    Estimate the scaling from the training data and apply the same transformation to new observations. `Network` does **not** perform this preprocessing automatically.

## Inspect the fit

Once the model is fitted, you can inspect both the training procedure and the learned parameters.
For a quick summary, you can use `print(network)`:

```python
print(network)
```

The `schema` records the input and output dimensions, the computation dtype and the active cluster counts:

```python
print(network.schema)
```

!!! note "Active clusters"
    The number of active clusters can be smaller than the number requested in the recipe, because clusters that become empty are removed during fitting.
    This does not change the recipe itself.
    You can check the difference between the requested clusters and the final active ones in the following way:
    ```python
    requested = next(block.K for block in network.recipe.blocks if block.name == "input")
    active = dict(network.schema.K_active)["input"]
    print(f"Input clusters: {requested} requested, {active} active")
    ```

`diagnostics` records convergence, the iteration count and the loss history.
```python
print(network.diagnostics.converged)
print(network.diagnostics.n_iter)
print(network.diagnostics.loss_history)
```

Access learned parameters with `network.inspect`:

```python
centroids = network.inspect("continuous_centroids")["input"]
feature_weights = network.inspect("feature_weights")["input"]
training_affiliations = network.inspect("training_affiliations")["input"]
```

The result is a dictionary keyed by block names. Here, `"input"` is the default name assigned by `Recipe.chain`.
If you named the input block `"features"`, you should use that name instead.
For continuous inputs, the centroids have shape `(n_active_clusters, n_features)`, and the training affiliations have shape `(n_samples, n_active_clusters)`.
`network.inspect("affiliation_regimes")` reports whether each input and hidden block assigns [hard or soft](../concepts/learning.md#affiliation-temperatures) affiliations, for example `{"input": "soft", "hidden_1": "hard"}`.

Training affiliations describe the observations during fitting.
To obtain affiliations for new observations, request them alongside the predictions:

```python
details = network.predict_with_details(query, details=("affiliations",))
query_affiliations = details.affiliations["input"]
```

!!! info "Query and training affiliations"
    Even for the original observations, query affiliations can differ from training affiliations.
    Inspecting training affiliations and instance weights requires a fit or a [resumable save](#save-the-network).

You can also plot the fitted model, provided that the `plotting` extra is installed:

```python
input_figure = network.plot(block="input", X_cont=X)
```

Omit `block` to show all blocks and the loss history. The figure can be displayed directly in a notebook.
Observations are coloured by their [query affiliations](../concepts/prediction.md#the-single-forward-pass).

To follow fitting in the terminal, pass `verbose=1` for a final summary, or `verbose=2` for iteration logs.
The default `verbose=0` disables these logs, but warnings are still emitted. You can inspect retained warnings through `network.diagnostics.warnings`.
See [Troubleshooting](troubleshooting.md) to send the logs elsewhere and for the meaning of each warning.

See [reconstruction and inlier percentiles](#reconstruction-and-inlier-percentiles) for other queries, and [Plotting](plotting.md) for plots.

## Initialisation

Fitting starts from an initial set of values, and different starting points can lead to different fitted models.
`Network.fit` normally takes care of initialisation, using the provided `seed`.

`Network.initialise` returns an `InitialState` containing starting geometry that you can supply to a fit:

```python
initial_state = Network.initialise(recipe, X, y, seed=7)
network_from_state = Network.fit(
    recipe,
    X,
    y,
    initial_state=initial_state,
    seed=7,
    max_iter=20,
)
```

??? question "Compare fits from the same geometry"
    By reusing an `InitialState`, you can compare compatible configurations from the *same* starting geometry. Keeping the data, fitting settings and execution conditions unchanged allows you to repeat a fit, or explore what changes when you modify a single parameter.
    [Model selection](model_selection.md#supply-initial-states-to-a-search) shows how to do this across a parameter search.

    You can also supply your own `continuous_centroids` or `feature_weights` to `Network.initialise`.

The original `InitialState` is available after fitting through `network.initial_state`.
It records the starting point before training and pruning. Use `network.inspect` for the learned values.

!!! note "Cluster counts and eligible observations"
    When generating input geometry, the initialiser caps `K` at the number of eligible observations with positive sample weight and emits a warning. With balanced classification initialisation, only labelled rows are eligible. Supplying `init_rows` further restricts the pool.
    This happens before fitting and is separate from pruning empty clusters during training. The recipe keeps the requested `K`. `network.initial_state.input_geometry.K_active` records the initial width, and `network.schema.K_active` records the fitted width.
    Supplied geometry is not truncated. A later fit from a capped state with the full `K` warns, and fits only the clusters the state holds.

Set `n_inits` to compare several starting states:

```python
best_network = Network.fit(
    recipe,
    X,
    y,
    n_inits=3,
    seed=7,
    max_iter=20,
    retain="members",
)
```

This fits three candidates and returns the one with the lowest final training objective by default.
Use either `n_inits` to generate several candidates or `initial_state` to supply one starting geometry (note: this requires `n_inits=1`).
With `retain="members"`, the fitted candidates are available in `best_network.members`, in initialisation order.
You can obtain their individual predictions together:

```python
candidate_predictions = best_network.predict_all(query)
```

The result has shape `(n_inits, n_queries, output_width)`. Ordinary `predict` uses the selected candidate.
If you only need the original starting states, use `retain="states"` instead.

For comparisons using validation data, see [Model selection](model_selection.md).

### Reproducible fits

The random initialisation uses a generator on the device of the data. As a result, the same `seed` gives different initial states on CPU and on CUDA.
This includes the seeds of the candidates when `n_inits` is above one.
To repeat a fit, use the same seed, the same number of initialisations and the same kind of device.

On CUDA, enable `torch.use_deterministic_algorithms(True)` before fitting to request deterministic operations. Reproducibility also depends on keeping the data, settings, device and software environment unchanged.

A [saved network](#save-the-network) stores its fitted parameters, selected candidate and any retained members. Loading restores them without rerunning initialisation.

## Predict, continue, or start again

### Predict new observations

`network.predict(query)` uses the fitted parameters to predict new observations.
For **classification** it returns a matrix of class probabilities, while for **regression** it returns a matrix of predicted values, with shape `(n_queries, n_targets)`.

The prediction procedure is controlled by a `PredictConfig`, stored in `network.predict_config`.
By default, predictions use a **single forward pass**, but **iterative prediction** can also be employed.

??? question "Single-pass and iterative prediction"
    A **single forward pass** assigns query affiliations once, block by block. **Iterative prediction** repeatedly updates affiliations and prediction coordinates with the learned parameters fixed, and takes more computation.

    To use iterative prediction, replace the fitted prediction configuration:
    
    ```python
    from dataclasses import replace
    
    policy = replace(network.predict_config, predict_mode="iterative", max_iter=50)
    refined_answer = network.predict(query, predict_config=policy)
    ```
    
    Here, `max_iter` limits prediction iterations. Evaluate the resulting predictions to decide whether refinement helps on your data.

!!! note "A prediction configuration replaces the complete policy"
    Passing a `PredictConfig` does not merge its settings with the fitted policy. Use `replace`, as above, if you only want to change selected settings. Omitting `predict_config` uses the fitted policy.

Iterative prediction reuses the fitted output temperature. Its stopping criterion uses the mean query objective, so splitting queries into different batches can change the stopping iteration.

### Inspect a prediction

Use `predict_with_details` to request additional results in the same prediction call:

| Detail | Returns |
| --- | --- |
| `"affiliations"` | Query affiliations keyed by input and hidden block names |
| `"instance_weights"` | Recovered query weights, when the fitted model supports recovery |
| `"reconstruction"` | Reconstructed input features |
| `"diagnostics"` | Refinement iteration count, convergence status and loss history |

For example, we can inspect refinement while starting from equal class probabilities:

```python
prediction_details = network.predict_with_details(
    query,
    predict_config=policy,
    predict_init="uniform",
    details=("affiliations", "diagnostics"),
)
print(prediction_details.n_iter, prediction_details.converged)
print(prediction_details.loss_history)
```

`prediction_details.prediction` contains the output. Unrequested fields are `None`.
`predict_init` is optional and only applies to iterative prediction; without it, refinement starts from the single-pass prediction. See the [Network reference](../reference/network.md#entlearn.Network.predict) for other starts, including regression options.

The loss history contains one entry per refinement iteration, excluding the initial forward pass. Single-pass diagnostics are `0`, `False` and an empty history, since no refinement was attempted.
For recovered weights and their requirements, see [Reconstruction and inlier percentiles](#reconstruction-and-inlier-percentiles).

For retained candidates, `predict_all_with_details` returns one result per member in initialisation order, allowing for different pruned cluster counts.
[Tutorial 5](../tutorials/series.md#tutorial-5) illustrates how predictions pass through the fitted blocks.

### Reconstruction and inlier percentiles

`network.reconstruct(query)` approximates the input features. It returns a `ReconstructionResult` containing a continuous matrix and, with `X_cat`, one categorical probability matrix per categorical feature.
Categorical reconstructions blend centroid distributions. They preserve ambiguity, rather than decoding a single category, and are not missing-feature imputations.

For a model with learned instance weights and a finite training normaliser, `network.score_samples(query)` measures each query's inlier percentile: how well it fits the learned input representation.
It returns percentile ranks in `[0, 1]`: a low value means the recovered query weight is low compared with weights recovered on the training rows using the fitted prediction policy.
These ranks are neither probabilities of correct prediction nor calibrated inlier probabilities.

Raw recovered weights are available through `predict_with_details(..., details=("instance_weights",))`. They need not reproduce the fitted training weights.
Frozen or hard instance-weight regimes allow no instance-weight recovery, and raise when recovery or scoring is requested. A query-policy override changes query recovery but leaves the fitted scoring reference unchanged.

### Continue fitting

If you want to continue an existing fit on the same data, you can use `resume`:

```python
continued = network.resume(X, y, max_iter=100)
```

For this, the data must have the same rows, in the same order, with the same targets and weights as the original fit.
Here, `max_iter=100` means at most 100 iterations **in total**, including those already completed.
If the model has already converged, increasing this limit alone does not make it continue; a tighter `tol` may be needed.
The returned network contains the continued fit, while the original `network` remains unchanged.

To adapt the learned parameters to *different*, *compatible* observations, you can use `fine_tune` instead:

```python
# A new batch with the same feature and class meanings
new_X = torch.tensor([[0.02], [0.15], [0.85], [0.98]], dtype=X.dtype, device=X.device)
new_y = torch.tensor([0, 0, 1, 1], device=y.device)
adapted = network.fine_tune(new_X, new_y, max_iter=20)
```

Fine-tuning starts *from the learned parameters* and rebuilds the affiliations for the supplied data.
It returns a new network with a new loss history and iteration count. The model structure and feature and output meanings must remain compatible.

### Start again

Calling `Network.fit(recipe, X, y, ...)` starts a fresh fit.
Use this when you want to initialise again or change the model structure, for example by adding a hidden block.
You can also supply an `initial_state`, as described above, to control the starting point of the fresh fit.

### Reuse learned parameters

`capture_current_state` creates an `InitialState` from selected learned parameters for use in a fresh fit. `network.initial_state` contains the original starting values.

For example, we can reuse the input geometry of our classifier to start a regression model:

```python
from entlearn import RegressionHead

captured_input = network.capture_current_state(blocks=("input",))
regression_recipe = Recipe.chain(captured_input.input_geometry.input, RegressionHead())
regression_targets = X[:, 0].square()
regression_network = Network.fit(
    regression_recipe,
    X,
    regression_targets,
    initial_state=captured_input,
    seed=7,
    max_iter=20,
)
```

Here, the regression head is initialised normally, while the input starts from the learned geometry and can continue learning.
Using `captured_input.input_geometry.input` preserves its active cluster count, including any pruning during the original fit.

Blocks are matched by name and must have compatible dimensions and feature meanings. Unmatched captured blocks are ignored with a warning. Select the blocks to reuse with `blocks`, or omit it to capture all fitted parameter groups.
The new fit computes fresh affiliations and diagnostics.
See the [Network reference](../reference/network.md#entlearn.Network.capture_current_state) for the full compatibility rules.

## Save the network

You can save a fitted network to a `.safetensors` file and load it later:

```python
network.save("network.safetensors")
loaded = Network.load("network.safetensors")
loaded_answer = loaded.predict(query)
```

The file is self-contained: tensors and metadata are stored together.
Both save modes retain the recipe, fitted parameters and prediction policy, diagnostics, original initial states and any retained fitted members.

The default save **omits training affiliations and instance weights**, reducing the stored information tied to individual training rows. The loaded model supports prediction, reconstruction of supplied observations and fine-tuning. Resuming the original fit requires a resumable save.
Loading places the model on the CPU by default and preserves its computation dtype. To load onto another device, you can pass `device` to `Network.load`, provided that the query tensors reside on that device too.

To **resume** training later, include the training state:

```python
network.save("checkpoint.safetensors", resumable=True)
checkpoint = Network.load("checkpoint.safetensors")
print(checkpoint.can_resume)
continued = checkpoint.resume(X, y, max_iter=100)
```

The main differences after loading are:

| Operation | Default save | `resumable=True` |
| --- | --- | --- |
| Predict, reconstruct and inspect fitted parameters | Yes | Yes |
| Read fit diagnostics | Yes | Yes |
| Recover query weights and compute inlier percentiles | If recovery was enabled | If recovery was enabled |
| Inspect training affiliations and instance weights | No | Yes |
| Reports requiring training state, such as manifold tangent participation | No | Yes |
| Capture learned parameters or fine-tune | Yes | Yes |
| Resume the original fit | No | Yes |

!!! warning "Considerations for resuming a fit from a saved model"

    You must still retain the original data, targets and weights, and supply them in the same order when resuming. The loader checks the saved state, but cannot verify that the data you later supply is identical to the original data.
    
    Keep the preprocessing steps, feature meanings and mappings between class codes and external labels alongside the model: `Network` does not manage these for you.

### What loading checks

`Network.save` and `Network.load` accept only paths ending in `.safetensors`.

`Network.load` checks the whole file before returning a network: the format and its version, the metadata, the exact set of saved tensors, their consistency with the recipe and the fitted schema, the computation dtype, and the retained states and members.
A file that fails any check, or a request for an unavailable device, raises a `ValueError`. A missing file raises a `FileNotFoundError`:

```python
try:
    Network.load("missing.safetensors")
except FileNotFoundError:
    print("No saved network at this path")
```

The file records two versions.
The **format version** describes how the network is stored, and loading accepts only the current one.
The **package version** records which version of `entlearn` wrote the file. It is kept for your information only: loading does not require it to match the installed version.
