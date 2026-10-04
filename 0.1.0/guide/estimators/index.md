# Using estimators

The estimators provide a convenient way to use `Network` with a scikit-learn-compatible interface. We currently offer `EONClassifier` and `EONRegressor`. Both wrap a `Network` and also take care of housekeeping tasks like converting the data to tensors, keeping track of the columns and encoding categorical values.

## Requirements

To use them, install the package with the `sklearn` extra:

```
pip install "entlearn[sklearn]"
```

## Creating and using estimators

Specify a `Recipe` whose output block matches the task: `ClassificationHead` for a classifier, or `RegressionHead` for a regressor.

For example, using data stored in **NumPy arrays**:

```
import numpy as np
from entlearn import ClassificationHead, Input, Recipe
from entlearn.scikit_adapter import EONClassifier

X = np.array([[0.0], [0.1], [0.9], [1.0]])
y = np.array(["low", "low", "high", "high"])
recipe = Recipe.chain(Input(K=2), ClassificationHead())
model = EONClassifier(recipe, random_state=7, max_iter=20).fit(X, y)
```

`random_state` controls the random initialisation and is the estimator's equivalent of the `seed` argument used by `Network.fit`. The fitted estimator can then predict labels, return class probabilities, or calculate a score:

```
X_eval = np.array([[0.05], [0.95]])
y_eval = np.array(["low", "high"])
labels = model.predict(X_eval)
probabilities = model.predict_proba(X_eval)
accuracy = model.score(X_eval, y_eval)
```

You can also pass a **pandas** or eager **Polars** `DataFrame`, provided the corresponding library is installed separately. With an array, columns are identified by their positions. With a named DataFrame, the estimator records the column names as well. The same names and order should be kept when predicting.

What about lazy Polars DataFrames?

The estimators do not accept a Polars `LazyFrame`, and never run a lazy query for you: passing one raises a `TypeError`. To use them, call `collect()` first, which runs the query and loads its result into memory.

```
import polars as pl

table = query.collect()
X_polars = table.drop("label")
y_polars = table["label"].to_numpy()
polars_model = EONClassifier(
    recipe, categorical_features="from_dtype", random_state=7, max_iter=20
).fit(X_polars, y_polars)
polars_probabilities = polars_model.predict_proba(X_polars)
```

The values are encoded using the category vocabulary learned during fitting, not the category codes stored by Polars. For prediction, you can pass either a pandas or a Polars `DataFrame`, as long as the column names, their order and the category values match those used for fitting.

### Categorical features

If some features are **categorical**, you can specify their column index using `categorical_features`, for example `categorical_features=[1, 3]`. For `DataFrame`s, `categorical_features="from_dtype"` identifies categorical, enum, object and string columns from their dtypes. The estimator learns the category encodings during fitting and reuses them for prediction. Missing or previously unseen categories are rejected. For an object-dtype NumPy array, specify the categorical column positions explicitly, since the shared dtype does not identify which columns are categorical. A categorical column with only one observed value is dropped with a warning, as it cannot tell observations apart. Keep that column in the input when predicting, since it remains part of the recorded layout.

Feature scaling

The estimator converts and encodes the data, but does **not** automatically scale continuous features. We recommend scaling continuous features, and regression targets, to [0, 1]. The model computes the squared distance between an observation (x_t) and a centroid (c_k) through the expansion (\\Vert x_t-c_k\\Vert^2=\\Vert x_t\\Vert^2-2x_t\\cdot c_k+\\Vert c_k\\Vert^2), which avoids forming every difference (x_t-c_k). Regression targets are compared with the output centroids in the same way. Its rounding error grows with the size of the values rather than with the distance, so with large feature values a small distance can be lost to cancellation. In [0, 1], the terms stay small, and so does the error. The [categorical costs](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#the-discretisation-error) are also scaled to match that range, and the suggested [search ranges](https://entropiclearning.github.io/entlearn/0.1.0/guide/model_selection/#choosing-search-ranges) assume it.

Fit any preprocessing on the training rows only, and apply the same transformation to evaluation data and new observations.

### Sample and class weights

Sample weights are positive weights representing the relative contribution of each data point. They can be provided by passing `sample_weight` to `fit`.

sample weights are not related to the instance weights

Instance weights are **learned** by the model during the optimization process, whereas sample weights are provided by you and fixed. Both can be used in conjunction. Your provided sample weights will affect the coupling terms, affiliation entropy and output term, see [Weights](https://entropiclearning.github.io/entlearn/0.1.0/concepts/learning/#instance-weights) and [Loss](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/index.md).

For classification, `class_weights` specifies the relative weight of each class in the output term. Class weights can be provided as a dictionary (including every possible class) or as a vector (in sorted class order).

```
weighted_model = EONClassifier(
    recipe,
    class_weights={"low": 1.0, "high": 2.0},
    random_state=7,
    max_iter=100,
).fit(X, y, sample_weight=np.array([1.0, 0.5, 1.0, 1.0]))
```

Here, the second observation has half the sample weight, while the `"high"` class has twice the class weight of `"low"`.

### Regression and reconstruction

For a continuous target, use `EONRegressor` with a `RegressionHead`:

```
from entlearn import RegressionHead
from entlearn.scikit_adapter import EONRegressor

regressor = EONRegressor(
    Recipe.chain(Input(K=2), RegressionHead()),
    random_state=7,
).fit(X, X[:, 0] ** 2)
predicted_targets = regressor.predict(X_eval)
reconstructed_features = regressor.reconstruct(X_eval)
```

Here, `predict` estimates the target, while `reconstruct` approximates the input features by blending centroids. Both estimators support reconstruction. [Manifold](https://entropiclearning.github.io/entlearn/0.1.0/guide/manifold/index.md) explains the local-plane version.

`reconstruct` returns a `TabularReconstruction`: `continuous` contains reconstructed continuous values, and `categorical` contains probability matrices. Use `continuous_indices`, `categorical_indices` and `categories` to recover their original column positions and category meanings. Dropped constant categories return one-column distributions of ones. These distributions express a mixture of centroids, not calibrated category predictions or missing-data imputations.

For target-dependent row weights, `regression_weighting` accepts a callable receiving the labelled target tensor and returning one finite, strictly positive tensor weight per labelled row. This is separate from `sample_weight` and from [output-dimension weighting](https://entropiclearning.github.io/entlearn/0.1.0/guide/hyperparameters/#regression-output-weights).

## Labels, scores, and fitted attributes

Unlike the direct `Network` interface, a **classifier**'s `predict` returns the original class labels, such as `"low"` and `"high"` in the example above. As you may expect, `predict_proba` can be used to obtain probabilities instead. The labels are stored in `model.classes_`: column `j` of the probability matrix corresponds to `model.classes_[j]`.

```
print(model.classes_)
```

For **regression**, the prediction shape follows the shape of the targets used for fitting. If `y` is a vector with shape `(n_samples,)`, predictions have shape `(n_queries,)`. If `y` is a matrix with shape `(n_samples, n_targets)`, predictions have shape `(n_queries, n_targets)`, also when there is only one target column.

What does `score` measure? For a **classifier**, it returns accuracy; for a **regressor**, it returns the coefficient of determination, (R^2). These measure predictive performance and are different from the training objective, which is recorded in `loss_curve_`. Use evaluation data to assess how well the model generalises.

Unlabelled observations

During fitting, a numeric value of `-1` marks an **unlabelled** classification row, which can still be used for training without contributing to the output term. For regression, a row whose targets are all `NaN` is treated as unlabelled. Partially missing multi-target rows are rejected by the estimator. When calling `score`, make sure to pass only fully labelled evaluation rows as they are not filtered automatically.

After fitting, the estimator exposes several useful attributes:

| Attribute         | Contains                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `network_`        | The fitted `Network`                                                                            |
| `loss_curve_`     | The initial training loss and the loss after each completed iteration                           |
| `n_iter_`         | The number of completed training iterations                                                     |
| `feature_layout_` | The continuous, categorical and dropped column positions, and the learned category vocabularies |

For example, you can inspect the learned centroids through the underlying network:

```
centroids = model.network_.inspect("continuous_centroids")["input"]
print(model.n_iter_)
print(model.loss_curve_)
```

Using the estimator's `predict` and `predict_proba` methods for tabular data automatically applies its column and label mappings. They then predict with the fitted prediction policy: by default a single forward pass and, for classification, the geometric read-out; see [Hyperparameters](https://entropiclearning.github.io/entlearn/0.1.0/guide/hyperparameters/#fitting-and-prediction-controls).

The fitted attributes are set only once a fit has succeeded. If `fit` raises, including when a warning is turned into an error, the estimator keeps its previous fit, or stays unfitted.

## Inlier percentiles and feature reports

Both estimators expose query weights and inlier percentiles, if instance-weight recovery is enabled:

```
support_model = EONClassifier(
    Recipe.chain(Input(K=2, epsilon_T=0.3), ClassificationHead()),
    random_state=7,
).fit(X, y)
query_weights = support_model.recover_instance_weights(X_eval)
support_scores = support_model.score_samples(X_eval)
```

Weights are the outlier measures, whereas the inlier scores are the same values, ranked against the fitted reference (i.e., the measures obtained on the training data). They describe [how well a query fits the learned input representation](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#reconstruction-and-inlier-percentiles), and are not automatically comparable across refits.

For standard inputs, `model.feature_importances_` returns fitted feature weights in original column order, with zeros for dropped columns. With non-uniform weights, `model.active_features(tol=1.0)` selects original column indices whose weights exceed the uniform share. For manifolds, importance instead measures [tangent participation](https://entropiclearning.github.io/entlearn/0.1.0/guide/manifold/#select-dimension-and-interpret-limitations).

```
dimensions = model.effective_dimensions(normalise=False)
parameter_count = model.count_parameters()
```

Effective dimensions reflects how concentrated the fitted distributions are. Parameter counting uses active fitted dimensions, excludes fixed weights (for instance, if feature weights are not learned, they are not counted as active parameters) and accounts for constraints by default. Training affiliations can be included in the parameter count by using `include_affiliations=True`, provided the training state was retained. For tensor models, the corresponding functions are in [reporting helpers](https://entropiclearning.github.io/entlearn/0.1.0/reference/helpers/#fitted-reporting) and accept a `Network`.

## Saving an estimator

Saving an estimator must preserve its class labels, column layout and categorical encodings alongside the fitted `Network`. Use `joblib` to save the complete estimator. The `.safetensors` format described in [Using Networks](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#save-the-network) saves the underlying `Network`:

```
import joblib

joblib.dump(model, "estimator.joblib")
restored = joblib.load("estimator.joblib")
restored_labels = restored.predict(X_eval)
```

The loaded estimator can be used with the same tabular inputs. It also retains the underlying network's ability to resume, if that state was available when saving.

Only load files from trusted sources

Loading a pickle or joblib file can execute Python code. Record the Python and package versions used to save the estimator, and use the same environment when loading it.

## Refit and reuse

What happens if you call `fit` again on the same estimator? By default, `warm_start` is set to `False`, so a new model is fitted from scratch using the current configuration. To reuse a previously fitted model, choose one of the following modes:

| `warm_start`  | Behaviour                                       |
| ------------- | ----------------------------------------------- |
| `False`       | Start a fresh fit                               |
| `"resume"`    | Continue the existing fit on its original data  |
| `"fine_tune"` | Adapt the learned parameters to compatible data |

For example, to resume the classifier fitted above:

```
model.set_params(warm_start="resume", max_iter=100)
model.fit(X, y)
```

Supply the same training rows, in the same order, with the same targets and any weights used originally. The model must retain its training state, and its recipe must be unchanged. Here, `max_iter` is the maximum **total** number of iterations, which includes those already completed. An already-converged model may need a tighter `tol` to continue, even with a larger iteration limit.

To fit on new observations while keeping the learned parameters as the starting point, use `"fine_tune"`:

```
new_X = np.array([[0.02], [0.15], [0.85], [0.98]])
new_y = np.array(["low", "low", "high", "high"])

model.set_params(warm_start="fine_tune", n_inits=1, max_iter=20)
model.fit(new_X, new_y)
```

Fine-tuning starts a new loss history and iteration count. The feature layout, class labels and category vocabularies must remain compatible with the existing model. You can change compatible objective settings, such as temperatures, but not the model structure or cluster counts. Both continuation modes require `n_inits=1` and update the estimator's fitted attributes after a successful fit. An incompatible continuation raises, rather than silently starting a fresh fit.

If the original fit retained several members, each continues independently and the original winner remains selected.

Choose the continuation mode explicitly

`warm_start=True` is not accepted. Use `"resume"` or `"fine_tune"`. If the estimator has not yet been fitted, either mode starts a fresh fit and emits a warning.

If you want a **separate** estimator with the same configuration, use scikit-learn's `clone`:

```
from sklearn.base import clone

fresh_model = clone(model).set_params(warm_start=False)
fresh_model.fit(X, y)
```

Cloning preserves the constructor parameters, but not the fitted network, label mappings or learned vocabularies. For parameter searches, see [Model selection](https://entropiclearning.github.io/entlearn/0.1.0/guide/model_selection/index.md). See the [adapter reference](https://entropiclearning.github.io/entlearn/0.1.0/reference/scikit_adapter/index.md) for the complete continuation controls.
