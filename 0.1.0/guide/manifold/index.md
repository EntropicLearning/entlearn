# Manifold

Sometimes observations have many features, but their variation follows only a few directions. Consider points lying near a curve in three-dimensional space: locally, a short segment of that curve can be approximated by a line. `ManifoldInput` uses this idea to represent data with a combination of local linear subspaces. It implements the input geometry of [Entropy-Optimal Manifold Clustering (EOMC)](https://arxiv.org/abs/2512.17926).

## Geometry: local affine planes

A standard `Input` represents each cluster using a centroid, with feature weights controlling how distances are measured. `ManifoldInput` gives each cluster a centroid and also a set of orthonormal directions. Together, these define a local affine plane, i.e., a linear subspace passing through the centroid.

The parameter `subspace_dimension` specifies the number of directions per cluster. For example, `subspace_dimension=1` gives a line, and `subspace_dimension=2` gives a plane. All clusters use the same dimension, but each has its own orientation. This allows several local planes to approximate a curved structure, without requiring a single coordinate system for the whole dataset.

How is the distance to a cluster measured? Write (c_k) for its centroid and (U_k\\in\\mathbb{R}^{D\_{\\mathrm{cont}}\\times d\_{\\mathrm{sub}}}) for its orthonormal basis, where (d\_{\\mathrm{sub}}) is `subspace_dimension`. The discretisation error for an observation (x) is

\[ g_k(x)=\\underbrace{\\left\\Vert(I-U_kU_k^\\top)(x-c_k)\\right\\Vert^2}_{\\text{distance from the plane}} +\\underbrace{\\alpha\\left\\Vert x-c_k\\right\\Vert^2}_{\\text{distance from the centroid}}. \]

Here, (I) is the identity matrix and (U_kU_k^\\top) projects onto the cluster's subspace. The non-negative parameter `alpha` controls how much distance from the centroid matters. Displacement along the plane is penalised with weight (\\alpha), while displacement perpendicular to it is penalised with weight (1+\\alpha). At `alpha=0`, moving along the plane has no cost, even far from the centroid. A positive value also penalises that movement.

Continuous features only

`ManifoldInput` does not accept categorical features and has no feature-weight parameter. It can still learn instance weights through `epsilon_T`.

## When to use a manifold input

Use `ManifoldInput` when local subspaces are part of your hypothesis about how the data were generated, for example observations that lie near a curved surface. Estimating a plane for every cluster needs enough observations, and costs more time and memory than a standard `Input`, see [Suitability and scaling](https://entropiclearning.github.io/entlearn/0.1.0/guide/suitability/#cost).

## Fit a small regression model

You can use `ManifoldInput` in a recipe in the same way as `Input`. The output block determines the prediction task. For example, `RegressionHead` predicts a continuous target from the cluster affiliations. The input geometry and the output parameters are fitted jointly.

```
import numpy as np
from entlearn import ManifoldInput, Recipe, RegressionHead
from entlearn.scikit_adapter import EONRegressor

recipe = Recipe.chain(
    ManifoldInput(K=2, subspace_dimension=1, alpha=0.1, epsilon=0.1),
    RegressionHead(),
)
model = EONRegressor(recipe, random_state=7, max_iter=20).fit(X, y)
```

The main input settings are:

| Parameter            | Controls                                                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `K`                  | The requested number of clusters, each with its own local plane                                                                                                   |
| `subspace_dimension` | The number of directions per plane, between 1 and the number of input features                                                                                    |
| `alpha`              | The additional cost of displacement from the centroid                                                                                                             |
| `epsilon`            | The affiliation temperature: zero gives hard assignments, positive values allow soft affiliations                                                                 |
| `epsilon_T`          | The instance-weight temperature: finite positive values learn weights. The default infinity keeps supplied sample weights, or uniform weights when none are given |

## Inspect planes and reconstruct

Prediction estimates the target `y`, while reconstruction approximates the input features.

You can inspect the local planes through the fitted network:

```
bases = model.network_.inspect("manifold_projectors")["input"]
centroids = model.network_.inspect("continuous_centroids")["input"]
print(bases.shape)
print(centroids.shape)
```

The returned tensors contain the orthonormal bases (U_k). Their shape is `(n_active_clusters, n_features, subspace_dimension)`, while the centroids have shape `(n_active_clusters, n_features)`. The corresponding projection matrix is (U_kU_k^\\top).

For each query, every cluster proposes a projection onto its own plane. The reconstruction combines these projections using the query's input affiliations:

\[ \\hat{x}_t=\\sum_k\\Gamma_{tk}^{(0)}\\left[c_k+U_kU_k^\\top(x_t-c_k)\\right]. \]

The affiliations are those obtained at the end of the selected prediction procedure, including refinement when iterative prediction is used. With a hard affiliation, reconstruction is a projection onto one plane. With soft affiliations, it blends projections from several planes.

## Select dimension and interpret limitations

Compare candidate subspace dimensions using validation data and a metric appropriate to the task.

If an initial cluster's observations do not span enough independent directions, the initialiser emits a warning and completes the basis deterministically to the requested dimension. Those additional directions are not evidence of structure discovered in the data.

Feature importance has a different meaning here

For a manifold input, `model.feature_importances_` reports how strongly each feature participates in the learned tangent subspaces, averaged using training affiliations and instance weights.

[Tutorial 6](https://entropiclearning.github.io/entlearn/0.1.0/tutorials/series/#tutorial-6) explores an S-curve, local planes and queries outside the surface. For more details, see [regression and reconstruction](https://entropiclearning.github.io/entlearn/0.1.0/guide/estimators/#regression-and-reconstruction) and the [Recipe reference](https://entropiclearning.github.io/entlearn/0.1.0/reference/recipe/index.md).

See [Plotting](https://entropiclearning.github.io/entlearn/0.1.0/guide/plotting/#local-manifold-geometry) for plotting local planes and reconstructions.
