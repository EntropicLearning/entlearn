# Concepts

The [eSPA+ walkthrough](https://entropiclearning.github.io/entlearn/0.1.0/guide/introduction/#espa-walkthrough) introduces the objective for a model with one input block and a classification head. This section describes EON models with hidden blocks and/or regression heads, explaining the parameters and update rules. Symbols are listed in [Notation](https://entropiclearning.github.io/entlearn/0.1.0/concepts/notation/index.md) and terms in [Glossary](https://entropiclearning.github.io/entlearn/0.1.0/concepts/glossary/index.md).

## The components

An EON model is a stack of clusterings of the same observations, fitted by minimising a single objective.

| Object                                                                                                                    | Symbol                            | What it is                                                                                        | Inspect it with                                                              |
| ------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [Loss](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/index.md)                                          | (L)                               | The objective function minimised during fitting.                                                  | `network.diagnostics.loss_history`                                           |
| [Affiliations](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#observations-and-affiliations)  | (\\Gamma)                         | How strongly each observation belongs to each cluster of a block.                                 | `"training_affiliations"`                                                    |
| [Transitions](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#transitions-between-clusterings) | (\\Theta)                         | How the clusters of one block relate to the clusters of the next.                                 | `"transition_matrices"`                                                      |
| [Geometry](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#standard-input-geometry)            | (C), (d\_{tk})                    | Cluster centroids, local planes and discretisation errors.                                        | `"continuous_centroids"`, `"categorical_centroids"`, `"manifold_projectors"` |
| [Weights](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#feature-instance-and-output-weights) | (W_D), (W_T), (W_M)               | The contribution of features, training observations, and regression output dimensions to the loss | `"feature_weights"`, `"training_instance_weights"`, `"head_parameters"`      |
| [Heads](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#heads)                                 | (\\Theta\_{\\mathrm{out}}), (C_y) | How the affiliations of the final block become a prediction                                       | `"head_parameters"`                                                          |

The names in quotes can be passed to `network.inspect`, as shown in [Inspect the fit](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#inspect-the-fit). `"training_affiliations"` holds the affiliations of the training observations. For new observations, request them from `network.predict_with_details(..., details=("affiliations",))`.

## How they compose

Data flows from the input block, through any hidden blocks, to the head:

```
flowchart LR
    X[Observations] -->|geometry| A["Input affiliations Γ⁽⁰⁾"]
    A -->|"transition Θ⁽¹⁾"| B["Hidden affiliations Γ⁽¹⁾"]
    B -->|head| Y[Prediction]
```

The input block compares every observation with every centroid, using a **discretisation error** (d\_{tk}) for each cluster (k) and observation (t). A standard `Input` uses weighted squared distance for continuous features and cross-entropy for categorical ones. A `ManifoldInput` measures the distance from each cluster's local plane (see [Manifold](https://entropiclearning.github.io/entlearn/0.1.0/guide/manifold/index.md)).

Each clustering block converts assignment costs into affiliations. At `epsilon=0`, an observation is assigned exclusively to the cluster with minimum cost. At positive temperatures, the affiliations are a softmax of the negative cost, so an observation can share its affiliation between several clusters.

Each hidden block owns an incoming **transition matrix**. The coupling term measures agreement between affiliations of adjacent blocks, using the connection's `delta` as strength. With `Coupling.M`, columns are distributions over target clusters. With `Coupling.S`, rows are distributions over source clusters.

The head adds a supervised loss on labelled observations. A `ClassificationHead` uses cross-entropy between the labels and the label affiliations generated by the network. A `RegressionHead` uses squared residuals between targets and the network's output.

Feature weights (W_D) and instance weights (W_T) act on the input term. Output weights (W_M) act on the regression term. Supplied sample weights, when given, weight the coupling and output terms and the affiliation entropy, and contribute to the output weighting together with any class or task weights. Instance weights start at the normalised sample weights and are fixed there if not learned.

Affiliations and learned weight vectors receive entropy rewards, scaled by their temperatures. The temperatures control the balance between minimising the costs and spreading weight across entries.

## One iteration

Fitting updates one group of parameters at a time while the others are held fixed, using block-coordinate descent. A single **iteration** visits the blocks in reverse order, from the head to the input:

1. The head updates its parameters: (\\Theta\_{\\mathrm{out}}), or (C_y) followed by (W_M) when the output weights are learned.
1. Each hidden block, from the last to the first, updates its affiliations and then its incoming transition matrix.
1. The input block updates its affiliations, followed by any learned instance weights. A standard `Input` then updates any learned feature weights and its centroids. A `ManifoldInput` updates its centroids and then its plane bases.

Each update minimises the loss over one group of parameters, while the remainder is fixed. As such, the loss never increases within an iteration. The total iteration budget is set with `max_iter`.

Fitting removes clusters with zero affiliation mass, together with their associated parameters. Therefore, a fitted block can have fewer clusters than those requested in a recipe.

## Prediction

A new observation takes the same path in a single forward pass, with every learned parameter fixed. The input block computes its affiliations from the fitted geometry, and each hidden block assigns its affiliations from the incoming transition alone.

The head then forms the prediction:

- Classification uses a geometric read-out with a calibrated output temperature. M heads also support an arithmetic read-out.
- Regression returns the affiliation-weighted average of the output centroids.

[Iterative prediction](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#predict-new-observations) repeatedly updates the new observations' affiliations and prediction coordinates while keeping learned parameters fixed. [Prediction and calibration](https://entropiclearning.github.io/entlearn/0.1.0/concepts/prediction/index.md) describes the prediction modes and the read-outs in more detail.
