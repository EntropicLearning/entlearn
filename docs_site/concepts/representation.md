# Representing observations

An EON model describes the observations with a sequence of clusterings and relations.
The input clustering relates observations to feature-space geometry.
Hidden clusterings relate to their neighbours through transition matrices.
The head relates the final clustering to classes or continuous targets.

This page introduces these representations. [The objective](loss.md) combines them into a loss, [Learning the model](learning.md) derives their updates, and [Prediction](prediction.md) explains their use on new observations. [Notation](notation.md) lists tensor shapes and symbols.

## Observations and affiliations

Let $T$ be the amount of training observations, and the input block (block $0$) be followed by $N$ hidden blocks. Block $n$ has $K_n$ clusters and an affiliation matrix $\Gamma^{(n)}$ of shape $T\times K_n$. The row $\gamma_t^{(n)}$ describes observation $t$'s membership in its clusters:

$$
\Gamma_{tk}^{(n)}\geq 0,\qquad \sum_k\Gamma_{tk}^{(n)}=1.
$$

A hard affiliation affiliates observations to one cluster only.
A soft affiliation, instead, distributes it among more clusters.
Each clustering block has its own temperature `epsilon`, which controls the regime through the [affiliation update](learning.md#affiliation-updates).

For example, a row $(0.8,0.2)$ assigns most membership to the first cluster.
Thus, this observation can contribute to both clusters' fitted parameters and to the prediction formed from those parameters.

The input block's affiliations depend on feature-space errors.
Hidden affiliations depend on the connections on either side during fitting.
Inspect the training matrices with `network.inspect("training_affiliations")`.
Affiliations for new observations are computed separately during prediction.

## Standard input geometry

A standard `Input` has a representative for each cluster and feature type:

| Quantity | Shape | Representation |
| --- | --- | --- |
| $C$ | $K_0\times D_{\mathrm{cont}}$ | Continuous centroid $c_k$ in row $k$ |
| $C_j^{\mathrm{cat}}$ | $K_0\times M_j$ | Cluster distributions over the $M_j$ levels of categorical feature $j$ |

Continuous features use squared distance from a centroid, and categorical features use cross-entropy against a centroid's level distribution.
Categorical observations may be one-hot codes or distributions over levels.
Either feature type can be absent.
With only categorical features, $C$ has no columns.

Inspect the centroids with `network.inspect("continuous_centroids")` and `network.inspect("categorical_centroids")`.

### The discretisation error

The discretisation error $d_{tk}$ measures the error of representing observation $t$ by cluster $k$:

$$
d_{tk}=\sum_{d=1}^{D_{\mathrm{cont}}} W_D[d](X_{td}-C_{kd})^2
+\delta_{\mathrm{cat}}\sum_{j=1}^{D_{\mathrm{cat}}}W_D[D_{\mathrm{cont}}+j]s_j
\mathrm{CE}(X^{\mathrm{cat}}_{j,t,:},C^{\mathrm{cat}}_{j,k,:}).
$$

Here $\mathrm{CE}(p,q)=-\sum_i p_i\log q_i$.
Feature weights $W_D$ form one probability vector over continuous features followed by categorical features.
Each categorical feature receives one weight regardless of the cardinality of its support.

The scale $s_j$ adjusts the contribution for the number of categorical levels.
`delta_cat`, whose default is one, controls the contribution of categorical errors relative to continuous errors.

??? note "Deriving the categorical scale"
    The categorical scale is $s_j=(1-1/M_j)/\log M_j$. Against a uniform centroid, an observed level has scaled cross-entropy $1-1/M_j$. This equals the squared distance from a one-hot vector of length $M_j$ to the uniform vector. The scale is fixed and derived from the number of levels.

## Manifold geometry

A `ManifoldInput` represents each cluster by a centroid $c_k$ and an orthonormal basis $U_k$ with `subspace_dimension` columns.
They define an affine plane through the centroid, with error:

$$
g_k(x_t)=\Vert(I-U_kU_k^\top)(x_t-c_k)\Vert^2+\alpha\Vert x_t-c_k\Vert^2.
$$

The first term is the squared distance from the plane, and the second penalises distance from its centre with strength `alpha`.

Manifold inputs support continuous features and instance weights.
They have no categorical features or feature weights. `network.inspect("manifold_projectors")` returns the bases $U_k$. The [Manifold guide](../guide/manifold.md) provides more examples on the topic.

## Transitions between clusterings

A hidden block $n$ owns an incoming transition matrix $\Theta^{(n)}$ of shape $K_n\times K_{n-1}$.
Its entries relate source clusters in block $n-1$ to target clusters in block $n$.
The connection's coupling selects its normalisation:

| Coupling | Constraint | Interpretation |
| --- | --- | --- |
| `Coupling.M`, Markov's style, the default | Each column sums to one | Column $k'$ is a distribution over target clusters given source cluster $k'$ |
| `Coupling.S`, simplex | Each row sums to one | Row $k$ describes target cluster $k$ as a distribution over source clusters |

Both use the same [coupling expression](loss.md#coupling-terms).
Under `M` it averages cross-entropies of target affiliations against columns, weighted by source affiliations.
Under `S` it averages cross-entropies of source affiliations against rows, weighted by target affiliations.

You can inspect hidden transitions with `network.inspect("transition_matrices")`.

## Heads

The head connects the final block's $K_N$ clusters to $M$ classes or output dimensions.
With no hidden blocks, the final block is the input block.
A head's class or output axis is determined by the data.
Its cluster axis follows the final block's cluster count.

### Classification head

A `ClassificationHead` owns an $M\times K_N$ transition matrix $\Theta_{\mathrm{out}}$.

`M` coupling is the default, but `S` can be selected with `ClassificationHead("S")` or `ClassificationHead(coupling=Coupling.S)`. Both use the same supervised loss expression with their respective constraints.

### Regression head

A `RegressionHead` owns output centroids $C_y$ of shape $M\times K_N$.
Column $k$ is the target vector represented by cluster $k$.
Output weights $W_M$ control the contribution of each output dimension to the supervised loss.

Inspect either head with `network.inspect("head_parameters")`.
The returned fields include `theta` for classification or `C_y` for regression, with `W_M` when learned or supplied.

## Feature, instance and output weights

Three probability vectors determine contributions to the input and regression terms:

| Weights | Entries | Owner | Temperature | Inspection key |
| --- | --- | --- | --- | --- |
| $W_D$ | Input features | Standard `Input` | `epsilon_D` | `"feature_weights"` |
| $W_T$ | Training observations | Either input block | `epsilon_T` | `"training_instance_weights"` |
| $W_M$ | Regression output dimensions | `RegressionHead` | `epsilon_M` | `"head_parameters"` |

By default, they are fixed.
Finite temperatures enable their [updates](learning.md#weight-updates), which favour entries with lower costs. Feature weights favour features with lower within-cluster error, instance weights favour observations with lower discretisation error, and output weights favour dimensions with lower prediction residuals.

!!! note "Sample weights and instance weights"
    Instance weights start at the normalised supplied sample weights, or uniform weights when none are supplied. Sample weights also act on other loss terms and remain fixed. [Observation weighting](loss.md#observation-weighting) explains their distinct roles.

Continue with [The objective](loss.md) to see how these representations are fitted together.
