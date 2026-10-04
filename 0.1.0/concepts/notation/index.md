# Notation

This page collects the dimensions, tensor shapes and settings shared across Concepts. Symbols used in a single derivation are defined beside their equations.

## Conventions

Observations are rows: (X) has shape (T\\times D\_{\\mathrm{cont}}) and (\\Gamma^{(n)}) has shape (T\\times K_n). The EON paper uses observations as columns, so these matrices are transposed relative to the paper.

The input block is (n=0) and hidden blocks are (n=1,\\dots,N). Transition (\\Theta^{(n)}) connects block (n-1) to block (n). Indices (t), (k), (m), (d) and (j) refer to observations, clusters, classes or output dimensions, continuous features and categorical features, respectively.

Lowercase (x_t), (\\gamma_t^{(n)}) and (c_k) denote rows of (X), (\\Gamma^{(n)}) and (C). Hats mark predictions. A prime marks a quantity associated with new observations.

## Dimensions and tensor shapes

| Symbol                                           | Meaning                                                                                           |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| (T), (T')                                        | Numbers of training observations and observations in a prediction batch                           |
| (D), (D\_{\\mathrm{cont}}), (D\_{\\mathrm{cat}}) | Total, continuous and categorical feature counts, with (D=D\_{\\mathrm{cont}}+D\_{\\mathrm{cat}}) |
| (M), (M_j)                                       | Number of classes or regression outputs, and number of levels of categorical feature (j)          |
| (N), (K_n)                                       | Number of hidden blocks and fitted cluster count of block (n)                                     |
| (d\_{\\mathrm{sub}})                             | Manifold plane dimension, set by `subspace_dimension`                                             |

| Quantity                                    | Shape                                              | Meaning                                                    |
| ------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------- |
| (X), (X_j^{\\mathrm{cat}})                  | (T\\times D\_{\\mathrm{cont}}), (T\\times M_j)     | Continuous features and categorical level distributions    |
| (\\Pi), (Y)                                 | (T\\times M)                                       | Classification target distributions and regression targets |
| (\\hat\\Pi), (\\hat Y)                      | (T'\\times M)                                      | Predicted class distributions and regression outputs       |
| (\\Gamma^{(n)})                             | (T\\times K_n)                                     | Affiliations in clustering block (n)                       |
| (C), (C_j^{\\mathrm{cat}})                  | (K_0\\times D\_{\\mathrm{cont}}), (K_0\\times M_j) | Continuous and categorical input centroids                 |
| (U_k)                                       | (D\_{\\mathrm{cont}}\\times d\_{\\mathrm{sub}})    | Orthonormal basis of manifold cluster (k)                  |
| (\\Theta^{(n)}), (\\Theta\_{\\mathrm{out}}) | (K_n\\times K\_{n-1}), (M\\times K_N)              | Hidden and classification-head transitions                 |
| (C_y)                                       | (M\\times K_N)                                     | Regression output centroids, one cluster per column        |
| (W_D), (W_T), (W_M)                         | (D), (T), (M)                                      | Feature, instance and regression output weights            |
| (w)                                         | (T)                                                | Supplied sample weights, normalised to sum to one          |

[Representing observations](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/index.md) defines these quantities and their normalisation constraints. The output weightings (Q) and (\\omega) are defined in [Observation weighting](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/#observation-weighting).

## Strengths and temperatures

| Symbol                                               | Setting                               | Role                                                                          |
| ---------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| (\\delta_n)                                          | Connection `delta`                    | Strength of the connection into block (n), with (\\delta\_{N+1}) for the head |
| (\\delta\_{\\mathrm{cat}})                           | `delta_cat`                           | Relative scale of categorical input errors                                    |
| (\\alpha)                                            | `alpha`                               | Centroid-distance penalty in manifold geometry                                |
| (\\varepsilon_n)                                     | Block `epsilon`                       | Affiliation temperature                                                       |
| (\\varepsilon_D), (\\varepsilon_T), (\\varepsilon_M) | `epsilon_D`, `epsilon_T`, `epsilon_M` | Feature-, instance- and output-weight temperatures                            |
| (\\varepsilon_P)                                     | `PredictConfig.epsilon_P`             | Geometric classification read-out temperature                                 |

[Learning the model](https://entropiclearning.github.io/entlearn/0.1.0/concepts/learning/#affiliation-temperatures) describes affiliation regimes and [weight regimes](https://entropiclearning.github.io/entlearn/0.1.0/concepts/learning/#temperatures-and-effective-dimension). [Prediction](https://entropiclearning.github.io/entlearn/0.1.0/concepts/prediction/#the-calibrated-output-temperature) explains the output temperature. Dtype machine precision (\\epsilon) is defined with the [numerical conventions](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/#logarithms-and-zeros).
