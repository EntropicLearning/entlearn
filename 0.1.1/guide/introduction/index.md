# Introduction

## Small data

**Big data** refers to having many observations relative to the number of features of the data. In many real world cases, however, a problem may have many features and only few observations. In this case, which we denote as **small data**, conventional machine learning (ML) methods may struggle with generalisation.

Consider the following synthetic example, based on a bioinformatics problem. 100 genes (features) have been measured, and we *know* there is a way to easily distinguish which point should be one class or the other (*blue* vs *orange*) by looking at two features only.

Assume the model's perspective: you can use the sliders to select two dimensions to plot. Can you spot which features are the relevant ones?

Reveal the answer

Check features **19** and **72**.

This is a simple example to illustrate that even if the information is present and sufficient for perfect discrimination, it may require solving a *"needle in a haystack"* type of problem. In the situation above, there are 4950 feature pairs that can be inspected, and only *one* of them is useful.

## Pipelining

Conventional ML methods involve combining different steps in a sequence, or *pipeline*. A typical solution for the high-dimensional example described above is to apply Principal Component Analysis (PCA) before passing the data to the ML algorithm of your choice.

```
block-beta
  columns 4

  Data --> PCA
  PCA --> ML
  ML --> Output
```

However, PCA and the ML algorithm are not intrinsically linked: the former is an unsupervised technique, and as such it is not aware of the downstream classification task. Important information could be hidden in dimensions with small variance, which are dropped by PCA when truncating.

Consider the dataset illustrated below. The blue and orange points differ across the "minor axis" (i.e., the low variance direction), while both classes have the same distribution along its "major axis" (dotted line in the plot). PCA with one component keeps the latter, as it explains most of the variance. The projected classes then overlap, while the discarded component would have retained the information needed to separate them.

## What is entropic learning?

Entropic learning is a framework that originated from the work of Prof. Horenko's lab, originating with the Scalable Probabilistic Approximation (SPA) algorithm. The idea is simple: all the steps can be combined as a **single optimisation problem**. For example, feature selection can be included in the problem, without needing a separate preprocessing step (making it optimal with respect to the learning task).

To achieve this, all parts of the task have to be defined mathematically.

[SPA](#spa-scalable-probabilistic-approximation) introduced a formulation to simultaneously:

- Discretise the state space using a set of reference points and soft affiliations (for which K-means is an upper-bound)
- Identify the relevant features

[eSPA](#espa-entropy-optimal-spa) (entropy-optimal SPA) reformulated SPA for classification, and introduced an improved feature selection mechanism based on the principle of maximum entropy to learn a vector of feature weights. [eSPA+](#espa) employs the same upper-bound approximation to eSPA as K-means does to SPA, which can be cheaply solved using coordinate descent, where each substep has an analytical solutions (for an efficient implementation in Julia of eSPA and eSPA+ see [EntropicLearning/eSPA.jl](https://github.com/EntropicLearning/eSPA.jl)).

[SPARTAN](#spartan) (not implemented in this package) replaced the Bayesian classification term in eSPA+ with piecewise-linear regressions. [EON](#eon-entropy-optimal-networks) generalises eSPA+ with hidden blocks, soft affiliations (with a cheaper way of including these compared to SPA) and learned instance weights, and can be applied to both classification and regression tasks. [EOMC](#eomc-entropy-optimal-manifold-clustering) is a related method for using linear manifolds (rather than centroids) to describe the clusters, and the package's [manifold input](https://entropiclearning.github.io/entlearn/0.1.1/guide/manifold/index.md) generalises it to EON models.

### eSPA+ walkthrough

We will first describe how eSPA+ combines the three steps into a single problem rather than starting from EON, as it is a gentle introduction to the perspective used when deriving EON (in fact eSPA+ is a special case of EON), and the explanation is meant to give an idea if you have never heard of the technique.

If you are already familiar with eSPA+ (or you are not afraid of jumping into the mathematics), feel free to skip to the [Concepts](https://entropiclearning.github.io/entlearn/0.1.1/concepts/index.md) pages.

Consider (T) observations of (D) features for a classification problem with (M) classes. We will use (K) clusters in our model to represent the observations. The quality with which an observation can be reconstructed by the (k)th centroid (C\_{k,:}) can be measured with weighted squared distance:

[ d\_{t,k} = \\sum\_{d=1}^D W_d (X\_{t,d} - C\_{k,d})^2 ]

where each feature has a weight (W_d \\in [0,1]) and all weights sum to one.

Assignment of points to clusters can be collected into a matrix (\\Gamma \\in {0,1}^{T \\times K}), where (\\Gamma\_{t,k} = 1) if the (t)th point is affiliated with the (k)th cluster. Each cluster can be mapped to the output classes through the conditional probabilities (\\Theta), where each entry (\\Theta\_{m,k}) represents the probability that a point in cluster (k) belongs to the (m)th class.

Furthermore, the regularisation term on the *entropy* of the feature weights (W) biases them to be more or less sparse.

Combining everything, we can write a joint loss:

\[ L = \\underbrace{\\frac{1}{T} \\sum\_{t,k} \\Gamma\_{t,k} d\_{t,k}}_{\\text{discretisation}} ;\\underbrace{-\\frac{\\delta}{T} \\sum_{t,k,m} \\Gamma\_{t,k} \\Pi\_{t,m} \\log \\Theta\_{m,k}}_{\\text{labelling}} ;\\underbrace{+\\varepsilon_D \\sum_{d} W_d \\log(W_d)}\_{\\text{feature selection}}. \]

The first term captures how well the discretisation can describe the original data. The second term steers towards affiliations and conditional probabilities that are close to the true label probability distributions (\\Pi \\in [0,1]^{T \\times M}), with (\\delta) being the regularisation factor controlling the importance of this. The last term is the negative entropy. Small values of (\\varepsilon_D) concentrate the weights on only a few features, whereas larger values result in more equally weighted features (effectively maximising the entropy).

Training a model can be performed via coordinate descent one parameter at a time, keeping the others fixed, as a sequence of **closed form analytical solutions**, during which the loss is guaranteed to decrease monotonically.

In the [Concepts](https://entropiclearning.github.io/entlearn/0.1.1/concepts/index.md) this will be expanded to a more general problem, which can be solved with EON.

## Literature

### EON — Entropy-Optimal Networks

Bassetti, D., Pospíšil, L., Groom, M., O'Kane, T. J., & Horenko, I. (2025). *An entropy-optimal path to humble AI.* arXiv:2506.17940. [arxiv.org/abs/2506.17940](https://arxiv.org/abs/2506.17940)

BibTeX

```
@article{bassetti2025eon,
  title   = {An entropy-optimal path to humble {AI}},
  author  = {Bassetti, Davide and Pospíšil, Lukáš and Groom, Michael and
             O'Kane, Terence J. and Horenko, Illia},
  journal = {arXiv preprint arXiv:2506.17940},
  year    = {2025},
  doi     = {10.48550/arXiv.2506.17940}
}
```

### EOMC — Entropy-Optimal Manifold Clustering

Horenko, I. (2025). *Linearly-scalable and entropy-optimal learning of nonstationary and nonlinear manifolds.* arXiv:2512.17926. [arxiv.org/abs/2512.17926](https://arxiv.org/abs/2512.17926)

BibTeX

```
@article{horenko2025eomc,
  title   = {Linearly-scalable and entropy-optimal learning of nonstationary
             and nonlinear manifolds},
  author  = {Horenko, Illia},
  journal = {arXiv preprint arXiv:2512.17926},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.17926}
}
```

### SPA — Scalable Probabilistic Approximation

Gerber, S., Pospíšil, L., Navandar, M., & Horenko, I. (2020). *Low-cost scalable discretisation, prediction, and feature selection for complex systems.* Science Advances, 6(5), eaaw0961. [doi.org/10.1126/sciadv.aaw0961](https://doi.org/10.1126/sciadv.aaw0961)

BibTeX

```
@article{gerber2020spa,
  title   = {Low-cost scalable discretisation, prediction, and feature
             selection for complex systems},
  author  = {Gerber, S. and Pospíšil, L. and Navandar, M. and Horenko, I.},
  journal = {Science Advances},
  volume  = {6},
  number  = {5},
  pages   = {eaaw0961},
  year    = {2020},
  doi     = {10.1126/sciadv.aaw0961}
}
```

### eSPA — entropy-optimal SPA

Horenko, I. (2020). *On a Scalable Entropic Breaching of the Overfitting Barrier for Small Data Problems in Machine Learning.* Neural Computation, 32(8), 1563–1579. [doi.org/10.1162/neco_a_01296](https://doi.org/10.1162/neco_a_01296) · [arXiv:2002.03176](https://arxiv.org/abs/2002.03176)

BibTeX

```
@article{horenko2020espa,
  title   = {On a Scalable Entropic Breaching of the Overfitting Barrier
             for Small Data Problems in Machine Learning},
  author  = {Horenko, Illia},
  journal = {Neural Computation},
  volume  = {32},
  number  = {8},
  pages   = {1563--1579},
  year    = {2020},
  doi     = {10.1162/neco_a_01296}
}
```

### eSPA+

Vecchi, E., Pospíšil, L., Albrecht, S., O'Kane, T. J., & Horenko, I. (2022). *eSPA+: Scalable Entropy-Optimal Machine Learning Classification for Small Data Problems.* Neural Computation, 34(5), 1220–1255. [doi.org/10.1162/neco_a_01490](https://doi.org/10.1162/neco_a_01490)

BibTeX

```
@article{vecchi2022espaplus,
  title   = {eSPA+: Scalable Entropy-Optimal Machine Learning Classification
             for Small Data Problems},
  author  = {Vecchi, Edoardo and Pospíšil, Lukáš and Albrecht, Steffen and
             O'Kane, Terence J. and Horenko, Illia},
  journal = {Neural Computation},
  volume  = {34},
  number  = {5},
  pages   = {1220--1255},
  year    = {2022},
  doi     = {10.1162/neco_a_01490}
}
```

### SPARTAN

Horenko, I., Vecchi, E., Kardoš, J., Wächter, A., Schenk, O., O'Kane, T. J., Gagliardini, P., & Gerber, S. (2023). *On cheap entropy-sparsified regression learning.* Proceedings of the National Academy of Sciences, 120(1), e2214972120. [doi.org/10.1073/pnas.2214972120](https://www.pnas.org/doi/10.1073/pnas.2214972120)

BibTeX

```
@article{horenko2023spartan,
  title   = {On cheap entropy-sparsified regression learning},
  author  = {Horenko, Illia and Vecchi, Edoardo and Kardoš, Juraj and
             W{\"a}chter, Andreas and Schenk, Olaf and O'Kane, Terence J. and
             Gagliardini, Patrick and Gerber, Susanne},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {120},
  number  = {1},
  pages   = {e2214972120},
  year    = {2023},
  doi     = {10.1073/pnas.2214972120}
}
```

### EOS — Entropic Outlier Sparsification

Horenko, I. (2022). *Cheap robust learning of data anomalies with analytically solvable entropic outlier sparsification.* Proceedings of the National Academy of Sciences, 119(9), e2119659119. [doi.org/10.1073/pnas.2119659119](https://www.pnas.org/doi/10.1073/pnas.2119659119)

BibTeX

```
@article{horenko2022eos,
  title   = {Cheap robust learning of data anomalies with analytically
             solvable entropic outlier sparsification},
  author  = {Horenko, Illia},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {119},
  number  = {9},
  pages   = {e2119659119},
  year    = {2022},
  doi     = {10.1073/pnas.2119659119}
}
```

## Where to go next

- [Introduction to the package](https://entropiclearning.github.io/entlearn/0.1.1/guide/package/index.md) connects these ideas to the public interfaces.
- [Fit your first network](https://entropiclearning.github.io/entlearn/0.1.1/tutorials/series/#tutorial-1) is the hands-on starting point.
- [Manifold](https://entropiclearning.github.io/entlearn/0.1.1/guide/manifold/index.md) introduces local subspaces after the basic model.
