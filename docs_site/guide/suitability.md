# Suitability and scaling

In this page, we answer the question of whether this package could be a good fit for your problem and of the costs with trying it, together with other information to keep in mind while using it. 
For more details, please refer to [the paper](../guide/introduction.md#eon-entropy-optimal-networks)

## When is EON worth trying?

EON learns a compact representation through exact block-coordinate updates. Its centroids, affiliations and weights can help explain the fitted model.

Consider trying it when:

- the problem presents very few observations, very large number of features, or both (see [small data](introduction.md#small-data))
- not all the features may be relevant for the problem at hand, but it is not clear _a priori_ which could be. 
- interpretability is of paramount importance
- the quality of the data is questionable, and some training example should be discarded. EON models have a way to automatically scale the importance of the data points _as part of the optimization problem_.
- the data is only partially observed, for example some observation have no label
- a model with a small footprint size is desired
- inference speed is of importance

Note that the conditions above are where EON shines, but not necessary conditions for its use!

## Cost

The computational cost of a single iteration of the optimization procedure for an EON model scales as:

$$
\mathcal{O}\left(TDK_0 + TK_{\mathrm{all}}\right),
\qquad
K_{\mathrm{all}}=\sum_{n=1}^{N} K_{n-1}K_n ,
$$

where $T$ is the number of observations, $D$ the number of input features, $K_0$ the number of input clusters, and $K_{\mathrm{all}}$ sums the products of the cluster counts of each pair of blocks joined by a hidden block's transition matrix (see [Notation](../concepts/notation.md)).

That means that **scaling is linear** in most of the parameters.

??? note "On Manifold usage"
    A `ManifoldInput` can increase both time and memory substantially, because each cluster has its own subspace to fit.
    Write $d_{\mathrm{sub}}$ for its `subspace_dimension`. In each iteration:
    
    - the distances project every observation onto every cluster's subspace, which costs $O(TDK_0d_{\mathrm{sub}})$ time and holds $O(TK_0d_{\mathrm{sub}})$ projected coordinates in memory;
    - each cluster's subspace is then re-estimated, one cluster at a time, from the weighted observations around its centroid: through the eigenvectors of their $D\times D$ covariance when $D\le T$, or a singular value decomposition otherwise. This costs $O(K_0TD\min(T,D))$ time and a workspace of $O(TD + \min(T,D)^2)$ values.
    
    Since $d_{\mathrm{sub}}$ is usually smaller than both $T$ and $D$, the second step dominates: the term $TDK_0$ becomes $K_0TD\min(T,D)$, which grows with $D^2$ while $D\le T$.

The cost of an experiment multiplies this by the number of iterations, initialisations and validation fits.

The initialisations and validation fits do not depend on each other, so they can run in parallel.
Set `n_jobs` on an estimator or on `Network.fit` to fit its initialisations in parallel, and the `n_jobs` of a scikit-learn search, such as `GridSearchCV`, to spread its parameter settings and folds over several workers (see [Parallel initialisations](model_selection.md#parallel-initialisations)).

## Limitations

- **The objective is non-convex.** Each update is exact, but the fit can still depend on its initialisation.
- **Only chains can be fitted.** For the time being, a `Network` fits one input block, zero or more hidden blocks and one head; see [Package and Recipes](package.md#what-shapes-can-a-model-have).
- **Settings interact.** Cluster counts, coupling, connection strengths and temperatures compensate for each other, so their values are not interpretable on their own.
- **Large problems remain expensive.** Very large $T\times D\times K_0$, many initialisations or large validation searches take time, although often less than alternative methods.


## Behaviour that may surprise you

- More clusters or more hidden blocks can worsen generalisation and make selection less stable. Compare held-out scores and include small cluster counts in the search.
- Predictive skill can be quite sensitive to the settings, so a poor result may only mean a poor choice of settings. [Model selection](model_selection.md) shows how to search them.
- The initialisation matters too, and selecting a good one can improve performance noticeably, see [Select an initialisation for one recipe](model_selection.md#select-an-initialisation-for-one-recipe).
- A larger temperature makes affiliations or weights more uniform, but an **infinite weight temperature** freezes the weights instead of making them as soft as possible; see [Weights](../concepts/learning.md#temperatures-and-effective-dimension).
- `epsilon=0` gives hard affiliations.
- Feature importance means something different for a standard `Input` and a [`ManifoldInput`](manifold.md#select-dimension-and-interpret-limitations).

### Differences from scikit-learn

The estimators follow scikit-learn's interface, but depart from its conventions in several places:

- **Unlabelled rows.** A numeric label of `-1` marks an unlabelled classification row, and a regression row whose targets are all `NaN` is unlabelled. Both still take part in the fit. With string labels, `"-1"` is an ordinary class. A multi-output row with only some `NaN` targets is rejected, and `score` does not remove unlabelled rows for you; see [Labels, scores, and fitted attributes](estimators.md#labels-scores-and-fitted-attributes).
- **Sample weights.** Every `sample_weight` must be strictly positive. A zero weight raises an error, so drop those rows instead.
- **Class weights.** The parameter is `class_weights`, a mapping with a weight for every labelled class, or a vector in sorted class order. A missing class raises an error rather than defaulting to a weight of one, and there is no `"balanced"` option.
- **Continuation.** `warm_start` takes the named modes `"resume"` and `"fine_tune"`, and `True` raises an error. On an unfitted estimator, either mode fits from scratch and emits a warning. With `"resume"`, `max_iter` includes the iterations already completed; see [Refit and reuse](estimators.md#refit-and-reuse).
- **The estimator's own `cv` and `scoring`.** They select one of the `n_inits` initialisations of a single recipe, not parameter settings, and `fit` passes its `groups` to that splitter; see [Select an initialisation for one recipe](model_selection.md#select-an-initialisation-for-one-recipe).
- **Categories.** The estimator encodes the columns named in `categorical_features` itself. A missing or previously unseen category raises an error, rather than being ignored or treated as missing.
- **Loss history.** `loss_curve_` starts with the loss before the first iteration, so it has `n_iter_ + 1` entries.
- **`score_samples`.** It returns inlier percentiles in [0, 1], relative to the training rows, and requires instance weights learned with a finite positive `epsilon_T`.
- **Failed fits.** If `fit` raises, the estimator keeps its previous fit, or stays unfitted.
