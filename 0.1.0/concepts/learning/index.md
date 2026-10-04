# Learning the model

Fitting uses block-coordinate descent on [the objective function](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/index.md). Each update sets one group of parameters to its minimiser while the other groups remain fixed. Affiliations and weights solve linear costs with entropy rewards, transitions normalise weighted counts, and centroids solve weighted least-squares problems.

One complete iteration proceeds from the head back to the input:

1. Update the head's transition matrix, or its output centroids followed by any learned output weights.
1. Visit hidden blocks from last to first. Update each block's affiliations, prune empty clusters, then update its incoming transition.
1. Update the input affiliations and prune. Update any learned instance weights. A standard input then updates any learned feature weights and its centroids. A manifold input updates its centroids and plane bases.

Parameters held fixed have no update step. The following sections derive the updates, then describe initialisation, pruning and stopping. Symbols and shapes are listed in [Notation](https://entropiclearning.github.io/entlearn/0.1.0/concepts/notation/index.md).

## Affiliation updates

For observation (t) in block (n), collect all non-entropy terms involving its row into assignment costs (u\_{tk}^{(n)}). The row's objective is

[ \\sum_k\\Gamma\_{tk}^{(n)}u\_{tk}^{(n)}-\\varepsilon_nw_tH(\\gamma_t^{(n)}). ]

For positive row temperature (\\tau_t=\\varepsilon_nw_t), its minimiser is

[ \\Gamma\_{tk}^{(n)}=\\frac{\\exp(-u\_{tk}^{(n)}/\\tau_t)}{\\sum\_{k'}\\exp(-u\_{tk'}^{(n)}/\\tau_t)}. ]

At zero temperature, the update assigns the observation to a minimum-cost cluster, breaking ties by the lowest index. Each row is updated independently using the current neighbouring affiliations. Because the iteration proceeds backwards, outgoing costs read the next block's affiliations from the same iteration.

### Assignment costs

An input cost combines discretisation error with the outgoing connection. A hidden cost combines incoming and outgoing connections. The final block uses the supervised head term as its outgoing cost.

Assignment costs for each block

For an input block followed by a hidden block,

\[ u\_{tk}^{(0)}=W_T[t]d\_{tk}-\\delta_1w_t\\sum\_{k'}\\Gamma\_{tk'}^{(1)}\\log\\Theta^{(1)}[k',k]. \]

For a hidden block with hidden neighbours,

\[ u\_{tk}^{(n)}=-\\delta_nw_t\\sum\_{k'}\\Gamma\_{tk'}^{(n-1)}\\log\\Theta^{(n)}[k,k'] -\\delta\_{n+1}w_t\\sum\_{k'}\\Gamma\_{tk'}^{(n+1)}\\log\\Theta^{(n+1)}[k',k]. \]

The incoming term favours clusters compatible with the source affiliations. The outgoing term favours clusters compatible with the next block's affiliations.

For the final clustering block, replace the outgoing coupling by the head cost:

\[ h_t^{\\mathrm{cls}}[k]=-\\delta\_{N+1}\\sum_mQ\_{tm}\\log\\Theta\_{\\mathrm{out}}[m,k], \\qquad h_t^{\\mathrm{reg}}[k]=\\delta\_{N+1}\\omega_t r_t[k]. \]

(Q) and (\\omega) are the [supervised weights](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/#observation-weighting). Unlabelled observations have zero head cost. With no hidden blocks, the input cost is (W_T[t]d\_{tk}+h_t[k]).

### Affiliation temperatures

Block `epsilon` sets (\\varepsilon_n). The default is zero, leading to hard affiliations. Finite positive values give soft affiliations, while larger values spread membership more broadly. Infinite affiliation temperatures are rejected.

The temperature setting determines the regime

The hard or soft regime is resolved once at the start of fitting, by comparing (\\varepsilon_n) with dtype machine precision. The comparison uses the block setting, even though the row temperature is (\\tau_t=\\varepsilon_nw_t). With uniform weights, (\\tau_t=\\varepsilon_n/T) and the common (1/T) factor in the weighted costs cancels from the exponent. Increasing the number of rows therefore does not change the regime, which is kept during prediction. `network.inspect("affiliation_regimes")` reports `"hard"` or `"soft"` for each block.

Temperature effects depend on the feature scale and connection strengths. The [Model selection guide](https://entropiclearning.github.io/entlearn/0.1.0/guide/model_selection/index.md) describes how to search over them.

## Transition updates

For hidden connection (n), form the weighted co-occurrence counts

[ A\_{kk'}=\\sum_tw_t\\Gamma\_{tk}^{(n)}\\Gamma\_{tk'}^{(n-1)}+a_n. ]

Normalise columns under M coupling and rows under S coupling:

\[ \\Theta^{(n)}[k,k']=\\frac{A\_{kk'}}{\\sum_jA\_{jk'}}\\quad\\text{under M}, \\qquad \\Theta^{(n)}[k,k']=\\frac{A\_{kk'}}{\\sum_jA\_{kj}}\\quad\\text{under S}. \]

The coupling objective separates into one count-weighted cross-entropy per constrained slice. Its minimiser is the normalised count vector. The [prior coefficient](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/#coupling-terms) (a_n) acts as an equal pseudocount in each entry. Positive pseudocounts keep entries positive.

A slice with zero total count becomes uniform: (1/K_n) for an M column or (1/K\_{n-1}) for an S row.

## Centroid updates

For an input cluster, define its weighted mass

\[ m_k=\\sum_tW_T[t]\\Gamma\_{tk}^{(0)}. \]

The continuous centroid is the weighted mean

\[ c_k=\\frac{\\sum_tW_T[t]\\Gamma\_{tk}^{(0)}x_t}{m_k}. \]

Each categorical centroid is the weighted level distribution

\[ C\_{j,k,:}^{\\mathrm{cat}}=\\frac{\\sum_tW_T[t]\\Gamma\_{tk}^{(0)}X\_{j,t,:}^{\\mathrm{cat}}}{m_k}. \]

Feature weights multiply each feature's objective by a constant and leave these means as minimisers. The updates use instance weights from the same iteration.

Weighted mass and empty clusters

When (m_k) is at or below dtype machine precision, continuous centroids become zero and categorical centroids become uniform. This weighted-mass fallback is distinct from [pruning](#pruning), which uses unweighted affiliation mass.

### Manifold bases

After updating manifold centroids, fit each local plane using the leading eigenvectors of its cluster’s weighted covariance.

Covariance and basis update

A manifold input uses the same centroid mean. After updating the centroid, compute

\[ \\Sigma_k=\\frac{1}{m_k}\\sum_tW_T[t]\\Gamma\_{tk}^{(0)}(x_t-c_k)(x_t-c_k)^\\top. \]

(U_k) becomes the leading `subspace_dimension` eigenvectors of this weighted covariance. These directions maximise captured covariance and minimise the weighted squared distance from the plane. A cluster with no mass keeps its previous basis.

## Weight updates

For any learned weight vector (W), collect its linear coefficients in a cost vector (b). Its objective is (\\sum_iW[i]b_i-\\varepsilon H(W)), with minimiser:

\[ W[i]=\\frac{\\exp(-b_i/\\varepsilon)}{\\sum_j\\exp(-b_j/\\varepsilon)}. \]

### Feature weights

For continuous feature (d) and categorical feature (j),

\[ \\begin{aligned} b_d^{(D)}&=\\sum\_{t,k}W_T[t]\\Gamma\_{tk}^{(0)}(X\_{td}-C\_{kd})^2,\\ b\_{D\_{\\mathrm{cont}}+j}^{(D)}&=\\delta\_{\\mathrm{cat}}s_j\\sum\_{t,k}W_T[t]\\Gamma\_{tk}^{(0)} \\mathrm{CE}(X\_{j,t,:}^{\\mathrm{cat}},C\_{j,k,:}^{\\mathrm{cat}}). \\end{aligned} \]

Both feature types share one softmax. Its entries are total within-cluster errors, with the [categorical scaling](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#the-discretisation-error) applied before the update. At infinite `epsilon_D`, feature weights retain their starting values: uniform, a random draw controlled by `W_std`, or supplied weights. A hard update selects a single feature.

### Instance weights

The cost of observation (t) is its affiliation-weighted input error:

[ b_t^{(T)}=\\sum_k\\Gamma\_{tk}^{(0)}d\_{tk}. ]

For a manifold input, use (g_k(x_t)) in place of (d\_{tk}). Observations with larger errors receive less weight in the soft update. At infinite `epsilon_T`, weights stay at the normalised sample weights. With finite temperature these starting values can change, and sample weights continue to act separately on the other terms.

### Output weights

For regression output dimension (m),

\[ b_m^{(M)}=\\delta\_{N+1}\\sum_t\\omega_t\\sum_k\\Gamma\_{tk}^{(N)}(Y\_{tm}-C_y[m,k])^2. \]

The update follows the output-centroid update. At infinite `epsilon_M`, output weights are uniform unless supplied as a `W_M` tuple, which is normalised to sum to one. Supplied output weights require infinite `epsilon_M`.

### Temperatures and effective dimension

Weight temperatures are infinite by default, which holds their values fixed. A finite temperature above dtype machine precision enables the softmax update once per iteration. A temperature at or below machine precision selects the cheapest entry, with ties going to the lowest index. Fixed weights and zero-temperature weights contribute no entropy term to the reported loss.

A temperature is measured in the units of its cost vector. Note that the same value can yield broad weights on one dataset and concentrated weights on another. The normalised effective dimension (\\exp(H(p))/n) summarises concentration for a probability vector with (n) entries. It is one for uniform weights and (1/n) for a single selected entry. The unnormalised effective dimension is (\\exp(H(p))).

The [reporting helper](https://entropiclearning.github.io/entlearn/0.1.0/reference/helpers/#fitted-reporting) `effective_dimensions` reports normalised effective dimensions for affiliations and weights. For cost vectors whose entries follow comparable distributions, a given temperature can yield similar normalised effective dimensions at different vector lengths.

## Head updates

### Classification

Form the labelled co-occurrence counts

[ \\nu\_{mk}=\\sum_tQ\_{tm}\\Gamma\_{tk}^{(N)}. ]

Normalise columns for an M head and rows for an S head, as for hidden transitions. Under M, a cluster with no labelled mass gets a uniform class distribution (1/M). Under S, a class with no labelled mass gets a uniform row (1/K_N), while an unlabelled cluster contributes a column of zeros.

### Regression

For each cluster and output dimension, the weighted target mean minimises the squared residual:

\[ C_y[m,k]=\\frac{\\sum_t\\omega_t\\Gamma\_{tk}^{(N)}Y\_{tm}}{\\sum_t\\omega_t\\Gamma\_{tk}^{(N)}}. \]

(W_M[m]) scales the dimension's objective and leaves this mean as a minimiser. If the cluster has no labelled mass, use the overall labelled target mean (\\sum_t\\omega_tY\_{tm}/\\sum_t\\omega_t).

## Initialisation

Different starting states can lead to different solutions. `n_inits` compares fits from several initialisations and retains the best. The [Initialisation guide](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#initialisation) explains the controls and parameter reuse.

### Input geometry

Input seeding procedure

Feature weights start at (1/D) unless supplied or randomised with positive `W_std`. Randomisation draws independent normal logits with mean zero and standard deviation `W_std`, then applies softmax. Small `W_std` gives near-uniform weights and large values tend to concentrate them.

Centroids start at (K_0) distinct observations selected by weighted k-means++ using both feature types and the starting feature weights. Categorical seeds are initially one-hot. Before the first iteration, categorical centroids are replaced by their weighted-frequency update using the first affiliations and starting instance weights. Continuous seeds retain their values.

### Hidden transitions

Transition seeding procedure

Connections are seeded in order from the input using the fit's random seed. After each transition is seeded, its target affiliations are assigned using the forward prediction rule.

- **M with no more target than source clusters:** group source clusters into (K_n) groups using k-means++ over their feature-space centroids, weighted by cluster masses. Each source column assigns all its mass to its group.
- **M with more target than source clusters:** select (K_n) observations by k-means++ over source affiliations using cross-entropy dissimilarity. Assign observations to their nearest selection, then compute the transition from this hard partition.
- **S:** select observations by the same affiliation procedure and use their source affiliations as transition rows.

An active pseudocount is incorporated into seeded counts before normalisation. A transition supplied from a fitted network retains its values.

### Heads

Classification heads start with the uniform matrix under their selected constraint. Regression output centroids start with the overall weighted mean of labelled targets in every column.

## Pruning

Immediately after each affiliation update, fitting checks cluster (k)'s unweighted mass (\\sum_t\\Gamma\_{tk}^{(n)}) (without applying sample and instance weights). A mass at or below dtype machine precision is empty.

Removal drops the affiliation column and associated input geometry or head-parameter columns. An incoming hidden transition loses target rows, and an outgoing transition loses source columns. Renormalisation depends on the coupling.

## Convergence and stopping

Every coordinate update minimises its part of the same objective, so an iteration can only decrease or preserve the loss up to numerical precision. Initialisation and update order can affect the solution reached because the joint problem need not be convex. Updates and loss evaluation use the same observation weights, strengths and floored logarithms.

`network.diagnostics.loss_history` stores the loss before the first iteration and after each complete iteration. Fitting stops when the relative change is at most `tol`, or after `max_iter` iterations. A rise exceeding (256\\epsilon\\max(1,|L\_{\\mathrm{previous}}|)) emits a `LossIncreaseWarning`, where (\\epsilon) is dtype machine precision.

Continue with [Predicting new observations](https://entropiclearning.github.io/entlearn/0.1.0/concepts/prediction/index.md) for inference with the learned parameters fixed.
