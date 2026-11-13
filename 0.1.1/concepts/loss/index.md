# The objective

EON fits its representations by minimising the total loss. The input term measures feature-space error, coupling terms relate adjacent clusterings, and the output term relates the final clustering to targets. Entropy terms control how concentrated affiliations and learned weights become.

This page uses the quantities introduced in [Representing observations](https://entropiclearning.github.io/entlearn/0.1.1/concepts/representation/index.md). Start with an input block and a head, so (N=0). Hidden blocks add coupling terms to the same objective. [Learning the model](https://entropiclearning.github.io/entlearn/0.1.1/concepts/learning/index.md) gives the updates that minimise it.

## Input term

The input term combines each observation's cluster errors using its affiliations and instance weight:

\[ L\_{\\mathrm{input}}=\\sum\_{t=1}^{T}W_T[t]\\sum\_{k=1}^{K_0}\\Gamma\_{tk}^{(0)}d\_{tk}. \]

A hard affiliation selects one cluster's error. A soft affiliation averages errors across its clusters. The errors come from the [standard geometry](https://entropiclearning.github.io/entlearn/0.1.1/concepts/representation/#the-discretisation-error) or the [manifold geometry](https://entropiclearning.github.io/entlearn/0.1.1/concepts/representation/#manifold-geometry).

## Coupling terms

Each connection into hidden block (n=1,\\dots,N) contributes

\[ L\_{\\mathrm{coupling}}^{(n)}=-\\delta_n\\sum\_{t=1}^{T}w_t \\sum\_{k=1}^{K_n}\\sum\_{k'=1}^{K\_{n-1}} \\Gamma\_{tk}^{(n)}\\Gamma\_{tk'}^{(n-1)}\\log\\Theta^{(n)}[k,k']. \]

A source–target cluster pair has cost (-\\log\\Theta^{(n)}[k,k']), weighted by the observation's affiliations to both clusters. Larger transition entries give lower costs. The term is non-negative and linear in either affiliation matrix when the other is fixed. The connection's `delta` sets (\\delta_n).

Transition prior

When `theta_alpha` exceeds one, the coupling term also includes

\[ -\\delta_n a_n\\sum\_{k,k'}\\log\\Theta^{(n)}[k,k'], \\qquad a_n=\\frac{\\theta\_\\alpha-1}{K_n^{\\mathrm{initial}}K\_{n-1}^{\\mathrm{initial}}}. \]

The denominator uses the cluster counts that the blocks are created with, so pruning leaves the per-entry coefficient unchanged. This prior corresponds to the [pseudocount in the transition update](https://entropiclearning.github.io/entlearn/0.1.1/concepts/learning/#transition-updates). The default `theta_alpha=1` gives (a_n=0).

## Output term

Only labelled observations contribute to the supervised term. Classification uses weighted target entries (Q\_{tm}), defined in [Observation weighting](#observation-weighting):

\[ L\_{\\mathrm{out}}^{\\mathrm{cls}}=-\\delta\_{N+1}\\sum\_{t,m,k} Q\_{tm}\\Gamma\_{tk}^{(N)}\\log\\Theta\_{\\mathrm{out}}[m,k]. \]

The labels determine the cost of each final cluster, and the affiliations average those costs. This has the coupling term's cross-entropy form with fixed targets replacing one affiliation matrix.

Regression uses observation weights (\\omega_t) and output-weighted squared residuals:

\[ L\_{\\mathrm{out}}^{\\mathrm{reg}}=\\delta\_{N+1}\\sum_t\\omega_t\\sum_k\\Gamma\_{tk}^{(N)}r_t[k], \\qquad r_t[k]=\\sum_m W_M[m](Y_%7Btm%7D-C_y%5Bm,k%5D)^2. \]

The training term averages residuals against individual output centroids. The [regression prediction](https://entropiclearning.github.io/entlearn/0.1.1/concepts/prediction/#regression) averages the centroids themselves. Unless learned or supplied, (W_M[m]=1/M).

With no hidden blocks, (\\Gamma^{(N)}=\\Gamma^{(0)}) and (\\delta_1) scales the output term.

## Observation weighting

Three weightings act on different parts of the objective:

| Weighting                                        | Role                                                                              | Normalisation                                                              |
| ------------------------------------------------ | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| (W_T)                                            | Input term                                                                        | Sums to one, starts at (w), learned only with finite `epsilon_T`           |
| (w)                                              | Coupling terms, affiliation entropy, and the prior of the instance-weight penalty | Supplied sample weights normalised to sum to one, uniform (1/T) by default |
| (Q) for classification, (\\omega) for regression | Output term                                                                       | Total weight is the labelled share (f) of the sample weights               |

Let (\\mathcal L) be the labelled observations and (f=\\sum\_{t\\in\\mathcal L}w_t).

Class and task weighting

For class weights (c_m), classification uses

[ Q\_{tm}=\\begin{cases} \\displaystyle f\\frac{w_t\\Pi\_{tm}c_m}{\\sum\_{s\\in\\mathcal L}\\sum_j w_s\\Pi\_{sj}c_j},&t\\in\\mathcal L,\\ 0,&t\\notin\\mathcal L. \\end{cases} ]

For regression task weights (q_t), the equivalent observation weights are

[ \\omega_t=\\begin{cases} \\displaystyle f\\frac{w_tq_t}{\\sum\_{s\\in\\mathcal L}w_sq_s},&t\\in\\mathcal L,\\ 0,&t\\notin\\mathcal L. \\end{cases} ]

Class weights apply to each component of a soft label. Without class weights, (c_m=1) and (Q\_{tm}=w_t\\Pi\_{tm}) on labelled rows. Expressions written as (\\omega_t\\Pi\_{tm}) in the introductory formulation are replaced by (Q\_{tm}) here to make that component weighting explicit.

Without task weights, (q_t=1) and (\\omega_t=w_t) on labelled rows. Both normalisations preserve the total output weight (f). Supervision has full weight when all observations are labelled and a reduced coefficient when some labels are missing. Unlabelled observations still contribute to the input and coupling terms. A positive-weight labelled mass is required.

These normalisations prevent the data terms from scaling directly with the number of observations. Strengths and temperatures require no adjustment solely to compensate for a change in sample count.

## Entropy penalties

Affiliations and learned weight vectors receive negative-entropy penalties, which are smallest for spread-out distributions. For (H(p)=-\\sum_i p_i\\log p_i), their contributions are

\[ -\\sum\_{n=0}^{N}\\varepsilon_n\\sum_t w_t H(\\gamma_t^{(n)}), \\qquad -\\varepsilon_DH(W_D),\\qquad -\\varepsilon_T\\Big\[H(W_T)+\\sum_tW_T[t]\\log(Tw_t)\\Big\],\\qquad -\\varepsilon_MH(W_M). \]

Affiliation entropy is weighted per observation. With uniform sample weights its contribution for block (n) is (-\\varepsilon_n T^{-1}\\sum_tH(\\gamma_t^{(n)})). Each weight vector is one distribution, so its entropy penalty has no per-observation weighting.

The instance-weight penalty measures (W_T) against the sample weights. In the bracket above, (\\sum_tW_T[t]\\log(Tw_t)) is the expectation of (\\log(Tw_t)) under (W_T), so the penalty is lower when (W_T) puts its mass on observations with large sample weights. The penalty equals

\[ \\varepsilon_T\\sum_tW_T[t]\\log\\frac{W_T[t]}{w_t}-\\varepsilon_T\\log T, \]

which is (\\varepsilon_T) times the relative entropy of (W_T) from (w), minus the constant (\\varepsilon_T\\log T). It is therefore smallest when (W_T=w). With uniform sample weights (\\log(Tw_t)=0), and the penalty is (-\\varepsilon_TH(W_T)).

A higher temperature gives these penalties more weight against the data terms, which pulls each distribution towards uniform, or (W_T) towards (w). [Learning the model](https://entropiclearning.github.io/entlearn/0.1.1/concepts/learning/#temperatures-and-effective-dimension) describes hard, soft and fixed regimes.

Missing labels and output-weight temperature

When labels are missing, the regression data term scales with (f), while the output-weight entropy coefficient stays at (\\varepsilon_M). Relative to that data term, its temperature is (\\varepsilon_M/f). For comparable labelled residual costs, reducing (f) therefore moves learned output weights towards uniform.

Entropy terms for fixed weights are omitted. Feature-weight entropy applies only to standard inputs, and output-weight entropy only to regression heads.

## The assembled loss

The complete objective is

\[ \\begin{aligned} L={}&L\_{\\mathrm{input}}+\\sum\_{n=1}^{N}L\_{\\mathrm{coupling}}^{(n)}+L\_{\\mathrm{out}}\\ &-\\sum\_{n=0}^{N}\\varepsilon_n\\sum_t w_tH(\\gamma_t^{(n)}) -\\varepsilon_DH(W_D)-\\varepsilon_T\\Big\[H(W_T)+\\sum_tW_T[t]\\log(Tw_t)\\Big\]-\\varepsilon_MH(W_M). \\end{aligned} \]

Here (L\_{\\mathrm{out}}) is the classification or regression term. Coupling terms include any active prior.

Continue with [Learning the model](https://entropiclearning.github.io/entlearn/0.1.1/concepts/learning/index.md) for the minimisers of this objective.
