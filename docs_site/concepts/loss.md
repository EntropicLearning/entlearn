# The objective

EON fits its representations by minimising the total loss.
The input term measures feature-space error, coupling terms relate adjacent clusterings, and the output term relates the final clustering to targets.
Entropy terms control how concentrated affiliations and learned weights become.

This page uses the quantities introduced in [Representing observations](representation.md). Start with an input block and a head, so $N=0$. Hidden blocks add coupling terms to the same objective. [Learning the model](learning.md) gives the updates that minimise it.

<span id="the-terms"></span>

## Input term

The input term combines each observation's cluster errors using its affiliations and instance weight:

$$
L_{\mathrm{input}}=\sum_{t=1}^{T}W_T[t]\sum_{k=1}^{K_0}\Gamma_{tk}^{(0)}d_{tk}.
$$

A hard affiliation selects one cluster's error.
A soft affiliation averages errors across its clusters.
The errors come from the [standard geometry](representation.md#the-discretisation-error) or the [manifold geometry](representation.md#manifold-geometry).

## Coupling terms

Each connection into hidden block $n=1,\dots,N$ contributes

$$
L_{\mathrm{coupling}}^{(n)}=-\delta_n\sum_{t=1}^{T}w_t
\sum_{k=1}^{K_n}\sum_{k'=1}^{K_{n-1}}
\Gamma_{tk}^{(n)}\Gamma_{tk'}^{(n-1)}\log\Theta^{(n)}[k,k'].
$$

A source–target cluster pair has cost $-\log\Theta^{(n)}[k,k']$, weighted by the observation's affiliations to both clusters, larger transition entries give lower costs.
The term is non-negative and linear in either affiliation matrix when the other is fixed. The connection's `delta` sets $\delta_n$.

??? note "Transition prior"
    When `theta_alpha` exceeds one, the coupling term also includes

    $$
    -\delta_n a_n\sum_{k,k'}\log\Theta^{(n)}[k,k'],
    \qquad a_n=\frac{\theta_\alpha-1}{K_n^{\mathrm{initial}}K_{n-1}^{\mathrm{initial}}}.
    $$

    The denominator uses cluster counts at the start of the fit, so pruning leaves the per-entry coefficient unchanged. This prior corresponds to the [pseudocount in the transition update](learning.md#transition-updates). The default `theta_alpha=1` gives $a_n=0$.

## Output term

Only labelled observations contribute to the supervised term.
Classification uses weighted target entries $Q_{tm}$, defined in [Observation weighting](#observation-weighting):

$$
L_{\mathrm{out}}^{\mathrm{cls}}=-\delta_{N+1}\sum_{t,m,k}
Q_{tm}\Gamma_{tk}^{(N)}\log\Theta_{\mathrm{out}}[m,k].
$$

The labels determine the cost of each final cluster, and the affiliations average those costs.
This has the coupling term's cross-entropy form with fixed targets replacing one affiliation matrix.

Regression uses observation weights $\omega_t$ and output-weighted squared residuals:

$$
L_{\mathrm{out}}^{\mathrm{reg}}=\delta_{N+1}\sum_t\omega_t\sum_k\Gamma_{tk}^{(N)}r_t[k],
\qquad r_t[k]=\sum_m W_M[m](Y_{tm}-C_y[m,k])^2.
$$

The training term averages residuals against individual output centroids.
The [regression prediction](prediction.md#regression) averages the centroids themselves.
Unless learned or supplied, $W_M[m]=1/M$.

With no hidden blocks, $\Gamma^{(N)}=\Gamma^{(0)}$ and $\delta_1$ scales the output term.

<span id="three-weightings"></span>

## Observation weighting

Three weightings act on different parts of the objective:

| Weighting | Role | Normalisation |
| --- | --- | --- |
| $W_T$ | Input term | Sums to one, starts at $w$, learned only with finite `epsilon_T` |
| $w$ | Coupling terms and affiliation entropy | Supplied sample weights normalised to sum to one, uniform $1/T$ by default |
| $Q$ for classification, $\omega$ for regression | Output term | Total weight is the labelled share $f$ of the sample weights |

Let $\mathcal L$ be the labelled observations and $f=\sum_{t\in\mathcal L}w_t$.

??? note "Class and task weighting"

    For class weights $c_m$, classification uses

    $$
    Q_{tm}=\begin{cases}
    \displaystyle f\frac{w_t\Pi_{tm}c_m}{\sum_{s\in\mathcal L}\sum_j w_s\Pi_{sj}c_j},&t\in\mathcal L,\\
    0,&t\notin\mathcal L.
    \end{cases}
    $$

    For regression task weights $q_t$, the equivalent observation weights are

    $$
    \omega_t=\begin{cases}
    \displaystyle f\frac{w_tq_t}{\sum_{s\in\mathcal L}w_sq_s},&t\in\mathcal L,\\
    0,&t\notin\mathcal L.
    \end{cases}
    $$

Class weights apply to each component of a soft label.
Without class weights, $c_m=1$ and $Q_{tm}=w_t\Pi_{tm}$ on labelled rows.
Expressions written as $\omega_t\Pi_{tm}$ in the introductory formulation are replaced by $Q_{tm}$ here to make that component weighting explicit.

Without task weights, $q_t=1$ and $\omega_t=w_t$ on labelled rows.
Both normalisations preserve the total output weight $f$.
Supervision has full weight when all observations are labelled and a reduced coefficient when some labels are missing.
Unlabelled observations still contribute to the input and coupling terms.
A positive-weight labelled mass is required.

These normalisations prevent the data terms from scaling directly with the number of observations. Strengths and temperatures require no adjustment solely to compensate for a change in sample count.

## Entropy rewards

Affiliations and learned weight vectors receive entropy rewards (negative entropy penalty).
For $H(p)=-\sum_i p_i\log p_i$, their contributions are

$$
-\sum_{n=0}^{N}\varepsilon_n\sum_t w_t H(\gamma_t^{(n)}),
\qquad -\varepsilon_DH(W_D),\qquad -\varepsilon_TH(W_T),\qquad -\varepsilon_MH(W_M).
$$

Affiliation entropy is weighted per observation.
With uniform sample weights its contribution for block $n$ is $-\varepsilon_n T^{-1}\sum_tH(\gamma_t^{(n)})$.
Each weight vector is one distribution, so its entropy reward has no per-observation weighting.

A higher temperature increases the relative reward for spreading a distribution.
[Learning the model](learning.md#temperatures-and-effective-dimension) describes hard, soft and fixed regimes.

!!! note "Missing labels and output-weight temperature"
    When labels are missing, the regression data term scales with $f$, while the output-weight entropy coefficient stays at $\varepsilon_M$. Relative to that data term, its temperature is $\varepsilon_M/f$. For comparable labelled residual costs, reducing $f$ therefore moves learned output weights towards uniform.

Entropy terms for fixed weights are omitted.
Feature-weight entropy applies only to standard inputs, and output-weight entropy only to regression heads.

## The assembled loss

The complete objective is

$$
\begin{aligned}
L={}&L_{\mathrm{input}}+\sum_{n=1}^{N}L_{\mathrm{coupling}}^{(n)}+L_{\mathrm{out}}\\
&-\sum_{n=0}^{N}\varepsilon_n\sum_t w_tH(\gamma_t^{(n)})
-\varepsilon_DH(W_D)-\varepsilon_TH(W_T)-\varepsilon_MH(W_M).
\end{aligned}
$$

Here $L_{\mathrm{out}}$ is the classification or regression term.
Coupling terms include any active prior.
Weight entropies appear only for learned weights, and temperatures treated as zero contribute no entropy term. For $N=0$, the coupling sum is empty.

## Logarithms and zeros

Logarithms are evaluated as $\log(\max(x,\epsilon))$, where $\epsilon=\texttt{torch.finfo(dtype).eps}$ is dtype machine precision.
This keeps costs finite for zero probabilities while preserving the stored probabilities.
Assignment costs and loss evaluation use the same flooring, and entropy uses the convention $0\log0=0$.

Softmax subtracts the largest logit before exponentiation. The common shift cancels in normalisation and prevents overflow. A slice with a non-finite normaliser falls back to a uniform distribution.

Continue with [Learning the model](learning.md) for the minimisers of this objective.
