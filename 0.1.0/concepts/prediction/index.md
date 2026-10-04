# Predicting new observations

Prediction computes affiliations for new observations with the learned geometry, transitions and head parameters held fixed. A single forward pass is the default. Iterative prediction additionally updates affiliations using outgoing connections and a prediction coordinate instead of the unknown target.

This page follows [Learning the model](https://entropiclearning.github.io/entlearn/0.1.0/concepts/learning/index.md), and describes both prediction modes, the head read-outs, calibration and instance-weight recovery. [Predict new observations](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#predict-new-observations) shows the corresponding settings in code.

## The single forward pass

By default, a prediction takes one pass from the input block to the head. The input block compares each new observation with the fitted [geometry](https://entropiclearning.github.io/entlearn/0.1.0/concepts/representation/#standard-input-geometry), and its affiliations are a softmax of the negative discretisation errors:

[ \\Gamma\_{tk}^{(0)}=\\frac{\\exp(-d\_{tk}/\\varepsilon_0)}{\\sum\_{k'}\\exp(-d\_{tk'}/\\varepsilon_0)}. ]

Each hidden block then takes its affiliations from the incoming transition alone:

\[ \\Gamma\_{tk}^{(n)}\\propto\\exp\\Big(\\frac{\\delta_n}{\\varepsilon_n}\\sum\_{k'=1}^{K\_{n-1}}\\Gamma\_{tk'}^{(n-1)}\\log\\Theta^{(n)}[k,k']\\Big), \]

normalised so that each row sums to one. Each block keeps the hard or soft regime it had during fitting, whatever the size of the query batch. Every row is computed on its own, so a single-pass prediction does not depend on the other observations in the batch.

During fitting, the [assignment cost](https://entropiclearning.github.io/entlearn/0.1.0/concepts/learning/#assignment-costs) of each block also contains the coupling term of its outgoing connection, which depends on the affiliations of the next block. For the final block, that term is the head's term, which depends on the label. The forward pass computes each block before the next block exists and before a target is available, so it omits these outgoing terms. [Iterative prediction](#iterative-prediction) includes them, with a prediction coordinate in place of the unknown label.

Weight cancellation in the forward pass

During fitting, sample weights multiply both hidden coupling costs and affiliation temperatures, so they cancel in the forward exponent. The input block's corresponding exponent, after omitting its outgoing term, is (-W_T[t]d\_{tk}/(\\varepsilon_0w_t)). With fixed instance weights (W_T=w), it reduces to (-d\_{tk}/\\varepsilon_0). With learned instance weights, training errors are scaled by (W_T[t]/w_t), while the forward pass scores new observations with that ratio set to one.

Hidden propagation uses the same rule under M and S coupling. Each block applies its own softmax or hard assignment to the transition-derived cost. A chain therefore composes these assignment operations rather than multiplying transition matrices.

## Reading out a prediction

A classification head predicts a matrix (\\hat\\Pi) with one row per observation: row (\\hat\\pi_t) is observation (t)'s predicted distribution over the classes. A regression head predicts a matrix (\\hat Y) in the same way: row (\\hat y_t) is observation (t)'s predicted output. Both have (M) columns.

### Regression

A regression head predicts the affiliation-weighted mixture of its output centroids:

\[ \\hat Y\_{tm}=\\sum\_{k}\\Gamma\_{tk}^{(N)}C_y[m,k]. \]

The predicted vector lies in the convex hull of the output centroids. Output weights (W_M) do not enter this average.

### Classification: the geometric read-out

The default classification read-out works with the **logits** of observation (t) for each class (m),

\[ z\_{tm}=\\sum\_{k}\\Gamma\_{tk}^{(N)}\\log\\Theta\_{\\mathrm{out}}[m,k], \]

which are the same quantities that the [classification term](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/#output-term) of the loss uses. The predicted probability of class (m) is a softmax of the logits, scaled by an inverse temperature (\\beta):

[ \\hat\\Pi\_{tm}=\\frac{\\exp(\\beta z\_{tm})}{\\sum\_{m'}\\exp(\\beta z\_{tm'})}, \\qquad \\beta=\\frac{\\delta\_{N+1}}{\\varepsilon_P}. ]

Here, (\\delta\_{N+1}) is the strength of the connection into the head, and (\\varepsilon_P) is the output temperature `PredictConfig.epsilon_P`.

At (\\beta=1), the probabilities are proportional to (\\prod_k\\Theta\_{\\mathrm{out}}[m,k]^{\\Gamma\_{tk}^{(N)}}), a weighted **geometric mean** of the head's entries for class (m). A larger (\\beta) sharpens the probabilities, and a smaller one flattens them.

### Classification: the arithmetic read-out

With `PredictConfig(output_mode="arithmetic")`, an M head instead takes the affiliation-weighted **arithmetic mean** of its columns:

\[ \\hat\\Pi\_{tm}=\\sum\_{k}\\Gamma\_{tk}^{(N)}\\Theta\_{\\mathrm{out}}[m,k]. \]

Each column is a class distribution, and the affiliations are its mixture weights. The arithmetic read-out has no temperature. An S head rejects it, because its columns are not distributions over the classes.

## The calibrated output temperature

With the geometric read-out, fitting chooses (\\varepsilon_P) automatically unless you supply it. The chosen value is stored in `network.predict_config.epsilon_P`, and every later prediction reuses it. Calibration runs automatically during fitting.

Replacing the prediction configuration

If you pass a new `PredictConfig` at prediction time and leave `epsilon_P` out, the read-out falls back to (\\varepsilon_P=\\delta\_{N+1}), which is (\\beta=1). [Classification calibration](https://entropiclearning.github.io/entlearn/0.1.0/guide/hyperparameters/#classification-calibration) shows how to keep the fitted value instead.

Calibration minimises the weighted cross-entropy (\\ell(\\beta)) of the single-pass predictions (\\hat\\Pi(\\beta)) on the training rows:

[ \\ell(\\beta)=-\\frac{1}{\\sum\_{t,m}Q\_{tm}}\\sum\_{t=1}^{T}\\sum\_{m=1}^{M}Q\_{tm}\\log\\hat\\Pi\_{tm}(\\beta). ]

The [weighted target entries](https://entropiclearning.github.io/entlearn/0.1.0/concepts/loss/#observation-weighting) (Q\_{tm}) account for sample weights and class weights. Only labelled rows with positive weight contribute. The logits (z\_{tm}) come from a single forward pass through the fitted model, and they are computed once. (\\ell) is convex in (\\beta), so a zero of its derivative is a minimum.

Calibration search and candidate selection

The algorithm runs once per fitted model:

1. **Find the stationary point.** Search (\\log\\beta) between (\\log 10^{-6}) and (\\log\\beta\_{\\max}) for a zero of the derivative, where (\\beta\_{\\max}) is (10^{6}), or smaller if a larger (\\beta) would take (\\varepsilon_P) to the machine precision. The search uses Chandrupatla's method, which combines inverse quadratic interpolation with bisection, and it stops when the bracket in (\\log\\beta) is narrower than the square root of the machine precision. If the derivative has no sign change within the bracket, the search returns the bound at which (\\ell) is smaller.
1. **Collect the candidates.** They are the uncalibrated value (\\varepsilon_P=\\delta\_{N+1}), the value (\\delta\_{N+1}/\\beta) from step 1 and, when (\\delta\_{N+1}/10^{6}) is at or below the machine precision, the hard read-out it implies.
1. **Score them with the read-out.** A candidate whose read-out gives zero probability to a class with target mass scores infinity. A hard read-out does this whenever a row's target mass is not all on its predicted class.
1. **Keep the lowest score.** Ties go to the uncalibrated value. If every candidate scores infinity, fitting raises an error.

The chosen temperature has training cross-entropy at most that of (\\beta=1). This comparison applies to the training single-pass predictions. Performance on new observations and iterative predictions can differ.

Each fitted model calibrates on its own training rows. During model selection with validation pairs, each fold model calibrates on its training partition, and the final refit calibrates again on all rows. Each calibration belongs to its own fitted model. Continuing or fine-tuning a fit recalibrates a derived temperature and keeps a supplied one. Calibration applies to geometric classification read-outs.

## Instance weights for new observations

When the instance weights are learned at a positive temperature, fitting also keeps the normaliser of their softmax:

[ \\log Z\_{\\mathrm{train}}=\\log\\sum\_{t=1}^{T}\\exp(-b_t^{(T)}/\\varepsilon_T), ]

recomputed from the final geometry once fitting stops. A new observation then receives the weight its cost would have had in the training softmax, capped at one:

\[ W_T'[t]=\\min\\big(1,\\exp(-b_t^{(T)}/\\varepsilon_T-\\log Z\_{\\mathrm{train}})\\big). \]

Here (b_t^{(T)}) uses the new observation's own affiliations and discretisation errors. Each recovered weight lies between zero and one. Recovery uses the training normaliser, so the query weights are independent of other query rows and their sum can differ from one. `network.score_samples` ranks them against the weights recovered on the training rows, as described in [Reconstruction and inlier percentiles](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#reconstruction-and-inlier-percentiles). Recovery requires soft learned instance weights, as fixed and hard instance weights keep no normaliser.

## Iterative prediction

The single pass solves each block once, without the coupling term of its outgoing connection. Iterative prediction, selected with `PredictConfig(predict_mode="iterative")`, instead keeps minimising the fitted loss over the quantities that belong to the new observations. Every learned parameter stays fixed.

The unknown label is replaced by a **prediction coordinate**: a class distribution (\\hat\\pi_t) for classification, or a value (\\hat y_t) for regression. The iterations minimise a query objective (J), using the fitted costs and prediction coordinate. The input term also accounts for recovered instance weights when these are available.

The iterative prediction objective

The objective (J) is the loss for the batch of (T') new observations, with that coordinate in place of the label and every row weighted (1/T'):

\[ \\begin{aligned} J={}& J\_{\\mathrm{input}} -\\sum\_{n=0}^{N}\\frac{\\varepsilon_n}{T'}\\sum\_{t=1}^{T'}H\\big(\\Gamma\_{t,:}^{(n)}\\big) -\\sum\_{n=1}^{N}\\frac{\\delta_n}{T'}\\sum\_{t=1}^{T'}\\sum\_{k,k'}\\Gamma\_{tk}^{(n)}\\Gamma\_{tk'}^{(n-1)}\\log\\Theta^{(n)}[k,k'] +J\_{\\mathrm{out}}, \\ J\_{\\mathrm{out}}={}&-\\frac{\\delta\_{N+1}}{T'}\\sum\_{t,m,k}\\hat\\Pi\_{tm}\\Gamma\_{tk}^{(N)}\\log\\Theta\_{\\mathrm{out}}[m,k] -\\frac{\\varepsilon_P}{T'}\\sum\_{t=1}^{T'}H(\\hat\\pi_t) &&\\text{for classification,} \\ J\_{\\mathrm{out}}={}&\\frac{\\delta\_{N+1}}{T'}\\sum\_{t,k}\\Gamma\_{tk}^{(N)}\\sum\_{m}W_M[m]\\big(\\hat Y\_{tm}-C_y[m,k]\\big)^2 &&\\text{for regression.} \\end{aligned} \]

The prediction-coordinate entropy favours a spread-out class distribution, with strength set by the fitted or supplied output temperature (\\varepsilon_P).

The input term depends on whether the model learned its instance weights. If it did, the new observations receive the [recovered instance weights](#instance-weights-for-new-observations) (W_T'), and

\[ J\_{\\mathrm{input}}=\\frac{T}{T'}\\sum\_{t=1}^{T'}\\Big(W_T'[t]b_t^{(T)}+\\varepsilon_T W_T'[t]\\big(\\log W_T'[t]+\\log Z\_{\\mathrm{train}}-1\\big)\\Big), \]

where (b_t^{(T)}) is the [instance-weight cost](https://entropiclearning.github.io/entlearn/0.1.0/concepts/learning/#instance-weights) of new observation (t), computed from its own affiliations and discretisation errors. Each (W_T'[t]) lies between zero and one, and the recovery formula is the exact minimiser of this term. Otherwise, (J\_{\\mathrm{input}}=\\frac{1}{T'}\\sum_t b_t^{(T)}).

One iteration updates the coordinates in the same order as fitting:

1. **Prediction coordinate.** For classification, (\\hat\\pi_t) becomes the geometric read-out at (\\varepsilon_P). For regression, (\\hat y_t) becomes the mixture of the output centroids.
1. **Hidden blocks**, from the last to the first. Each block's assignment cost now also includes the coupling term of its outgoing connection. For the final block, this is the head's term, evaluated at the prediction coordinate.
1. **Input block.** The input affiliations update, and then the recovered instance weights, if any.

Each update is the exact minimiser of (J) over its coordinates, so (J) never increases. The iterations stop when the relative change in (J) is at most `tol`, or after `max_iter` iterations.

Batch-dependent stopping

Because (J) averages over the batch, the stopping iteration, and therefore the prediction, can depend on which observations are predicted together.

The first iteration starts from the single-pass prediction, unless `predict_init` supplies another start. With the arithmetic read-out, the iterations still use the geometric prediction coordinate, at (\\varepsilon_P=\\delta\_{N+1}), and the returned prediction is the arithmetic read-out of the final affiliations.
