# Optional helpers

These helpers keep no state. `feature_weights` chooses starting feature weights from
statistics of the labelled rows, `delta_cat_guess` computes the default
`delta_cat` from the training data, and `reporting` summarises a fitted `Network` through its
public methods. The statistical helpers load SciPy
or scikit-learn only when called; the `sklearn` extra installs both.

## Initialisation

When the features far outnumber the labelled rows, pass `null_permutations` (three is
usually enough with hundreds of features) and `random_state`. The helper then subtracts a
permutation-null quantile from each score, so the many unrelated features no longer take
most of the weight. It is off by default.

::: entlearn.helpers.feature_weights

## Categorical scale

::: entlearn.helpers.delta_cat_guess

## Fitted reporting

::: entlearn.helpers.reporting
