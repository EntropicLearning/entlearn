# Scikit-learn adapter

The optional tabular interface. See the [estimator guide](../guide/estimators.md)
for shared staging and named parameters, and the
[regression and reconstruction guide](../guide/estimators.md#regression-and-reconstruction) for target shapes,
missing-target rules and original-space reconstruction.

[Refit and reuse](../guide/estimators.md#refit-and-reuse) describes the `warm_start` modes that
continue a fit, and how `clone` copies an estimator's parameters without its fitted network.
[Save the entire estimator](../guide/estimators.md#saving-an-estimator) shows how pickle and
joblib keep a fitted estimator, including its ability to resume.

::: entlearn.scikit_adapter.EONClassifier

::: entlearn.scikit_adapter.EONRegressor

Both estimators share one `fit` method:

::: entlearn.scikit_adapter.EONClassifier.fit
    options:
      heading: "EONClassifier.fit and EONRegressor.fit"
      toc_label: "fit"

Both estimators record their fitted column layout as `feature_layout_`, listed with the
other [fitted attributes](../guide/estimators.md#labels-scores-and-fitted-attributes):

::: entlearn.scikit_adapter.FeatureLayout

[Select with validation folds](../guide/model_selection.md#select-with-validation-folds) shows how
`common_train_rows` keeps validation rows out of the initialisation.

::: entlearn.scikit_adapter.common_train_rows

Both estimators inherit the same fitted queries:

::: entlearn.scikit_adapter.EONClassifier.reconstruct

::: entlearn.scikit_adapter.TabularReconstruction

::: entlearn.scikit_adapter.EONClassifier.recover_instance_weights

::: entlearn.scikit_adapter.EONClassifier.score_samples

::: entlearn.scikit_adapter.EONClassifier.effective_dimensions

::: entlearn.scikit_adapter.EONClassifier.feature_importances_

::: entlearn.scikit_adapter.EONClassifier.active_features

::: entlearn.scikit_adapter.EONClassifier.count_parameters

These methods are inherited by `EONRegressor`. Estimator methods own tabular conversion
and original-column mapping; the [state-free reporting helpers](helpers.md) accept only
a fitted Network.
