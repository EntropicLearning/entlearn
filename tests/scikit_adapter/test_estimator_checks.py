"""Native sklearn checks and explicit conditional-query contracts at default precision."""

from functools import partial

import numpy as np
import pytest
from sklearn.utils.estimator_checks import estimator_checks_generator

from entlearn import ConvergenceWarning
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import fitted_protocol_estimator, protocol_estimator

# These checks require contracts this estimator does not offer. Keep their
# applicability visible rather than marking numerical failures as expected.
_INAPPLICABLE = {
    "check_sample_weight_equivalence_on_dense_data": "Entropy and random initialisation depend on row count, not only weighted mass.",
    "check_estimators_pickle": "Fitted Network serialisation is not available; parameter-only cloning is tested.",
    "check_classifiers_classes": "The check uses -1 as a class; numeric -1 deliberately denotes unlabelled rows.",
    "check_array_api_input": "This is a NumPy/pandas adapter, not an Array API adapter.",
    "check_methods_sample_order_invariance": "Calls score_samples without instance-weight recovery. See TestDefaultQueryContracts; recovery-enabled invariance is covered in test_queries.py.",
    "check_methods_subset_invariance": "Calls score_samples without instance-weight recovery. See TestDefaultQueryContracts; recovery-enabled invariance is covered in test_queries.py.",
}
_EXPECTED_FAILURES = {
    "check_classifiers_one_label_sample_weights": "zero sample weights are rejected",
}

_close = partial(np.testing.assert_allclose, atol=1e-7, rtol=1e-7)
# Class labels must agree exactly; continuous predictions agree to rounding.
_PREDICTIONS = {
    EONClassifier: {"predict": np.testing.assert_array_equal, "predict_proba": _close},
    EONRegressor: {"predict": _close},
}


def _checks():
    # Upstream generators do not obey the unit-range feature convention. Exercise
    # the default float64 estimator here; precision-specific tests use scaled data.
    for estimator_type in (EONClassifier, EONRegressor):
        estimator = protocol_estimator(estimator_type)
        for instance, check in estimator_checks_generator(estimator):
            name = check.func.__name__
            if name in _INAPPLICABLE:
                continue
            reason = _EXPECTED_FAILURES.get(name)
            marks = () if reason is None else pytest.mark.xfail(reason=reason, strict=True)
            yield pytest.param(
                instance, check, id=f"{type(estimator).__name__}-{name}", marks=marks
            )


@pytest.mark.filterwarnings(f"ignore::{ConvergenceWarning.__module__}.ConvergenceWarning")
class TestEstimatorProtocol:
    @pytest.mark.parametrize(("estimator", "check"), list(_checks()))
    def test_sklearn_protocol(self, estimator, check):
        check(estimator)


class TestDefaultQueryContracts:
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_predictions_preserve_query_order(self, estimator_type):
        model, X = fitted_protocol_estimator(estimator_type)
        for name, assert_agree in _PREDICTIONS[estimator_type].items():
            query = getattr(model, name)
            assert_agree(query(X[::-1])[::-1], query(X))

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_single_pass_predictions_agree_across_batches(self, estimator_type):
        model, X = fitted_protocol_estimator(estimator_type)
        for name, assert_agree in _PREDICTIONS[estimator_type].items():
            query = getattr(model, name)
            assert_agree(np.concatenate([query(row[None, :]) for row in X]), query(X))

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_scoring_without_instance_weight_recovery_raises(self, estimator_type):
        model, X = fitted_protocol_estimator(estimator_type)
        with pytest.raises(ValueError, match="instance-weight recovery"):
            model.score_samples(X)
