"""Native estimator transport preserves fitted capability and tabular meanings."""

import io
import pickle

import joblib
import numpy as np
import pytest
from conftest import FIXED_BUDGET
from sklearn.base import clone

from entlearn import PredictConfig
from entlearn.scikit_adapter import EONClassifier

from ._fixtures import continuation_case, continuation_frame


@pytest.fixture(params=["pickle", "joblib"])
def round_trip(request):
    def transport(value):
        if request.param == "pickle":
            return pickle.loads(pickle.dumps(value))
        stream = io.BytesIO()
        joblib.dump(value, stream)
        stream.seek(0)
        return joblib.load(stream)

    return transport


@FIXED_BUDGET
class TestEstimatorTransport:
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("members", [False, True])
    def test_transport_keeps_tabular_meanings_and_continues_like_the_source(
        self, mode, members, round_trip
    ):
        model, X, y = continuation_case(
            EONClassifier,
            n_inits=2 if members else 1,
            retain="members" if members else "winner",
        )
        frame = continuation_frame(X)
        model.set_params(categorical_features="from_dtype").fit(frame, y)
        model.set_params(n_inits=1, warm_start=mode, max_iter=6)
        restored = round_trip(model)
        np.testing.assert_array_equal(restored.feature_names_in_, model.feature_names_in_)
        np.testing.assert_array_equal(restored.feature_layout_.categories[0], ["a", "b"])
        np.testing.assert_array_equal(restored.classes_, model.classes_)
        model.fit(frame, y)
        restored.fit(frame, y)
        assert restored.loss_curve_ == model.loss_curve_
        np.testing.assert_array_equal(restored.predict_proba(frame), model.predict_proba(frame))

    @pytest.mark.parametrize("supplied", [False, True])
    def test_transport_preserves_the_original_prediction_policy_request(self, supplied, round_trip):
        model, X, y = continuation_case(
            EONClassifier, predict_config=PredictConfig(epsilon_P=0.4) if supplied else None
        )
        model.fit(X, y)
        restored = round_trip(model.set_params(warm_start="fine_tune"))
        policy = PredictConfig(
            epsilon_P=None if supplied else restored.network_.predict_config.epsilon_P
        )
        restored.set_params(predict_config=policy)
        with pytest.raises(ValueError, match="temperature choice"):
            restored.fit(X, y)
        restored.set_params(predict_config=None).fit(X, y)
        assert restored.network_.can_resume

    def test_an_unfitted_estimator_with_a_supplied_state_fits_like_the_source(self, round_trip):
        source, X, y = continuation_case(EONClassifier)
        source.fit(X, y)
        model = clone(source).set_params(initial_state=source.network_.capture_current_state())
        restored = round_trip(model)
        restored.fit(X, y)
        model.fit(X, y)
        assert restored.loss_curve_ == model.loss_curve_
        model.set_params(warm_start="resume", max_iter=6).fit(X, y)
        restored.set_params(warm_start="resume", max_iter=6).fit(X, y)
        assert restored.loss_curve_ == model.loss_curve_
        np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
