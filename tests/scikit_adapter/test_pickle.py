"""Native estimator transport preserves fitted capability and tabular meanings."""

import io
import pickle

import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone

from entlearn import Coupling, Hidden, Network, PredictConfig
from entlearn.scikit_adapter import EONClassifier, EONRegressor

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


class TestEstimatorTransport:
    @pytest.mark.parametrize(
        "estimator_type,policy",
        [
            (EONClassifier, None),
            (EONClassifier, PredictConfig(epsilon_P=0.4)),
            (EONRegressor, None),
        ],
    )
    @pytest.mark.parametrize("members", [False, True])
    def test_resumable_file_then_native_transport_preserves_advancing_trajectories(
        self, estimator_type, policy, members, round_trip, tmp_path
    ):
        model, X, y = continuation_case(
            estimator_type,
            max_iter=1,
            n_inits=2 if members else 1,
            retain="members" if members else "winner",
            predict_config=policy,
        )
        frame = continuation_frame(X)
        model.set_params(categorical_features="from_dtype").fit(frame, y)
        source = model.network_
        before = model.predict(frame)
        for member in source.members or (source,):
            assert not member.diagnostics.converged
        path = tmp_path / "checkpoint.safetensors"
        source.save(path, resumable=True)
        file_model = round_trip(model)
        file_model.network_ = Network.load(path, device=source.device)
        restored = round_trip(file_model)
        assert restored.network_.can_resume
        assert restored.network_.diagnostics == source.diagnostics
        assert restored.network_.schema == source.schema
        assert restored.network_.predict_config == source.predict_config
        np.testing.assert_array_equal(restored.predict(frame), before)
        np.testing.assert_array_equal(restored.feature_names_in_, model.feature_names_in_)
        np.testing.assert_array_equal(restored.feature_layout_.categories[0], ["a", "b"])
        model.set_params(warm_start="resume", n_inits=1, max_iter=6).fit(frame, y)
        restored.set_params(warm_start="resume", n_inits=1, max_iter=6).fit(frame, y)
        assert restored.loss_curve_ == model.loss_curve_
        assert restored.network_.diagnostics == model.network_.diagnostics
        assert restored.network_.predict_config == model.network_.predict_config
        np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))
        np.testing.assert_array_equal(file_model.predict(frame), before)
        for previous, current in zip(
            source.members or (source,),
            restored.network_.members or (restored.network_,),
            strict=True,
        ):
            assert current.diagnostics.n_iter > previous.diagnostics.n_iter
        if members:
            states = restored.network_.initial_states
            assert any(state is restored.network_.initial_state for state in states)
            for state, member in zip(states, restored.network_.members, strict=True):
                assert member.initial_state is state

    @pytest.mark.parametrize("supplied", [False, True])
    def test_transport_preserves_the_original_prediction_policy_request(self, supplied, round_trip):
        model, X, y = continuation_case(
            EONClassifier,
            warm_start="fine_tune",
            predict_config=PredictConfig(epsilon_P=0.4) if supplied else None,
        )
        model.fit(X, y)
        restored = round_trip(model)
        policy = PredictConfig(
            epsilon_P=None if supplied else restored.network_.predict_config.epsilon_P
        )
        restored.set_params(predict_config=policy)
        with pytest.raises(ValueError, match="temperature choice"):
            restored.fit(X, y)
        restored.set_params(predict_config=None).fit(X, y)
        assert restored.network_.can_resume

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    @pytest.mark.parametrize("members", [False, True])
    def test_fitted_transport_preserves_members_meanings_and_continuation(
        self, estimator_type, mode, members, round_trip
    ):
        model, X, y = continuation_case(
            estimator_type,
            n_inits=2 if members else 1,
            retain="members" if members else "winner",
        )
        frame = continuation_frame(X)
        model.set_params(categorical_features="from_dtype").fit(frame, y)
        model.set_params(n_inits=1, warm_start=mode, max_iter=6)
        restored = round_trip(round_trip(model))
        assert restored.network_.can_resume
        assert restored.network_.diagnostics == model.network_.diagnostics
        np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))
        np.testing.assert_array_equal(restored.feature_names_in_, model.feature_names_in_)
        np.testing.assert_array_equal(restored.feature_layout_.categories[0], ["a", "b"])
        assert restored.effective_dimensions() == model.effective_dimensions()
        if members:
            states = restored.network_.initial_states
            assert len(restored.network_.members) == 2
            assert any(state is restored.network_.initial_state for state in states)
            for state, member in zip(states, restored.network_.members, strict=True):
                assert member.initial_state is state
        model.fit(frame, y)
        restored.fit(frame, y)
        assert restored.loss_curve_ == model.loss_curve_
        np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_prediction_only_transport_cannot_resume_but_can_fine_tune(
        self, estimator_type, round_trip, tmp_path
    ):
        model, X, y = continuation_case(estimator_type)
        model.fit(X, y)
        path = tmp_path / "prediction.safetensors"
        model.network_.save(path)
        model.network_ = Network.load(path, device=model.network_.device)
        model.set_params(warm_start="resume")
        restored = round_trip(model)
        source = restored.network_
        assert not source.can_resume
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
        with pytest.raises(ValueError, match=r"resume|row"):
            restored.fit(X, y)
        assert restored.network_ is source
        restored.set_params(warm_start="fine_tune").fit(X[::2], y[::2])
        assert restored.network_.can_resume
        assert not source.can_resume
        assert round_trip(restored).network_.can_resume

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("selective", [False, True])
    def test_unfitted_explicit_state_transport_and_clone_do_not_acquire_a_network(
        self, estimator_type, selective, round_trip, tmp_path
    ):
        source, X, y = continuation_case(estimator_type)
        source.set_params(
            recipe=source.recipe.chain(
                source.recipe.blocks[0],
                Hidden(K=2, name="middle", epsilon=1.0),
                source.recipe.blocks[-1],
                coupling=Coupling.M,
            )
        )
        source.fit(X, y)
        names = (source.recipe.blocks[0].name, source.recipe.blocks[-1].name)
        state = source.network_.capture_current_state(blocks=names if selective else None)
        if selective:
            assert state.block_names == names
            assert "middle" not in state.block_names
        model = clone(source).set_params(initial_state=state, warm_start="resume")
        restored = round_trip(model)
        assert not hasattr(restored, "network_")
        assert restored.initial_state.block_names == model.initial_state.block_names
        restored.fit(X, y)
        model.fit(X, y)
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
        path = tmp_path / "selective.safetensors"
        restored.network_.save(path, resumable=True)
        restored.network_ = Network.load(path, device=restored.network_.device)
        revived = round_trip(restored)
        assert revived.network_.initial_state.block_names == model.initial_state.block_names
        expected_state = model.network_.initial_state
        loaded_state = revived.network_.initial_state
        torch.testing.assert_close(
            loaded_state.input_geometry.continuous_centroids,
            expected_state.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            loaded_state.input_geometry.feature_weights,
            expected_state.input_geometry.feature_weights,
            rtol=0,
            atol=0,
        )
        for expected, loaded in zip(
            expected_state.parameters, loaded_state.parameters, strict=True
        ):
            assert loaded.description == expected.description
            assert getattr(loaded, "coupling", None) == getattr(expected, "coupling", None)
            for name in ("theta", "C_y", "W_M"):
                if hasattr(expected, name):
                    torch.testing.assert_close(
                        getattr(loaded, name), getattr(expected, name), rtol=0, atol=0
                    )
        model.set_params(max_iter=6).fit(X, y)
        revived.set_params(max_iter=6).fit(X, y)
        assert revived.loss_curve_ == model.loss_curve_
        np.testing.assert_array_equal(revived.predict(X), model.predict(X))
