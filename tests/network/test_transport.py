"""Public model transport uses canonical state and retains owned relationships."""

import copy
import pickle

import pytest
import torch
from conftest import FIXED_BUDGET
from network._fixtures import (
    assert_same_checkpoint,
    assert_same_values,
    change_metadata,
    diverging_regression,
    resume_case,
    round_trip_network,
)

from entlearn import InitialState, LossIncreaseWarning, Network

pytestmark = FIXED_BUDGET


class TestCanonicalTransport:
    def test_public_load_rejects_a_candidate_without_selection_outcomes(self, tmp_path):
        recipe, X, y, _ = resume_case("regression")
        model = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "model.safetensors"
        model.save(path, resumable=True)

        def remove_selection(metadata):
            metadata["selection"]["initialisation_outcomes"] = []

        change_metadata(path, remove_selection)
        with pytest.raises(ValueError, match="selection has not finished"):
            Network.load(path)

    @pytest.mark.parametrize(("n_inits", "retain"), [(1, "winner"), (2, "members")])
    def test_a_diverged_loss_history_transports_on_every_route(self, tmp_path, n_inits, retain):
        def trajectory(network):
            diagnostics = network.diagnostics
            scores = tuple(outcome.score for outcome in diagnostics.initialisation_outcomes)
            values = (*diagnostics.loss_history, *scores, diagnostics.selected_index)
            return torch.tensor(values, dtype=torch.float64)

        recipe, X, y = diverging_regression()
        with pytest.warns(LossIncreaseWarning):
            source = Network.fit(recipe, X, y, max_iter=3, seed=0, n_inits=n_inits, retain=retain)
        assert not torch.isfinite(trajectory(source)).all()
        path = tmp_path / "diverged.safetensors"
        source.save(path)

        for restored in (
            copy.deepcopy(source),
            pickle.loads(pickle.dumps(source)),
            Network.load(path, device=X.device),
        ):
            for actual, expected in zip(
                (restored, *(restored.members or ())),
                (source, *(source.members or ())),
                strict=True,
            ):
                torch.testing.assert_close(
                    trajectory(actual), trajectory(expected), rtol=0, atol=0, equal_nan=True
                )

    @pytest.mark.parametrize("snapshot", [False, True])
    def test_deepcopy_restores_owned_inference_state(self, snapshot):
        recipe, X, y, _ = resume_case("regression")
        model = Network.fit(recipe, X, y, max_iter=2)
        source = model.initial_state if snapshot else model
        restored = copy.deepcopy(source)
        state = restored if snapshot else restored.initial_state
        assert state.input_geometry.continuous_centroids.is_inference()
        before = model.initial_state.input_geometry.continuous_centroids.clone()
        with torch.inference_mode():
            state.input_geometry.continuous_centroids.add_(1)
        torch.testing.assert_close(
            model.initial_state.input_geometry.continuous_centroids, before, rtol=0, atol=0
        )

    @pytest.mark.parametrize("resumable", [False, True])
    def test_file_and_pickle_carry_the_same_model(self, tmp_path, resumable):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        loaded = round_trip_network(source, tmp_path / "model.safetensors", resumable=resumable)
        transported = pickle.loads(pickle.dumps(loaded))
        assert transported.can_resume == resumable
        assert transported.diagnostics == loaded.diagnostics
        torch.testing.assert_close(
            transported.predict_all(X), loaded.predict_all(X), rtol=0, atol=0
        )
        assert_same_values(transported.initial_states, loaded.initial_states)
        if resumable:
            assert_same_checkpoint(transported, loaded)

    @pytest.mark.parametrize("transport", ["pickle", "deepcopy"])
    def test_transport_restores_the_public_record(self, transport):
        recipe, X, y, cats = resume_case("regression")
        model = Network.fit(recipe, X, y, X_cat=cats, max_iter=2, n_inits=2, retain="members")
        restored = (
            pickle.loads(pickle.dumps(model, protocol=5))
            if transport == "pickle"
            else copy.deepcopy(model)
        )
        assert restored.can_resume
        assert restored.diagnostics == model.diagnostics
        torch.testing.assert_close(restored.predict_all(X), model.predict_all(X), rtol=0, atol=0)
        for state, member in zip(restored.initial_states, restored.members, strict=True):
            torch.testing.assert_close(
                member.initial_state.input_geometry.continuous_centroids,
                state.input_geometry.continuous_centroids,
                rtol=0,
                atol=0,
            )

    def test_caller_uses_member_index_without_an_extra_payload(self):
        recipe, X, y, _ = resume_case("regression")
        model = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        bundle = pickle.loads(pickle.dumps({"network": model, "member_index": 1}))
        restored = bundle["network"]
        selected = restored.members[bundle["member_index"]]
        assert selected.initial_state is restored.initial_states[1]
        torch.testing.assert_close(selected.predict(X), model.members[1].predict(X), rtol=0, atol=0)
        before = model.initial_states[1].input_geometry.continuous_centroids.clone()
        with torch.inference_mode():
            selected.initial_state.input_geometry.continuous_centroids.add_(1)
        torch.testing.assert_close(
            model.initial_states[1].input_geometry.continuous_centroids, before, rtol=0, atol=0
        )


class TestInitialStateTransportRejection:
    def test_restoring_a_state_with_non_tuple_parameters_raises(self):
        malformed = InitialState(parameters=[1, 2, 3])
        with pytest.raises(ValueError, match="InitialState\\.parameters must be a tuple"):
            pickle.loads(pickle.dumps(malformed))
