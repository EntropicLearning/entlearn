"""Public model transport uses canonical state and retains owned relationships."""

import copy
import json
import pickle
from collections import Counter
from dataclasses import replace

import pytest
import torch
from network._fixtures import (
    change_metadata,
    diverging_regression,
    resume_case,
    squared_error,
    validation_pairs,
)
from safetensors import safe_open

from entlearn import InitialState, LossIncreaseWarning, Network


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
    def test_deepcopy_does_not_copy_tensors_before_owned_restoration(self, snapshot, monkeypatch):
        recipe, X, y, _ = resume_case("regression")
        model = Network.fit(recipe, X, y, max_iter=2)

        def unexpected_copy(*args, **kwargs):
            raise AssertionError("owned restoration must not follow a preliminary tensor deepcopy")

        monkeypatch.setattr(torch.Tensor, "__deepcopy__", unexpected_copy)
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

    def test_deepcopy_memo_keys_the_copy_on_the_source_alone(self):
        recipe, X, y, _ = resume_case("regression")
        model = Network.fit(recipe, X, y, max_iter=2)
        first, second = copy.deepcopy([model, model])
        assert first is second and first is not model
        alongside = copy.deepcopy({"model": model, "empty": None})
        assert alongside["model"] is not model
        assert alongside["empty"] is None

    @pytest.mark.parametrize("resumable", [False, True])
    def test_file_and_pickle_carry_the_same_model_representation(self, tmp_path, resumable):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        path = tmp_path / "model.safetensors"
        source.save(path, resumable=resumable)
        loaded = Network.load(path, device=X.device)
        _, (metadata, tensors, device) = loaded.__reduce__()
        with safe_open(path, framework="pt", device="cpu") as bundle:
            assert json.loads(metadata) == json.loads(bundle.metadata()["entlearn.network"])
            assert set(tensors) == set(bundle.keys())
            for name, value in tensors.items():
                torch.testing.assert_close(value.cpu(), bundle.get_tensor(name), rtol=0, atol=0)
        assert device == str(X.device)
        assert pickle.loads(pickle.dumps(loaded)).can_resume == resumable

    def test_restoration_copies_each_incoming_tensor_once_and_releases_the_map(self, monkeypatch):
        # A narrow allocation probe at the pickle seam: count storage copies, not
        # allocations used for validation or reconstructing derived log caches.
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        restore, arguments = source.__reduce__()
        tensors = arguments[1]
        pointers = {value.untyped_storage().data_ptr() for value in tensors.values()}
        copied = Counter()
        original_to = torch.Tensor.to

        def count_copy(value, *args, **kwargs):
            pointer = value.untyped_storage().data_ptr()
            if kwargs.get("copy") and pointer in pointers:
                copied[pointer] += 1
            return original_to(value, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", count_copy)
        restored = restore(*arguments)
        assert not tensors
        assert copied == Counter(dict.fromkeys(pointers, 1))
        torch.testing.assert_close(restored.predict_all(X), source.predict_all(X), rtol=0, atol=0)
        original = source.initial_state.input_geometry.continuous_centroids.clone()
        with torch.inference_mode():
            restored.initial_state.input_geometry.continuous_centroids.add_(1)
        torch.testing.assert_close(
            source.initial_state.input_geometry.continuous_centroids, original, rtol=0, atol=0
        )

    @pytest.mark.parametrize("transport", [pickle.loads, copy.deepcopy])
    def test_transport_does_not_encode_private_fitted_records(self, transport):
        recipe, X, y, cats = resume_case("regression")
        model = Network.fit(recipe, X, y, X_cat=cats, max_iter=2, n_inits=2, retain="members")
        payload = pickle.dumps(model, protocol=5)
        assert b"_FittedNetwork" not in payload
        assert b"_CompiledGraph" not in payload
        restored = transport(payload) if transport is pickle.loads else transport(model)
        assert restored.can_resume
        assert restored.diagnostics == model.diagnostics
        torch.testing.assert_close(restored.predict_all(X), model.predict_all(X), rtol=0, atol=0)
        for state, member in zip(restored.initial_states, restored.members, strict=True):
            assert member.initial_state is state
            assert (
                member.initial_state.input_geometry.continuous_centroids.untyped_storage().data_ptr()
                == state.input_geometry.continuous_centroids.untyped_storage().data_ptr()
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

    @pytest.mark.parametrize("operation", ["save", "resume"])
    def test_a_selected_state_outside_the_retained_states_is_named(self, tmp_path, operation):
        # A selected original state absent from the retained collection has no public
        # constructor; both consumers must name the broken identity.
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="states")
        source._fitted = replace(source._fitted, initial_state=copy.deepcopy(source.initial_state))
        with pytest.raises(ValueError, match="no retained original state"):
            if operation == "save":
                source.save(tmp_path / "model.safetensors")
            else:
                source.resume(X, y)

    def test_deepcopy_memoises_the_original_state_and_nothing_else(self):
        recipe, X, y, _ = resume_case("regression")
        state = Network.fit(recipe, X, y, max_iter=1).initial_state

        first, second, absent = copy.deepcopy((state, state, None))

        assert first is second
        assert first is not state
        assert absent is None
        torch.testing.assert_close(
            first.input_geometry.continuous_centroids,
            state.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("cross_validated", [False, True])
    def test_candidate_transport_does_not_encode_private_fitted_records(
        self, monkeypatch, cross_validated
    ):
        # Inspect worker envelopes produced by a real fit; do not construct private state.
        from entlearn.network import selection

        original = selection._map_seeds
        observed = []

        def inspect_envelopes(*args, **kwargs):
            backend, results = original(*args, **kwargs)

            def envelopes():
                for value in results:
                    envelope = pickle.dumps(value, protocol=5)
                    assert b"_FittedNetwork" not in envelope
                    assert b"_CompiledGraph" not in envelope
                    restored = pickle.loads(envelope)
                    observed.append((value, restored))
                    yield restored

            return backend, envelopes()

        monkeypatch.setattr(selection, "_map_seeds", inspect_envelopes)
        recipe, X, y, _ = resume_case("regression")
        folds = validation_pairs() if cross_validated else None
        model = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            n_inits=2,
            retain="members",
            validation_pairs=folds,
            selection_loss=squared_error if cross_validated else None,
            return_train_score=cross_validated,
        )
        assert len(observed) == 2
        for value, restored in observed:
            assert (restored.fitted is None) is cross_validated
            assert len(restored.trajectories) == (2 if cross_validated else 1)
            assert restored.trajectories == value.trajectories
            assert (restored.score, restored.train_score) == (value.score, value.train_score)
            assert (restored.train_score is not None) is cross_validated
        assert model.members is not None
        assert all(member.can_resume for member in model.members)


class TestInitialStateTransportRejection:
    def test_restoring_a_state_with_non_tuple_parameters_raises(self):
        malformed = InitialState(parameters=[1, 2, 3])
        with pytest.raises(ValueError, match="InitialState\\.parameters must be a tuple"):
            pickle.loads(pickle.dumps(malformed))
