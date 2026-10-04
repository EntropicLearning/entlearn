"""Portable checkpoints retain fitted trajectories through the public Network."""

import pickle
import warnings

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    assert_same_checkpoint,
    corrupt_saved_tensor,
    read_saved_metadata,
    replace_saved_tensors,
    resume_case,
    round_trip_network,
    set_metadata,
    write_saved_metadata,
)
from safetensors.torch import load_file

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, PredictConfig, Recipe
from entlearn.helpers.reporting import feature_importances


class TestResumablePersistence:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("mode", ["single", "iterative"])
    def test_loaded_resume_matches_uninterrupted_fit(self, tmp_path, task, kind, coupling, mode):
        recipe, X, y, cats = resume_case(task, kind, coupling)
        controls = dict(
            X_cat=cats, tol=0, seed=5, predict_config=PredictConfig(predict_mode=mode, max_iter=3)
        )
        source = Network.fit(recipe, X, y, max_iter=3, **controls)
        assert not source.diagnostics.converged
        full = Network.fit(recipe, X, y, max_iter=8, **controls)
        before = source.predict(X, X_cat=cats)
        path = tmp_path / "checkpoint.safetensors"
        loaded = round_trip_network(source, path, resumable=True)
        assert loaded.can_resume
        assert_same_checkpoint(loaded, source)
        continued = loaded.resume(X, y, X_cat=cats, max_iter=8)
        assert continued.diagnostics.n_iter > source.diagnostics.n_iter
        expected = source.resume(X, y, X_cat=cats, max_iter=8)
        assert_same_checkpoint(continued, expected)
        assert continued.diagnostics.loss_history == full.diagnostics.loss_history
        assert continued.diagnostics.n_iter == full.diagnostics.n_iter
        torch.testing.assert_close(
            continued.predict(X, X_cat=cats), full.predict(X, X_cat=cats), rtol=0, atol=0
        )
        torch.testing.assert_close(
            loaded.score_samples(X, X_cat=cats), source.score_samples(X, X_cat=cats), rtol=0, atol=0
        )
        torch.testing.assert_close(
            feature_importances(loaded), feature_importances(source), rtol=0, atol=0
        )
        torch.testing.assert_close(loaded.predict(X, X_cat=cats), before, rtol=0, atol=0)
        torch.testing.assert_close(source.predict(X, X_cat=cats), before, rtol=0, atol=0)

    @pytest.mark.parametrize("retention", ["winner", "states", "members"])
    @pytest.mark.parametrize("n_inits", [1, 3])
    def test_retained_trajectories_keep_original_winner_and_identity(
        self, tmp_path, retention, n_inits
    ):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            tol=0,
            n_inits=n_inits,
            retain=retention,
        )
        assert not source.diagnostics.converged
        assert all(not member.diagnostics.converged for member in source.members or ())
        loaded = round_trip_network(source, tmp_path / "retained.safetensors", resumable=True)
        expected = source.resume(X, y, max_iter=5)
        result = loaded.resume(X, y, max_iter=5)
        assert result.diagnostics.n_iter > source.diagnostics.n_iter
        assert_same_checkpoint(result, expected)
        assert (result.initial_states is None) == (retention == "winner")
        assert (result.members is None) == (retention != "members")
        if result.initial_states is not None:
            winner = next(
                i for i, s in enumerate(source.initial_states) if s is source.initial_state
            )
            assert result.initial_state is result.initial_states[winner]
            assert loaded.initial_state is loaded.initial_states[winner]
        if result.members is not None:
            for i, (actual, wanted) in enumerate(
                zip(result.members, expected.members, strict=True)
            ):
                assert actual.can_resume and loaded.members[i].can_resume
                assert actual.diagnostics.n_iter > loaded.members[i].diagnostics.n_iter
                assert actual.initial_state is result.initial_states[i]
                assert actual.members is None and actual.initial_states is None
                assert_same_checkpoint(actual, wanted)
            torch.testing.assert_close(
                result.predict_all(X), expected.predict_all(X), rtol=0, atol=0
            )
        assert_same_checkpoint(loaded, source)
        again = round_trip_network(result, tmp_path / "continued.safetensors", resumable=True)
        assert_same_checkpoint(again, result)
        assert_same_checkpoint(again.resume(X, y), result.resume(X, y))
        # Snapshot tensors are caller-editable. Editing a returned starting state
        # must not change the source's reusable starting state or fitted parameters.
        for current, original in zip(
            result.initial_states or (result.initial_state,),
            loaded.initial_states or (loaded.initial_state,),
            strict=True,
        ):
            before = original.input_geometry.continuous_centroids.clone()
            with torch.inference_mode():
                current.input_geometry.continuous_centroids.add_(1)
            torch.testing.assert_close(
                original.input_geometry.continuous_centroids, before, rtol=0, atol=0
            )
        assert_same_checkpoint(loaded, source)

    @pytest.mark.parametrize("converged", [False, True])
    def test_inherited_stopping_controls_survive_repeated_round_trips(self, tmp_path, converged):
        recipe, X, y, _ = resume_case()
        source = Network.fit(
            recipe, X, y, max_iter=8 if converged else 2, tol=1 if converged else 0
        )
        assert source.diagnostics.converged == converged
        path = tmp_path / "stopping.safetensors"
        loaded = round_trip_network(source, path, resumable=True)
        stopped = loaded.resume(X, y)
        assert stopped.diagnostics.loss_history == source.diagnostics.loss_history
        assert stopped.diagnostics.n_iter == source.diagnostics.n_iter
        # Relax only the exhausted control; the other one must be inherited.
        controls = {"tol": 0} if converged else {"max_iter": 8}
        result = loaded.resume(X, y, **controls)
        assert result.diagnostics.n_iter > source.diagnostics.n_iter
        assert_same_checkpoint(result, source.resume(X, y, **controls))
        reloaded = round_trip_network(result, path, resumable=True)
        assert_same_checkpoint(reloaded.resume(X, y), result.resume(X, y))

    def test_pruned_active_shapes_resume_and_fine_tune(self, tmp_path):
        X = torch.tensor([[0.0]] * 3 + [[1.0]] * 3, dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 0, 1, 1, 1], device=DEVICE)
        recipe = Recipe.chain(
            Input(K=5), Hidden(K=4), ClassificationHead(coupling=Coupling.M), coupling=Coupling.M
        )
        source = Network.fit(recipe, X, y, max_iter=1, tol=0)
        assert not source.diagnostics.converged
        assert dict(source.schema.K_active)["input"] < recipe.blocks[0].K
        loaded = round_trip_network(source, tmp_path / "pruned.safetensors", resumable=True)
        assert_same_checkpoint(loaded.resume(X, y, max_iter=4), source.resume(X, y, max_iter=4))
        assert_same_checkpoint(
            loaded.fine_tune(X[:2], y[:2], max_iter=2),
            source.fine_tune(X[:2], y[:2], max_iter=2),
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_saved_dtype_and_explicit_continuation_conversion(self, tmp_path, dtype, kind):
        recipe, X, y, _ = resume_case("regression", kind)
        source = Network.fit(recipe, X.to(torch.float32), y, max_iter=2, tol=0)
        assert not source.diagnostics.converged
        source = source.resume(X, y, computation_dtype=dtype)
        path = tmp_path / "dtype.safetensors"
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            source.save(path, resumable=True)
            loaded = Network.load(path)
            assert loaded.device == torch.device("cpu")
            assert loaded.schema.computation_dtype == dtype
            assert all(value.dtype == dtype for value in load_file(path).values())
            assert loaded.can_resume
        finally:
            torch.set_default_dtype(previous)
        other = torch.float32 if dtype == torch.float64 else torch.float64
        placed = Network.load(path, device=X.device)
        result = placed.resume(X, y, max_iter=4, computation_dtype=other)
        assert result.diagnostics.n_iter > source.diagnostics.n_iter
        expected = source.resume(X, y, max_iter=4, computation_dtype=other)
        assert_same_checkpoint(result, expected)
        assert result.schema.computation_dtype == other
        assert placed.schema.computation_dtype == dtype

    @pytest.mark.parametrize("resumable", [False, True])
    def test_pickle_preserves_loaded_capability_and_member_aliases(self, tmp_path, resumable):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        assert source.members is not None
        assert all(not member.diagnostics.converged for member in source.members)
        loaded = round_trip_network(source, tmp_path / "pickle.safetensors", resumable=resumable)
        transported, original = pickle.loads(pickle.dumps((loaded, loaded.initial_state)))
        assert transported.can_resume == resumable
        torch.testing.assert_close(
            transported.initial_state.input_geometry.continuous_centroids,
            original.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
        for i, member in enumerate(transported.members):
            assert member.can_resume == resumable
            assert member.initial_state is transported.initial_states[i]
        if resumable:
            assert_same_checkpoint(
                transported.resume(X, y, max_iter=4), loaded.resume(X, y, max_iter=4)
            )
        else:
            with pytest.raises(ValueError, match="row-bound"):
                transported.resume(X, y)

    def test_prediction_only_resave_drops_rows_without_mutating_source(self, tmp_path):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        loaded = round_trip_network(source, tmp_path / "resume.safetensors", resumable=True)
        prediction_path = tmp_path / "prediction.safetensors"
        prediction = round_trip_network(loaded, prediction_path, resumable=False)
        assert loaded.can_resume and not prediction.can_resume
        assert all(not member.can_resume for member in prediction.members)
        for name in ("training_affiliations", "training_instance_weights"):
            with pytest.raises(ValueError, match="unavailable"):
                prediction.inspect(name)
        with pytest.raises(ValueError, match="row-bound"):
            prediction.save(prediction_path, resumable=True)
        # A rejected capability upgrade leaves the existing artefact usable.
        assert not Network.load(prediction_path).can_resume
        adapted = prediction.fine_tune(X[:10], y[:10], max_iter=2)
        assert not adapted.diagnostics.converged
        restored = round_trip_network(adapted, tmp_path / "adapted.safetensors", resumable=True)
        assert_same_checkpoint(
            restored.resume(X[:10], y[:10], max_iter=4),
            adapted.resume(X[:10], y[:10], max_iter=4),
        )

    def test_resume_checks_meanings_and_rows_but_not_data_identity(self, tmp_path):
        recipe, X, y, cats = resume_case(kind="categorical")
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=2)
        assert not source.diagnostics.converged
        loaded = round_trip_network(source, tmp_path / "schema.safetensors", resumable=True)
        with pytest.raises(ValueError, match="row count"):
            loaded.resume(X[:10], y[:10], X_cat=(cats[0][:10],))
        for invalid in ((), (cats[0] + 100,), (cats[0], cats[0])):
            with pytest.raises(ValueError):
                loaded.resume(X, y, X_cat=invalid)
        with pytest.raises(ValueError):
            loaded.resume(X[:, :1], y, X_cat=cats)
        with pytest.raises(ValueError):
            loaded.resume(X, y + 100, X_cat=cats)
        result = loaded.resume(X.flip(0), y.flip(0), X_cat=(cats[0].flip(0),), max_iter=4)
        expected = source.resume(X.flip(0), y.flip(0), X_cat=(cats[0].flip(0),), max_iter=4)
        assert_same_checkpoint(result, expected)
        assert_same_checkpoint(loaded, source)


class TestContinuationFinalisation:
    @pytest.mark.parametrize("supplied", [False, True])
    @pytest.mark.parametrize("ceiling", [2, 5])
    def test_calibration_provenance_and_scoring_reference_rebuild(
        self, tmp_path, supplied, ceiling
    ):
        recipe, X, y, _ = resume_case()
        policy = PredictConfig(epsilon_P=0.3 if supplied else None)
        source = Network.fit(recipe, X, y, max_iter=2, tol=0, predict_config=policy)
        assert not source.diagnostics.converged
        loaded = round_trip_network(source, tmp_path / "calibrated.safetensors", resumable=True)
        changed_y = y.roll(7)
        result = loaded.resume(X, changed_y, max_iter=ceiling)
        expected = source.resume(X, changed_y, max_iter=ceiling)
        assert_same_checkpoint(result, expected)
        if supplied:
            assert result.predict_config.epsilon_P == 0.3
        else:
            assert result.predict_config.epsilon_P != source.predict_config.epsilon_P
        torch.testing.assert_close(
            result.score_samples(X), expected.score_samples(X), rtol=0, atol=0
        )
        assert_same_checkpoint(loaded, source)

    def test_backend_change_is_not_a_continuation_identity_check(self, tmp_path):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        assert not source.diagnostics.converged
        loaded = round_trip_network(source, tmp_path / "backend.safetensors", resumable=True)
        before = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(not before)
            assert_same_checkpoint(loaded.resume(X, y, max_iter=4), source.resume(X, y, max_iter=4))
        finally:
            torch.use_deterministic_algorithms(before, warn_only=warn_only)

    @pytest.mark.parametrize("kind", ["cuda", "mps"])
    def test_explicit_device_load_resume_and_cpu_default_reload(self, tmp_path, kind):
        recipe, X, y, _ = resume_case("regression")
        X, y = X.cpu().float(), y.cpu().float()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        assert not source.diagnostics.converged
        path = tmp_path / "placement.safetensors"
        source.save(path, resumable=True)
        available = (
            torch.cuda.is_available() if kind == "cuda" else torch.backends.mps.is_available()
        )
        if not available:
            with pytest.raises(ValueError, match="unavailable"):
                Network.load(path, device=kind)
            return
        loaded = Network.load(path, device=kind)
        query, targets = X.to(loaded.device), y.to(loaded.device)
        result = loaded.resume(query, targets, max_iter=4)
        assert_same_checkpoint(result, source.resume(query, targets, max_iter=4))
        result.save(path, resumable=True)
        on_cpu = Network.load(path)
        assert on_cpu.device == torch.device("cpu")
        assert on_cpu.schema.computation_dtype == torch.float32
        assert on_cpu.can_resume
        assert_same_checkpoint(on_cpu.resume(X, y, max_iter=6), result.resume(X, y, max_iter=6))

    def test_failed_post_load_continuation_keeps_source_unchanged(self, tmp_path):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        loaded = round_trip_network(source, tmp_path / "atomic.safetensors", resumable=True)
        # Turning a changed-objective warning into an error aborts before publication.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with pytest.raises(UserWarning):
                loaded.resume(X + 10, y, max_iter=4)
        assert_same_checkpoint(loaded, source)
        assert loaded.can_resume


class TestResumableValidation:
    @pytest.mark.parametrize("member", [0, 1])
    def test_one_member_cannot_silently_downgrade_the_collection(self, tmp_path, member):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        path = tmp_path / "mixed.safetensors"
        source.save(path, resumable=True)
        set_metadata(path, ("models", member, "capability"), "prediction")
        tensors = load_file(path)
        for name in tuple(tensors):
            if name.startswith(f"models.{member}.") and name.endswith(
                (".gamma", ".instance_weights")
            ):
                del tensors[name]
        replace_saved_tensors(tensors, path)
        with pytest.raises(ValueError, match="capability"):
            Network.load(path)

    @pytest.mark.parametrize("resumable", [False, True])
    def test_self_consistent_member_cannot_change_the_shared_training_row_count(
        self, tmp_path, resumable
    ):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        path = tmp_path / "rows.safetensors"
        source.save(path, resumable=resumable)
        metadata = read_saved_metadata(path)
        other = 1 - metadata["selection"]["selected_index"]
        metadata["models"][other]["training_rows"] -= 1
        write_saved_metadata(path, metadata)
        tensors = load_file(path)
        for key, value in tuple(tensors.items()):
            if key.startswith(f"models.{other}.") and key.endswith(
                (".gamma", ".instance_weights", ".Wt_ref")
            ):
                cropped = value[:-1].clone()
                if key.endswith(".instance_weights"):
                    cropped /= cropped.sum()
                tensors[key] = cropped
        replace_saved_tensors(tensors, path)
        with pytest.raises(ValueError, match="row count"):
            Network.load(path)

    @pytest.mark.parametrize("member", [0, 1])
    @pytest.mark.parametrize(
        "field", ["blocks.0.gamma", "blocks.1.gamma", "blocks.0.instance_weights"]
    )
    @pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nan", "simplex"])
    def test_every_members_complete_row_payload_is_validated(self, tmp_path, member, field, damage):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members")
        path = tmp_path / "damaged.safetensors"
        source.save(path, resumable=True)
        corrupt_saved_tensor(path, f"models.{member}.{field}", damage)
        with pytest.raises(ValueError):
            Network.load(path)

    @pytest.mark.parametrize("capability", ["prediction", "unknown", True])
    def test_capability_marker_cannot_contradict_row_payload(self, tmp_path, capability):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "capability.safetensors"
        source.save(path, resumable=True)
        set_metadata(path, ("models", 0, "capability"), capability)
        with pytest.raises(ValueError):
            Network.load(path)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("schema", {}),
            ("recipe", {}),
            ("training_rows", 31),
            ("connection_sub_seeds", []),
            ("diagnostics", {}),
            ("max_iter", None),
            ("tol", -1),
        ],
    )
    def test_continuation_metadata_is_validated_with_rows(self, tmp_path, field, value):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "metadata.safetensors"
        source.save(path, resumable=True)
        set_metadata(path, ("models", 0, field), value)
        with pytest.raises(ValueError):
            Network.load(path)
