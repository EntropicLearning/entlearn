"""Portable prediction-only workflows through the public Network interface."""

import math
import pickle
from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import assert_same_values, read_saved_metadata, resume_case
from safetensors.torch import load_file

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, PredictConfig, Recipe


class TestPredictionPersistence:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_prediction_only_round_trip(self, tmp_path, task, kind, coupling):
        recipe, X, y, categories = resume_case(task, kind, coupling)
        source = Network.fit(recipe, X, y, X_cat=categories, max_iter=3, seed=5, tol=0)
        path = tmp_path / "model.safetensors"
        source.save(path)
        assert read_saved_metadata(path)
        tensors = load_file(path)
        assert tensors
        assert not any("gamma" in name or "instance_weights" in name for name in tensors)
        restored = Network.load(path)
        assert restored.device == torch.device("cpu")
        assert restored.recipe == source.recipe
        assert restored.schema == source.schema
        assert restored.predict_config == source.predict_config
        assert restored.diagnostics == source.diagnostics
        assert source.can_resume and not restored.can_resume
        query, cats = X.cpu(), tuple(value.cpu() for value in categories)
        tolerance = 0 if X.device.type == "cpu" else 512 * torch.finfo(X.dtype).eps
        torch.testing.assert_close(
            restored.predict(query, X_cat=cats),
            source.predict(X, X_cat=categories).cpu(),
            rtol=tolerance,
            atol=tolerance,
        )
        same_device = Network.load(path, device=X.device)
        # Percentile ranks can jump at ties after cross-device reduction rounding.
        torch.testing.assert_close(
            same_device.score_samples(X, X_cat=categories),
            source.score_samples(X, X_cat=categories),
            rtol=0,
            atol=0,
        )
        for name in ("training_affiliations", "training_instance_weights"):
            with pytest.raises(ValueError, match="unavailable"):
                restored.inspect(name)
        with pytest.raises(ValueError, match="row-bound"):
            restored.resume(query, y.cpu(), X_cat=cats, max_iter=1)

    @pytest.mark.parametrize("retain", ["winner", "states", "members"])
    @pytest.mark.parametrize("n_inits", [1, 3])
    def test_retention_identity_and_independent_member_fine_tuning(self, tmp_path, retain, n_inits):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            n_inits=n_inits,
            retain=retain,
        )
        path = tmp_path / "retained.safetensors"
        source.save(path)
        restored = Network.load(path, device=X.device)
        assert restored.diagnostics == source.diagnostics
        assert (restored.initial_states is not None) == (retain != "winner")
        assert (restored.members is not None) == (retain == "members")
        if restored.initial_states is not None:
            winner = next(
                i for i, s in enumerate(source.initial_states) if s is source.initial_state
            )
            assert restored.initial_state is restored.initial_states[winner]
            assert_same_values(restored.initial_states, source.initial_states)
        if restored.members is not None:
            torch.testing.assert_close(
                restored.predict_all(X), source.predict_all(X), rtol=0, atol=0
            )
            for i, member in enumerate(restored.members):
                assert member.initial_state is restored.initial_states[i]
                assert member.initial_states is None and member.members is None
                assert not member.can_resume
            expected = source.fine_tune(X[:10], y[:10], max_iter=2, tol=0)
            actual = restored.fine_tune(X[:10], y[:10], max_iter=2, tol=0)
            torch.testing.assert_close(
                actual.predict_all(X), expected.predict_all(X), rtol=0, atol=0
            )
            assert actual.can_resume and not restored.can_resume
        restored.save(path)
        again = Network.load(path, device=X.device)
        assert again.diagnostics == restored.diagnostics
        assert not again.can_resume

    def test_members_that_pruned_to_different_widths_round_trip(self, tmp_path):
        X = torch.rand(24, 2, generator=torch.Generator().manual_seed(0), dtype=DTYPE).to(DEVICE)
        y = (X[:, 0] > 0.5).long()
        recipe = Recipe.chain(Input(K=6, epsilon=1e-4), ClassificationHead())
        source = Network.fit(recipe, X, y, n_inits=2, retain="members", seed=1, max_iter=30)
        assert source.members is not None
        widths = [dict(member.schema.K_active)["input"] for member in source.members]
        assert len(set(widths)) > 1
        path = tmp_path / "members.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        assert restored.members is not None
        assert [m.schema for m in restored.members] == [m.schema for m in source.members]
        torch.testing.assert_close(restored.predict_all(X), source.predict_all(X), rtol=0, atol=0)

    def test_a_hidden_block_wider_than_its_input_round_trips(self, tmp_path):
        recipe, X, y, _ = resume_case()
        recipe = recipe.replace_block("hidden_1", K=recipe.blocks[0].K + 2)
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "wide_hidden.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        torch.testing.assert_close(restored.predict(X), source.predict(X), rtol=0, atol=0)

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("mode", ["single", "iterative"])
    def test_queries_fine_tuning_and_capture(self, tmp_path, task, kind, mode):
        recipe, X, y, cats = resume_case(task, kind)
        policy = PredictConfig(predict_mode=mode, max_iter=3)
        source = Network.fit(recipe, X, y, X_cat=cats, predict_config=policy, max_iter=2)
        path = tmp_path / "queries.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        for name in ("continuous_centroids", "transition_matrices", "head_parameters"):
            assert_same_values(restored.inspect(name), source.inspect(name), dtype=DTYPE)
        before = source.predict_with_details(
            X, X_cat=cats, details=("affiliations", "instance_weights", "diagnostics")
        )
        after = restored.predict_with_details(
            X, X_cat=cats, details=("affiliations", "instance_weights", "diagnostics")
        )
        assert_same_values(after.affiliations, before.affiliations, dtype=DTYPE)
        assert_same_values(after.instance_weights, before.instance_weights, dtype=DTYPE)
        assert before.loss_history == after.loss_history
        assert before.n_iter == after.n_iter
        original = source.reconstruct(X, X_cat=cats)
        loaded = restored.reconstruct(X, X_cat=cats)
        assert_same_values(loaded.continuous, original.continuous, dtype=DTYPE)
        assert_same_values(loaded.categorical, original.categorical, dtype=DTYPE)
        for rows in (len(X), 2):
            kwargs = dict(X_cat=tuple(c[:rows] for c in cats), max_iter=2, tol=0)
            expected = source.fine_tune(X[:rows], y[:rows], **kwargs)
            actual = restored.fine_tune(X[:rows], y[:rows], **kwargs)
            assert actual.diagnostics == expected.diagnostics
            assert actual.predict_config == expected.predict_config
            torch.testing.assert_close(
                actual.predict(X, X_cat=cats), expected.predict(X, X_cat=cats), rtol=0, atol=0
            )
            assert actual.resume(X[:rows], y[:rows], **kwargs).can_resume
        state = restored.capture_current_state(blocks=("input",))
        assert state.input_geometry.captured and not restored.initial_state.input_geometry.captured
        assert state.connection_sub_seeds == source.capture_current_state().connection_sub_seeds
        assert_same_values(
            state.input_geometry.continuous_centroids,
            restored.inspect("continuous_centroids")["input"],
            dtype=DTYPE,
        )
        deeper = Recipe.chain(
            state.input_geometry.input,
            recipe.blocks[1],
            Hidden(name="deeper", K=2),
            recipe.blocks[-1],
            coupling=Coupling.M,
        )
        grown = Network.fit(deeper, X, y, X_cat=cats, initial_state=state, max_iter=2, seed=7)
        grown.save(path)
        grown_loaded = Network.load(path, device=DEVICE)
        assert grown_loaded.capture_current_state().connection_sub_seeds == (
            grown.capture_current_state().connection_sub_seeds
        )
        assert not grown_loaded.can_resume

    @pytest.mark.parametrize("scale", [0.5, 1.5])
    def test_near_threshold_assignments_survive_without_training_affiliations(
        self, tmp_path, scale
    ):
        epsilon = scale * torch.finfo(DTYPE).eps
        X = torch.tensor([[0.0], [math.sqrt(epsilon)]], dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 1], device=DEVICE)
        source = Network.fit(
            Recipe.chain(Input(K=2, epsilon=epsilon), ClassificationHead(coupling=Coupling.M)),
            X,
            y,
            max_iter=1,
            seed=2,
        )
        path = tmp_path / "threshold.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        query = X.mean(dim=0, keepdim=True)
        single = restored.predict(query)
        assert torch.count_nonzero(single).item() == (2 if scale > 1 else 1)
        torch.testing.assert_close(single, source.predict(query), rtol=0, atol=0)
        torch.testing.assert_close(restored.predict(query.repeat(16, 1)), single.expand(16, 2))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_saved_dtype_and_promoted_state_remain_portable(self, tmp_path, dtype, kind):
        recipe, X, y, _ = resume_case("regression", kind)
        source = Network.fit(recipe, X.to(torch.float32), y, max_iter=2)
        source = source.resume(X, y, computation_dtype=dtype, max_iter=2)
        path = tmp_path / "precision.safetensors"
        default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            source.save(path)
            restored = Network.load(path)
            assert all(value.dtype == dtype for value in load_file(path).values())
            assert restored.schema.computation_dtype == dtype
            assert restored.initial_state.input_geometry.continuous_centroids.dtype == dtype
            assert restored.predict(X.to(dtype).cpu()).dtype == dtype
        finally:
            torch.set_default_dtype(default)

    @pytest.mark.parametrize(
        "policy", [PredictConfig(epsilon_P=0.3), PredictConfig(output_mode="arithmetic")]
    )
    def test_supplied_and_temperature_free_policies_are_not_recalibrated(self, tmp_path, policy):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, predict_config=policy, max_iter=2)
        path = tmp_path / "policy.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        assert restored.predict_config == source.predict_config
        actual = restored.fine_tune(X[:5], y[:5], max_iter=2)
        assert actual.predict_config == source.predict_config

    @pytest.mark.parametrize("epsilon_T", [0.0, math.inf])
    def test_no_instance_weight_recovery_is_not_fabricated(self, tmp_path, epsilon_T):
        recipe, X, y, _ = resume_case()
        recipe = recipe.replace_block("input", epsilon_T=epsilon_T)
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "no_recovery.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        assert not any(name.endswith("Wt_ref") for name in load_file(path))
        with pytest.raises(ValueError):
            restored.score_samples(X)
        torch.testing.assert_close(source.predict(X), restored.predict(X), rtol=0, atol=0)

    def test_callable_not_retained_and_native_pickle_preserves_prediction_capability(
        self, tmp_path
    ):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(
            recipe,
            X,
            y,
            task_weights=lambda labelled: torch.ones(len(labelled), dtype=DTYPE, device=DEVICE),
            max_iter=2,
            n_inits=2,
            retain="members",
        )
        path = tmp_path / "pickle.safetensors"
        source.save(path)
        restored = pickle.loads(pickle.dumps(Network.load(path, device=DEVICE)))
        assert not restored.can_resume
        assert all(not member.can_resume for member in restored.members)
        for member, state in zip(restored.members, restored.initial_states, strict=True):
            assert member.initial_state is state
        torch.testing.assert_close(source.predict_all(X), restored.predict_all(X), rtol=0, atol=0)

    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_pruned_parameters_and_original_prior_scales_survive(self, tmp_path, kind):
        recipe, X, y, _ = resume_case(kind=kind)
        recipe = recipe.replace_block("input", K=8, epsilon=0)
        original = Network.initialise(recipe, X, y, seed=3)
        original = replace(
            original,
            input_geometry=replace(
                original.input_geometry,
                continuous_centroids=original.input_geometry.continuous_centroids[:1]
                .expand(8, -1)
                .clone(),
            ),
        )
        source = Network.fit(recipe, X, y, initial_state=original, max_iter=2)
        assert dict(source.schema.K_active)["input"] < 8
        path = tmp_path / "pruned.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        assert restored.schema.K_active == source.schema.K_active
        assert restored.initial_state.input_geometry.K_active == 8
        expected = source.fine_tune(X[:10], y[:10], max_iter=2)
        actual = restored.fine_tune(X[:10], y[:10], max_iter=2)
        assert actual.diagnostics == expected.diagnostics
        torch.testing.assert_close(actual.predict(X), expected.predict(X), rtol=0, atol=0)
        assert (
            restored.capture_current_state().input_geometry.K_active
            == dict(source.schema.K_active)["input"]
        )

    @pytest.mark.parametrize("kind", ["categorical", "distribution"])
    def test_categorical_only_and_distribution_inputs(self, tmp_path, kind):
        from network._fixtures import prediction_model

        source, X, cats = prediction_model(input_kind=kind)
        path = tmp_path / "categories.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        torch.testing.assert_close(
            restored.predict(X, X_cat=cats), source.predict(X, X_cat=cats), rtol=0, atol=0
        )
        assert restored.schema.M_cat == source.schema.M_cat

    def test_declaration_order_and_names_are_not_positional_graph_identity(self, tmp_path):
        recipe, X, y, _ = resume_case()
        recipe = replace(recipe, blocks=recipe.blocks[::-1])
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "order.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)
        assert restored.recipe == recipe
        torch.testing.assert_close(restored.predict(X), source.predict(X), rtol=0, atol=0)
