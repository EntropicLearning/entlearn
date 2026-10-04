from dataclasses import replace

import pytest
import torch
from network._fixtures import active_widths, resume_case

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, Recipe, RegressionHead


class TestSelectiveCapture:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_downstream_snapshots_own_active_parameters_without_rows(self, task):
        recipe, X, y, _ = resume_case(task)
        source = Network.fit(recipe, X, y, max_iter=2)
        predictions = source.predict(X)
        state = source.capture_current_state(blocks=("output", "hidden_1"))
        assert state.block_names == ("hidden_1", "output")
        hidden, head = state.parameters
        assert dict(source.schema.K_active)["hidden_1"] == hidden.description.K
        torch.testing.assert_close(
            hidden.theta,
            source.inspect("transition_matrices")["input_to_hidden_1"],
            rtol=0,
            atol=0,
        )
        output = source.inspect("head_parameters")["output"]
        for name, value in output.items():
            torch.testing.assert_close(getattr(head, name), value, rtol=0, atol=0)
        for group in state.parameters:
            assert not hasattr(group, "gamma")
            assert not hasattr(group, "cache")
        with torch.inference_mode():
            hidden.theta.zero_()
            if task == "classification":
                head.theta.zero_()
            else:
                head.C_y.zero_()
                head.W_M.zero_()
        torch.testing.assert_close(source.predict(X), predictions, rtol=0, atol=0)

    @pytest.mark.parametrize("blocks", ["input", 7, {"input"}, ("input", 7), (None,), [[]]])
    def test_rejects_non_sequences_and_non_name_elements(self, blocks):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=1)
        with pytest.raises(ValueError, match="blocks must be a sequence of block names"):
            source.capture_current_state(blocks=blocks)

    def test_capture_selects_named_parameter_groups(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        complete = source.capture_current_state()
        assert complete.block_names == ("input", "hidden_1", "output")
        selected = source.capture_current_state(blocks=("hidden_1",))
        assert selected.block_names == ("hidden_1",)
        assert selected.input_geometry is None
        assert source.capture_current_state(blocks=()).block_names == ()
        with pytest.raises(ValueError, match="unknown"):
            source.capture_current_state(blocks=("typo",))
        with pytest.raises(ValueError, match="duplicate"):
            source.capture_current_state(blocks=("input", "input"))
        assert not hasattr(source, "capture_initial_state")


class TestSelectiveReuse:
    @pytest.mark.parametrize("kind", ["standard", "categorical"])
    def test_new_connection_after_zero_mass_reused_clusters(self, kind):
        recipe, X, y, cats = resume_case(kind=kind)
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=1)
        state = source.capture_current_state(blocks=("input", "hidden_1"))
        provided = state.parameters[0]
        with torch.inference_mode():
            theta = torch.zeros_like(provided.theta)
            theta[0] = 1
        state = replace(state, parameters=(replace(provided, theta=theta),))
        target = Recipe.chain(
            state.input_geometry.input,
            replace(provided.description, epsilon=0),
            Hidden(name="new_hidden", K=2, epsilon=0.2),
            recipe.blocks[-1],
            coupling=Coupling.M,
            theta_alpha=1.1,
        )
        weights = torch.arange(1, len(X) + 1, dtype=X.dtype, device=X.device)
        result = Network.fit(
            target,
            X,
            y,
            X_cat=cats,
            sample_weights=weights,
            initial_state=state,
            max_iter=3,
        )
        assert result.can_resume
        assert torch.isfinite(result.predict(X, X_cat=cats)).all()
        assert all(
            torch.isfinite(matrix).all()
            for matrix in result.inspect("transition_matrices").values()
        )

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_complete_capture_transfers_every_parameter_before_updates(self, monkeypatch, task):
        recipe, X, y, _ = resume_case(task)
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state()
        target = active_widths(recipe, source)
        monkeypatch.setattr("entlearn.network.fit._fit_iteration_", lambda session: None)
        result = Network.fit(target, X[:2], y[:2], initial_state=state, max_iter=1)
        for name in (
            "continuous_centroids",
            "feature_weights",
            "transition_matrices",
            "head_parameters",
        ):
            before, after = source.inspect(name), result.inspect(name)
            for key, value in before.items():
                if isinstance(value, dict):
                    for parameter, tensor in value.items():
                        torch.testing.assert_close(after[key][parameter], tensor, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(after[key], value, rtol=0, atol=0)
        assert result.can_resume

    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_insertion_keeps_named_groups_on_both_sides(self, monkeypatch, coupling):
        _, X, y, _ = resume_case()
        recipe = Recipe.chain(
            Input(K=3, epsilon=0.5),
            Hidden(name="a", K=3, epsilon=0.5),
            Hidden(name="b", K=3, epsilon=0.5),
            ClassificationHead(coupling),
            coupling=coupling,
            theta_alpha=1.1,
        )
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state()
        target = Recipe.chain(
            state.input_geometry.input,
            recipe.blocks[1],
            Hidden(name="graft", K=3, epsilon=0.5),
            recipe.blocks[2],
            recipe.blocks[3],
            coupling=coupling,
            theta_alpha=1.1,
        )
        monkeypatch.setattr("entlearn.network.fit._fit_iteration_", lambda session: None)
        result = Network.fit(target, X, y, initial_state=state, max_iter=1)
        matrices = result.inspect("transition_matrices")
        before = source.inspect("transition_matrices")
        torch.testing.assert_close(matrices["input_to_a"], before["input_to_a"], rtol=0, atol=0)
        torch.testing.assert_close(matrices["graft_to_b"], before["a_to_b"], rtol=0, atol=0)
        torch.testing.assert_close(
            result.inspect("head_parameters")["output"]["theta"],
            source.inspect("head_parameters")["output"]["theta"],
            rtol=0,
            atol=0,
        )
        assert result.initial_state.block_names == ("input", "a", "b", "output")
        assert torch.isfinite(result.predict(X)).all()

    def test_task_change_keeps_input_and_hidden_but_rejects_only_head(self, monkeypatch):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state()
        target = Recipe.chain(
            state.input_geometry.input,
            replace(recipe.blocks[1], K=dict(source.schema.K_active)["hidden_1"]),
            RegressionHead(),
            coupling=Coupling.M,
        )
        monkeypatch.setattr("entlearn.network.fit._fit_iteration_", lambda session: None)
        with pytest.warns(UserWarning, match="output.*incompatible"):
            result = Network.fit(target, X, X[:, :2], initial_state=state, max_iter=1)
        assert result.schema.task == "regression"
        assert result.initial_state.block_names == ("input", "hidden_1")
        torch.testing.assert_close(
            result.inspect("transition_matrices")["input_to_hidden_1"],
            source.inspect("transition_matrices")["input_to_hidden_1"],
            rtol=0,
            atol=0,
        )

    def test_only_head_can_be_reused_while_input_initialises(self, monkeypatch):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state(blocks=("output",))
        target = replace(
            recipe,
            blocks=(
                recipe.blocks[0],
                replace(recipe.blocks[1], K=dict(source.schema.K_active)["hidden_1"]),
                recipe.blocks[-1],
            ),
        )
        monkeypatch.setattr("entlearn.network.fit._fit_iteration_", lambda session: None)
        result = Network.fit(target, X, y, initial_state=state, max_iter=1, seed=77)
        assert not result.initial_state.input_geometry.captured
        assert result.initial_state.block_names == ("input", "output")
        torch.testing.assert_close(
            result.inspect("head_parameters")["output"]["theta"],
            source.inspect("head_parameters")["output"]["theta"],
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("b_width,c_width", [(10, 10), (5, 7)])
    def test_names_do_not_search_for_a_matching_shape(self, monkeypatch, b_width, c_width):
        _, X, y, _ = resume_case()
        recipe = Recipe.chain(
            Input(K=5, epsilon=1),
            Hidden(name="a", K=5, epsilon=1),
            Hidden(name="b", K=7, epsilon=1),
            ClassificationHead(Coupling.M),
            coupling=Coupling.M,
        )
        source = Network.fit(recipe, X, y, max_iter=1)
        state = source.capture_current_state(blocks=("input", "a", "b"))
        target = Recipe.chain(
            state.input_geometry.input,
            recipe.blocks[1],
            Hidden(name="b", K=b_width, epsilon=1),
            Hidden(name="c", K=c_width, epsilon=1),
            recipe.blocks[-1],
            coupling=Coupling.M,
        )
        monkeypatch.setattr("entlearn.network.fit._fit_iteration_", lambda session: None)
        with pytest.warns(UserWarning, match="'b'.*incompatible"):
            result = Network.fit(target, X, y, initial_state=state, max_iter=1)
        assert result.initial_state.block_names == ("input", "a")
        assert result.inspect("transition_matrices")["a_to_b"].shape == (b_width, 5)
        assert result.inspect("transition_matrices")["b_to_c"].shape == (c_width, b_width)


class TestSelectivePrecision:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_float32_fit_does_not_create_float64_intermediates(self, task):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves

        class Float32Only(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                assert all(
                    not isinstance(t, torch.Tensor) or t.dtype != torch.float64
                    for t in tree_leaves(result)
                ), str(func)
                return result

        recipe, X, y, _ = resume_case(task, kind="manifold")
        source = Network.fit(recipe, X.double(), y, max_iter=2)
        state = source.capture_current_state()
        target = active_widths(recipe, source)
        X_new = X.float()
        y_new = y.float() if task == "regression" else y
        with Float32Only():
            result = Network.fit(
                target,
                X_new,
                y_new,
                initial_state=state,
                computation_dtype=torch.float32,
                max_iter=2,
            )
        assert result.predict(X_new).dtype is torch.float32

    def test_capture_after_zero_step_dtype_migration_keeps_projector_precision(self):
        recipe, X, y, _ = resume_case(kind="manifold")
        source = Network.fit(recipe, X.float(), y, max_iter=2)
        promoted = source.resume(X.double(), y, tol=1e10, computation_dtype=torch.float64)
        state = promoted.capture_current_state()
        target = active_widths(recipe, promoted)
        assert Network.fit(target, X.double(), y, initial_state=state, max_iter=2).can_resume

    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_captured_groups_convert_and_survive_transport(self, kind, dtype):
        import pickle

        recipe, X, y, cats = resume_case(kind=kind)
        source = Network.fit(recipe, X.float(), y, X_cat=cats, max_iter=2)
        state = pickle.loads(pickle.dumps(source.capture_current_state()))
        target = active_widths(recipe, source)
        result = Network.fit(
            target, X, y, X_cat=cats, initial_state=state, computation_dtype=dtype, max_iter=2
        )
        assert result.can_resume
        assert result.initial_state.input_geometry.continuous_centroids.dtype is dtype
        assert result.predict(X.to(dtype), X_cat=cats).dtype is dtype
        assert pickle.loads(pickle.dumps(result)).can_resume


class TestSelectiveValidation:
    @pytest.mark.parametrize(
        ("parameter", "subject"),
        [("hidden_1", "hidden transition"), ("output", "classification transition")],
    )
    @pytest.mark.parametrize(
        ("damage", "message"),
        [
            ("nan", "{} must be finite"),
            ("negative", "{} must be non-negative"),
            ("unnormalised", "{} rows must sum to one"),
            ("shape", "invalid {} shape"),
            ("rows", "invalid {} shape"),
            ("flat", "invalid {} shape"),
        ],
    )
    def test_malformed_groups_raise_even_when_not_in_target(
        self, parameter, subject, damage, message
    ):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state(blocks=(parameter,))
        group = state.parameters[0]
        with torch.inference_mode():
            value = group.theta.clone()
            if damage == "shape":
                value = value[None]
            elif damage == "rows":
                # The M-coupled columns still sum to one with an extra zero row.
                value = torch.cat((value, torch.zeros_like(value[:1])))
            elif damage == "flat":
                value = value[:, 0]
            elif damage == "nan":
                value[0, 0] = float("nan")
            elif damage == "negative":
                value[0, 0] = -1
            else:
                value.mul_(2)
        invalid = replace(state, parameters=(replace(group, theta=value),))
        target = Recipe.chain(Input(K=2), RegressionHead(name="new_output"))
        with pytest.raises(ValueError, match=message.format(subject)):
            Network.fit(target, X, X[:, 0], initial_state=invalid, max_iter=1)

    @pytest.mark.parametrize(
        ("damage", "message"),
        [
            ("C_y width", "invalid regression centroid shape"),
            ("description", "invalid regression centroid shape"),
            ("W_M nan", "W_M must be finite"),
            ("W_M unnormalised", "W_M rows must sum to one"),
            ("W_M length", "W_M has invalid shape"),
        ],
    )
    def test_malformed_regression_groups_raise_even_when_not_in_target(self, damage, message):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state(blocks=("output",))
        (group,) = state.parameters
        with torch.inference_mode():
            W_M = group.W_M.clone()
            if damage == "C_y width":
                group = replace(group, C_y=group.C_y[:, :0])
            elif damage == "description":
                group = replace(group, description=Hidden(name="output"))
            elif damage == "W_M nan":
                group = replace(group, W_M=W_M.fill_(float("nan")))
            elif damage == "W_M unnormalised":
                group = replace(group, W_M=W_M.mul_(2))
            else:
                group = replace(group, W_M=torch.cat((W_M, torch.zeros_like(W_M[:1]))))
        invalid = replace(state, parameters=(group,))
        target = Recipe.chain(Input(K=2), RegressionHead(name="new_output"))
        with pytest.raises(ValueError, match=message):
            Network.fit(target, X, X[:, 0], initial_state=invalid, max_iter=1)

    def test_missing_hidden_group_requires_profile_capacity_but_reused_one_does_not(self):
        recipe, X, y, _ = resume_case(coupling=Coupling.S)
        source = Network.fit(recipe, X, y, max_iter=2)
        target = active_widths(recipe, source)
        complete = source.capture_current_state()
        result = Network.fit(target, X[:2], y[:2], initial_state=complete, max_iter=1)
        assert result.can_resume
        missing = source.capture_current_state(blocks=("input", "output"))
        # One row fewer than the active hidden width: pruning during the source fit
        # decides that width, and it differs between device lanes.
        rows = dict(source.schema.K_active)["hidden_1"] - 1
        with pytest.raises(ValueError, match="profiles"):
            Network.fit(target, X[:rows], y[:rows], initial_state=missing, max_iter=1)

    def test_recipe_fixed_output_weights_override_captured_values(self, monkeypatch):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2)
        target = replace(
            recipe,
            blocks=tuple(
                replace(block, K=dict(source.schema.K_active)[block.name])
                if hasattr(block, "K")
                else replace(block, epsilon_M=float("inf"), W_M=(0.8, 0.2))
                for block in recipe.blocks
            ),
        )
        monkeypatch.setattr("entlearn.network.fit._fit_iteration_", lambda session: None)
        result = Network.fit(target, X, y, initial_state=source.capture_current_state(), max_iter=1)
        torch.testing.assert_close(
            result.inspect("head_parameters")["output"]["W_M"],
            torch.tensor([0.8, 0.2], dtype=X.dtype, device=X.device),
        )

    def test_input_fallback_does_not_invalidate_other_groups(self, monkeypatch):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state()
        target = active_widths(recipe, source)
        changed = torch.cat((X, X[:, :1]), dim=1)
        monkeypatch.setattr("entlearn.network.fit._fit_iteration_", lambda session: None)
        with pytest.warns(UserWarning, match="input.*feature layout"):
            result = Network.fit(target, changed, y, initial_state=state, max_iter=1)
        assert not result.initial_state.input_geometry.captured
        torch.testing.assert_close(
            result.inspect("transition_matrices")["input_to_hidden_1"],
            source.inspect("transition_matrices")["input_to_hidden_1"],
            rtol=0,
            atol=0,
        )

    def test_unknown_capture_names_and_duplicate_groups_raise(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state(blocks=("hidden_1",))
        with pytest.raises(ValueError, match="unique"):
            Network.fit(
                recipe,
                X,
                y,
                initial_state=replace(state, parameters=state.parameters * 2),
                max_iter=1,
            )

    def test_a_foreign_parameter_group_is_rejected(self):
        recipe, X, y, _ = resume_case()
        state = Network.fit(recipe, X, y, max_iter=2).capture_current_state()
        foreign = replace(state, parameters=(state.input_geometry,))
        with pytest.raises(ValueError, match="unsupported starting parameter group"):
            Network.fit(recipe, X, y, initial_state=foreign, max_iter=1)

    def test_source_and_provided_state_remain_detached_from_new_fit(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        state = source.capture_current_state()
        target = active_widths(recipe, source)
        prediction = source.predict(X)
        theta = state.parameters[0].theta.clone()
        result = Network.fit(target, X, y, initial_state=state, max_iter=3)
        torch.testing.assert_close(source.predict(X), prediction, rtol=0, atol=0)
        torch.testing.assert_close(state.parameters[0].theta, theta, rtol=0, atol=0)
        assert (
            state.parameters[0].theta.data_ptr()
            != result.initial_state.parameters[0].theta.data_ptr()
        )
