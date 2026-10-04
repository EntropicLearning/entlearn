"""Public continuation contracts for fitted trajectories."""

import math
import warnings
from dataclasses import fields, is_dataclass, replace

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import resume_case

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    LossIncreaseWarning,
    Network,
    PredictConfig,
    Recipe,
    RegressionHead,
)


class TestResumeTrajectory:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_matches_uninterrupted_trajectory(self, task, kind, coupling):
        recipe, X, y, categories = resume_case(task, kind, coupling)
        if categories:
            # A target-derived category can yield identical hard predictions before
            # convergence. Use alternating categories so prediction changes remain visible.
            categories = (torch.arange(len(X), device=X.device).remainder(2),)
        controls = dict(X_cat=categories, seed=5, tol=0)
        full = Network.fit(recipe, X, y, max_iter=8, **controls)
        source = Network.fit(recipe, X, y, max_iter=3, **controls)
        before = source.predict(X, X_cat=categories)
        expected = full.predict(X, X_cat=categories)
        assert not torch.equal(before, expected)
        continued = source.resume(X, y, X_cat=categories, max_iter=8, tol=0)
        assert source.can_resume and continued.can_resume
        assert continued is not source
        assert continued.recipe == source.recipe
        assert source.diagnostics.n_iter == 3
        assert continued.diagnostics.n_iter == full.diagnostics.n_iter
        assert continued.diagnostics.loss_history == full.diagnostics.loss_history
        torch.testing.assert_close(continued.predict(X, X_cat=categories), expected)
        torch.testing.assert_close(source.predict(X, X_cat=categories), before, rtol=0, atol=0)
        assert continued.initial_state is not source.initial_state
        torch.testing.assert_close(
            continued.initial_state.input_geometry.continuous_centroids,
            source.initial_state.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )

    def test_converged_resume_takes_no_extra_step_and_tighter_tolerance_continues(self):
        recipe, X, y, categories = resume_case()
        source = Network.fit(recipe, X, y, X_cat=categories, max_iter=5, tol=1, seed=5)
        assert source.diagnostics.converged
        result = source.resume(X, y, max_iter=10, tol=1)
        assert result.diagnostics.loss_history == source.diagnostics.loss_history
        assert result.diagnostics.n_iter == source.diagnostics.n_iter
        assert result.diagnostics.converged
        tightened = result.resume(X, y, max_iter=source.diagnostics.n_iter + 3, tol=0)
        full = Network.fit(recipe, X, y, max_iter=source.diagnostics.n_iter + 3, tol=0, seed=5)
        assert tightened.diagnostics.loss_history == full.diagnostics.loss_history
        assert tightened.diagnostics.n_iter == full.diagnostics.n_iter

    def test_convergence_uses_requested_tolerance_not_old_flag(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        assert not source.diagnostics.converged
        previous, current = source.diagnostics.loss_history[-2:]
        relaxed = 2 * abs(current - previous) / max(abs(previous), torch.finfo(DTYPE).eps)
        result = source.resume(X, y, max_iter=4, tol=relaxed)
        assert result.diagnostics.converged
        assert result.diagnostics.loss_history == source.diagnostics.loss_history

    @pytest.mark.parametrize("stop", [1, 3, 8])
    def test_continuation_to_convergence_has_identical_stopping_iteration(self, stop):
        recipe, X, y, _ = resume_case()
        full = Network.fit(recipe, X, y, max_iter=200, tol=1e-4, seed=5)
        assert full.diagnostics.converged
        source = Network.fit(recipe, X, y, max_iter=stop, tol=1e-4, seed=5)
        result = source.resume(X, y, max_iter=200, tol=1e-4)
        assert result.diagnostics.n_iter == full.diagnostics.n_iter
        assert result.diagnostics.loss_history == full.diagnostics.loss_history

    def test_pruned_topology_resumes_at_active_widths(self):
        X = torch.tensor([[0.0]] * 3 + [[1.0]] * 3, dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 0, 1, 1, 1], device=DEVICE)
        recipe = Recipe.chain(
            Input(K=5),
            Hidden(K=4),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
        )
        source = Network.fit(recipe, X, y, max_iter=1, tol=0)
        assert dict(source.schema.K_active)["input"] < recipe.blocks[0].K
        result = source.resume(X, y, max_iter=4, tol=0)
        full = Network.fit(recipe, X, y, max_iter=4, tol=0)
        assert result.schema == full.schema
        assert result.diagnostics.loss_history == full.diagnostics.loss_history

    def test_resume_after_a_prune_keeps_the_uninterrupted_scratch_widths(self):
        # Twelve distinct rows cap the input at 12 and the first step prunes the hidden
        # block from 24 to 12; enough rows make the arithmetic depend on the widths.
        points = torch.rand(12, 8, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
        labels = torch.arange(200, device=DEVICE) % 12
        X, y = points.to(DEVICE)[labels], labels % 2
        recipe = Recipe.chain(
            Input(K=32),
            Hidden(K=24),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
        )
        source = Network.fit(recipe, X, y, max_iter=1, tol=0)
        assert dict(source.schema.K_active) == {"input": 12, "hidden_1": 12, "output": 2}
        result = source.resume(X, y, max_iter=6, tol=0)
        full = Network.fit(recipe, X, y, max_iter=6, tol=0)
        assert result.diagnostics.loss_history == full.diagnostics.loss_history


class TestResumeStoppingControls:
    @pytest.mark.parametrize("controls", [{}, {"tol": 1}, {"tol": 2}, {"max_iter": 40}])
    def test_converged_trajectory_does_not_step_without_a_tighter_tolerance(self, controls):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=20, tol=1, seed=5)
        assert source.diagnostics.converged
        assert source.diagnostics.n_iter < 20
        result = source.resume(X, y, **controls)
        assert result.diagnostics.n_iter == source.diagnostics.n_iter
        assert result.diagnostics.loss_history == source.diagnostics.loss_history
        assert result.diagnostics.converged

    @pytest.mark.parametrize("controls", [{}, {"max_iter": 2}, {"max_iter": 1}, {"tol": 0}])
    def test_exhausted_trajectory_does_not_step_without_a_higher_ceiling(self, controls):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0, seed=5)
        assert source.diagnostics.n_iter == 2
        assert not source.diagnostics.converged
        result = source.resume(X, y, **controls)
        assert result.diagnostics.n_iter == 2
        assert result.diagnostics.loss_history == source.diagnostics.loss_history
        assert not result.diagnostics.converged
        for quantity in (
            "training_affiliations",
            "training_instance_weights",
            "continuous_centroids",
            "feature_weights",
        ):
            for name, before in source.inspect(quantity).items():
                torch.testing.assert_close(result.inspect(quantity)[name], before, rtol=0, atol=0)

    def test_raised_ceiling_is_total_and_persists_with_inherited_tolerance(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0, seed=5)
        expected = Network.fit(recipe, X, y, max_iter=6, tol=0, seed=5)
        result = source.resume(X, y, max_iter=6)
        assert result.diagnostics.n_iter == expected.diagnostics.n_iter
        assert result.diagnostics.loss_history == expected.diagnostics.loss_history
        repeated = result.resume(X, y)
        assert repeated.diagnostics.n_iter == result.diagnostics.n_iter
        assert repeated.diagnostics.loss_history == result.diagnostics.loss_history
        assert repeated.diagnostics.converged == result.diagnostics.converged

    def test_tightening_tolerance_uses_remaining_original_budget(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=20, tol=1, seed=5)
        expected = Network.fit(recipe, X, y, max_iter=20, tol=0, seed=5)
        assert source.diagnostics.converged
        result = source.resume(X, y, tol=0)
        assert result.diagnostics.n_iter > source.diagnostics.n_iter
        assert result.diagnostics.n_iter == expected.diagnostics.n_iter
        assert result.diagnostics.loss_history == expected.diagnostics.loss_history
        repeated = result.resume(X, y)
        assert repeated.diagnostics.loss_history == result.diagnostics.loss_history

    def test_dtype_promotion_does_not_reopen_convergence_at_unchanged_tolerance(self):
        X = torch.tensor([[0.0], [1e-4]], dtype=torch.float32, device=DEVICE)
        y = X[:, 0].clone()
        recipe = Recipe.chain(Input(K=2), RegressionHead())
        source = Network.fit(recipe, X, y, max_iter=20, tol=0.1)
        assert source.diagnostics.converged
        assert source.diagnostics.n_iter < 20
        promoted = source.resume(X, y, computation_dtype=torch.float64)
        assert promoted.diagnostics.loss_history == source.diagnostics.loss_history
        assert promoted.diagnostics.converged
        repeated = promoted.resume(X.double(), y.double())
        assert repeated.diagnostics.loss_history == source.diagnostics.loss_history
        assert repeated.diagnostics.converged

    def test_both_stopping_conditions_must_allow_another_iteration(self):
        recipe, X, y, _ = resume_case()
        converged = Network.fit(recipe, X, y, max_iter=20, tol=1, seed=5)
        ceiling = converged.diagnostics.n_iter
        source = Network.fit(recipe, X, y, max_iter=ceiling, tol=1, seed=5)
        assert source.diagnostics.converged
        for controls in ({"tol": 0}, {"max_iter": ceiling + 3}):
            result = source.resume(X, y, **controls)
            assert result.diagnostics.loss_history == source.diagnostics.loss_history
        continued = source.resume(X, y, tol=0, max_iter=ceiling + 3)
        assert continued.diagnostics.n_iter > ceiling
        assert continued.diagnostics.n_iter <= ceiling + 3

    @pytest.mark.parametrize("retain", ["winner", "states", "members"])
    def test_inherited_stops_apply_to_each_retained_member(self, retain):
        recipe, X, y, _ = resume_case()
        source = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            tol=0,
            n_inits=3,
            retain=retain,
        )
        result = source.resume(X, y)
        assert result.diagnostics.loss_history == source.diagnostics.loss_history
        if source.members is not None:
            assert result.members is not None
            for before, after in zip(source.members, result.members, strict=True):
                assert after.diagnostics.loss_history == before.diagnostics.loss_history
                assert after.diagnostics.n_iter == before.diagnostics.n_iter


class TestResumeRetention:
    @pytest.mark.parametrize("retain", ["winner", "states", "members"])
    @pytest.mark.parametrize("n_inits", [1, 3])
    def test_preserves_retention_and_continues_each_member(self, retain, n_inits):
        recipe, X, y, _ = resume_case()
        source = Network.fit(
            recipe,
            X,
            y,
            n_inits=n_inits,
            retain=retain,
            max_iter=2,
            tol=0,
        )
        result = source.resume(X, y, max_iter=5, tol=0)
        assert (
            result.diagnostics.initialisation_outcomes == source.diagnostics.initialisation_outcomes
        )
        assert (result.initial_states is not None) == (retain != "winner")
        assert (result.members is not None) == (retain == "members")
        if result.initial_states is not None:
            winner = next(
                i for i, state in enumerate(source.initial_states) if state is source.initial_state
            )
            assert result.initial_state is result.initial_states[winner]
            for before, after in zip(source.initial_states, result.initial_states, strict=True):
                torch.testing.assert_close(
                    before.input_geometry.continuous_centroids,
                    after.input_geometry.continuous_centroids,
                )
        if result.members is not None:
            for index, (before, after) in enumerate(
                zip(source.members, result.members, strict=True)
            ):
                expected = before.resume(X, y, max_iter=5, tol=0)
                assert after.diagnostics == expected.diagnostics
                assert after.initial_state is result.initial_states[index]
                assert after.members is None and after.initial_states is None
                assert after.diagnostics.n_iter > before.diagnostics.n_iter
                torch.testing.assert_close(after.predict(X), expected.predict(X), rtol=0, atol=0)
            torch.testing.assert_close(
                result.predict(X), result.members[winner].predict(X), rtol=0, atol=0
            )

    def test_regression_callable_runs_once_for_all_retained_members(self):
        recipe, X, y, _ = resume_case(task="regression")
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="members", tol=0)
        calls = []

        def weighting(target):
            calls.append(target.clone())
            return torch.ones(len(target), dtype=target.dtype, device=target.device)

        result = source.resume(X, y, task_weights=weighting, max_iter=4, tol=0)
        assert len(calls) == 1
        assert len(result.members) == 2

    def test_selection_is_not_repeated(self):
        recipe, X, y, _ = resume_case()
        calls = []

        def score(prediction, target, **kwargs):
            calls.append(prediction)
            return float(len(calls))

        source = Network.fit(
            recipe,
            X,
            y,
            selection_loss=score,
            n_inits=3,
            retain="members",
            max_iter=2,
            tol=0,
        )
        assert len(calls) == 3
        result = source.resume(X, y, max_iter=4, tol=0)
        assert len(calls) == 3
        assert result.initial_state is result.initial_states[0]
        assert (
            result.diagnostics.initialisation_outcomes == source.diagnostics.initialisation_outcomes
        )


class TestResumeValidation:
    @pytest.mark.parametrize(
        "fitted_count,supplied_count", [(0, 1), (2, None), (2, 0), (2, 1), (2, 3)]
    )
    def test_rejects_changed_categorical_feature_count(self, fitted_count, supplied_count):
        recipe, X, y, _ = resume_case()
        features = (y.remainder(2), y, y.remainder(2))
        fitted_features = features[:fitted_count]
        supplied_features = None if supplied_count is None else features[:supplied_count]
        source = Network.fit(recipe, X, y, X_cat=fitted_features, max_iter=2, tol=0)
        before = source.predict(X, X_cat=fitted_features)

        with pytest.raises(
            ValueError, match=f"X_cat must contain {fitted_count} fitted categorical"
        ):
            source.resume(X, y, X_cat=supplied_features)

        assert source.can_resume
        torch.testing.assert_close(source.predict(X, X_cat=fitted_features), before, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "field,expected,unrelated",
        [
            ("rows", "training row count must be positive", "computation dtype"),
            ("dtype", "computation dtype must be torch.float32 or torch.float64", "row count"),
        ],
    )
    def test_invalid_retained_metadata_reports_the_specific_cause(self, field, expected, unrelated):
        # Malformed retained metadata has no public constructor before persistence.
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        if field == "rows":
            source._graph.training_rows = 0
        else:
            source._fitted = replace(
                source._fitted,
                schema=replace(source.schema, computation_dtype=torch.float16),
            )

        assert not source.can_resume
        with pytest.raises(ValueError, match=expected) as error:
            source.resume(X, y)
        assert unrelated not in str(error.value)

    @pytest.mark.parametrize(
        "change", ["rows", "features", "outputs", "categories", "nonfinite", "weight"]
    )
    def test_rejects_incompatible_data_without_changing_source(self, change):
        recipe, X, y, categories = resume_case(kind="categorical")
        source = Network.fit(recipe, X, y, X_cat=categories, max_iter=2, tol=0)
        before = source.predict(X, X_cat=categories)
        X_new, y_new, cats_new = X.clone(), y.clone(), categories
        weights = None
        if change == "rows":
            X_new, y_new, cats_new = X[:-1], y[:-1], (categories[0][:-1],)
        elif change == "features":
            X_new = X[:, :-1]
        elif change == "outputs":
            y_new[-1] = 4
        elif change == "categories":
            cats_new = (categories[0].clone(),)
            cats_new[0][-1] = 3
        elif change == "nonfinite":
            X_new[0, 0] = torch.nan
        else:
            weights = torch.zeros(X.shape[0], dtype=DTYPE, device=DEVICE)
        with pytest.raises(ValueError):
            source.resume(X_new, y_new, X_cat=cats_new, sample_weights=weights)
        torch.testing.assert_close(before, source.predict(X, X_cat=categories), rtol=0, atol=0)
        assert source.can_resume

    @pytest.mark.filterwarnings("ignore::entlearn.LossIncreaseWarning")
    def test_accepts_changed_rows_targets_weights_and_missing_known_codes(self):
        recipe, X, y, categories = resume_case(kind="categorical")
        source = Network.fit(recipe, X, y, X_cat=categories, max_iter=2, tol=0)
        result = source.resume(
            X.flip(0) + 0.1,
            torch.zeros_like(y),
            X_cat=(torch.zeros_like(categories[0]),),
            sample_weights=torch.linspace(1, 2, len(X), dtype=DTYPE, device=DEVICE),
            max_iter=4,
            tol=0,
        )
        assert result.schema.M == source.schema.M
        assert result.schema.M_cat == source.schema.M_cat
        assert result.diagnostics.n_iter == 4
        assert torch.isfinite(result.predict(X, X_cat=categories)).all()

    @pytest.mark.parametrize("argument", ["recipe", "mode", "n_inits", "predict_config"])
    def test_rejects_fresh_fit_or_policy_controls(self, argument):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=1)
        with pytest.raises(TypeError):
            source.resume(X, y, **{argument: None})

    def test_can_resume_is_read_only(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=1)
        assert source.can_resume
        with pytest.raises(AttributeError):
            source.can_resume = False

    @pytest.mark.parametrize("field", ["centroids", "hidden", "head", "nonfinite"])
    def test_invalid_row_bound_payload_has_no_resume_capability(self, field):
        # Network.load rejects a corrupt payload, so no public route builds one.
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        block = source._graph.input
        if field == "centroids":
            block.continuous_centroids = block.continuous_centroids[:-1]
        elif field == "hidden":
            hidden = source._graph.blocks["hidden_1"]
            hidden.gamma = torch.zeros_like(hidden.gamma)
        elif field == "head":
            source._graph.head.theta = source._graph.head.theta[:, :-1]
        else:
            block.gamma = torch.full_like(block.gamma, torch.nan)
        assert not source.can_resume
        with pytest.raises(ValueError, match="resum"):
            source.resume(X, y)


class TestResumeFinalisation:
    def test_rebuilds_calibration_and_recovered_scoring_reference(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0, seed=5)
        result = source.resume(X, y, max_iter=5, tol=0)
        full = Network.fit(recipe, X, y, max_iter=5, tol=0, seed=5)
        assert result.predict_config == full.predict_config
        assert result.predict_config.epsilon_P != source.predict_config.epsilon_P
        torch.testing.assert_close(result.score_samples(X), full.score_samples(X), rtol=0, atol=0)
        torch.testing.assert_close(
            result.predict_with_details(X, details=("instance_weights",)).instance_weights,
            full.predict_with_details(X, details=("instance_weights",)).instance_weights,
            rtol=0,
            atol=0,
        )

    def test_supplied_temperature_stays_fixed(self):
        recipe, X, y, _ = resume_case()
        policy = PredictConfig(epsilon_P=0.37)
        source = Network.fit(recipe, X, y, max_iter=2, tol=0, predict_config=policy)
        result = source.resume(X, y, max_iter=4, tol=0)
        assert result.predict_config == source.predict_config

    def test_zero_step_resume_rebuilds_normaliser_on_changed_rows(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(
            recipe, X, y, max_iter=2, tol=0, predict_config=PredictConfig(epsilon_P=0.3)
        )
        changed = X + 0.3
        result = source.resume(changed, y, tol=1e9)
        assert result.diagnostics.loss_history == source.diagnostics.loss_history
        original_weights = source.predict_with_details(
            changed, details=("instance_weights",)
        ).instance_weights
        continued_weights = result.predict_with_details(
            changed, details=("instance_weights",)
        ).instance_weights
        assert not torch.allclose(original_weights, continued_weights)

    def test_repeated_zero_step_resume_preserves_prediction_and_scoring(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=50, tol=0.1)
        result = source.resume(X, y, tol=0.1).resume(X, y, tol=0.1)
        assert result.diagnostics.loss_history == source.diagnostics.loss_history
        torch.testing.assert_close(result.score_samples(X), source.score_samples(X), rtol=0, atol=0)
        torch.testing.assert_close(
            result.predict_with_details(X, details=("instance_weights",)).instance_weights,
            source.predict_with_details(X, details=("instance_weights",)).instance_weights,
            rtol=0,
            atol=0,
        )

    def test_failed_finalisation_does_not_change_any_source_member(self, monkeypatch):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, n_inits=3, retain="members", max_iter=2, tol=0)
        assert source.members is not None
        before = source.predict_all(X)
        histories = tuple(member.diagnostics for member in source.members)
        from entlearn.network import finalise

        real_calibrate = finalise._calibrate_epsilon_P
        calls = []

        def fail_second(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("calibration failed")
            return real_calibrate(*args)

        monkeypatch.setattr(finalise, "_calibrate_epsilon_P", fail_second)
        with pytest.raises(RuntimeError, match="calibration failed"):
            source.resume(X, y, max_iter=4, tol=0)
        torch.testing.assert_close(source.predict_all(X), before, rtol=0, atol=0)
        assert tuple(member.diagnostics for member in source.members) == histories


class TestResumePlacement:
    @pytest.mark.parametrize("max_iter", [2, 4])
    def test_promoted_frozen_simplexes_survive_repeated_float64_resume(self, max_iter):
        _, X, _, _ = resume_case()
        X = X.float()
        y = X.square()
        recipe = Recipe.chain(Input(K=3), RegressionHead(W_M=(1, 2, 4)))
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        feature_weights = source.inspect("feature_weights")["input"].double()
        output_weights = source.inspect("head_parameters")["output"]["W_M"].double()
        # Exact promotion preserves float32 rounding rather than repairing the simplex.
        assert abs(float(feature_weights.sum()) - 1) > 100 * torch.finfo(torch.float64).eps

        promoted = source.resume(X, y, computation_dtype=torch.float64, max_iter=max_iter)
        repeated = promoted.resume(X.double(), y.double())

        assert repeated.can_resume
        assert repeated.schema.computation_dtype is torch.float64
        assert repeated.diagnostics.n_iter == max_iter
        assert repeated.diagnostics.loss_history == promoted.diagnostics.loss_history
        torch.testing.assert_close(
            repeated.inspect("feature_weights")["input"], feature_weights, rtol=0, atol=0
        )
        torch.testing.assert_close(
            repeated.inspect("head_parameters")["output"]["W_M"], output_weights, rtol=0, atol=0
        )

    @pytest.mark.parametrize("family", ["target", "categorical"])
    def test_retained_precision_does_not_relax_new_float64_distributions(self, family):
        recipe, X, y, categories = resume_case(kind="categorical")
        source = Network.fit(recipe, X.float(), y, X_cat=categories, max_iter=2, tol=0)
        promoted = source.resume(X.double(), y, X_cat=categories, tol=1e9)
        target = torch.nn.functional.one_hot(y, promoted.schema.M).double()
        feature = torch.nn.functional.one_hot(categories[0], promoted.schema.M_cat[0]).double()
        values = target if family == "target" else feature
        values[0, 0] += torch.finfo(torch.float32).eps
        assert abs(float(values[0].sum()) - 1) > 100 * torch.finfo(torch.float64).eps

        with pytest.raises(ValueError, match="sum to one"):
            promoted.resume(X.double(), target, X_cat=(feature,))
        assert promoted.can_resume

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_converts_owned_state_and_all_retained_geometry(self, dtype):
        recipe, X, y, categories = resume_case(kind="categorical")
        source = Network.fit(
            recipe, X, y, X_cat=categories, retain="members", n_inits=2, max_iter=2, tol=0
        )
        result = source.resume(X, y, X_cat=categories, computation_dtype=dtype, max_iter=4, tol=0)
        assert source.schema.computation_dtype == DTYPE
        assert result.schema.computation_dtype == dtype
        assert result.predict(X.to(dtype), X_cat=categories).dtype == dtype
        for member, state in zip(result.members, result.initial_states, strict=True):
            assert member.schema.computation_dtype == dtype
            assert state.input_geometry.continuous_centroids.dtype == dtype
            assert state.input_geometry.feature_weights.dtype == dtype
            assert all(value.dtype == dtype for value in state.input_geometry.categorical_centroids)

    def test_converts_and_copies_captured_original_parameters(self):
        recipe, X, y, _ = resume_case()
        captured = Network.fit(recipe, X, y, max_iter=2).capture_current_state()
        source = Network.fit(recipe, X, y, initial_state=captured, max_iter=2, tol=0)
        other = torch.float64 if DTYPE is torch.float32 else torch.float32
        result = source.resume(X, y, computation_dtype=other, max_iter=3, tol=0)
        before, after = source.initial_state.parameters, result.initial_state.parameters
        assert [group.description.name for group in after] == ["hidden_1", "output"]
        for old, new in zip(before, after, strict=True):
            assert new.theta.dtype == other
            assert new.theta.data_ptr() != old.theta.data_ptr()

    def test_dtype_change_resolves_the_input_regime_before_prediction(self):
        epsilon = torch.finfo(torch.float32).eps
        X = torch.tensor([[0.0], [math.sqrt(epsilon)]], dtype=torch.float64, device=DEVICE)
        y = torch.tensor([0, 1], device=DEVICE)
        recipe = Recipe.chain(Input(K=2, epsilon=epsilon), ClassificationHead(coupling=Coupling.M))
        source = Network.fit(recipe, X, y, max_iter=1, predict_config=PredictConfig(epsilon_P=1))
        result = source.resume(X, y, computation_dtype=torch.float32, tol=1e9)
        query = X.mean(dim=0, keepdim=True)
        before = source.predict_with_details(query, details=("affiliations",)).affiliations["input"]
        after = result.predict_with_details(query.float(), details=("affiliations",)).affiliations[
            "input"
        ]
        assert torch.count_nonzero(before) == 2
        assert torch.count_nonzero(after) == 1
        assert result.diagnostics.loss_history == source.diagnostics.loss_history

    @pytest.mark.parametrize("operation", ["resume", "fine_tune"])
    def test_fitted_state_must_stay_finite_after_narrowing(self, operation):
        # A finite float64 value beyond the float32 range becomes infinite on conversion.
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X.double(), y, max_iter=2)
        block = source._graph.input
        huge = 2 * float(torch.finfo(torch.float32).max)
        block.continuous_centroids = torch.full_like(block.continuous_centroids, huge)

        with pytest.raises(ValueError, match=r"fitted state must stay finite in torch\.float32"):
            getattr(source, operation)(X.double(), y, computation_dtype=torch.float32)

    @pytest.mark.parametrize("operation", ["resume", "fine_tune"])
    def test_original_state_must_stay_finite_after_narrowing(self, operation):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X.double(), y, max_iter=2)
        with torch.inference_mode():
            source._fitted.initial_state.input_geometry.continuous_centroids.fill_(
                2 * float(torch.finfo(torch.float32).max)
            )

        with pytest.raises(ValueError, match=r"fitted state must stay finite in torch\.float32"):
            getattr(source, operation)(X.double(), y, computation_dtype=torch.float32)

    def test_float32_resume_retains_no_float64_tensors_under_float64_default(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0, retain="members", n_inits=2)
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            result = source.resume(X, y, computation_dtype=torch.float32, max_iter=4, tol=0)
        finally:
            torch.set_default_dtype(default)

        def check(value):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert value.dtype == torch.float32
            elif is_dataclass(value):
                for field in fields(value):
                    check(getattr(value, field.name))
            elif isinstance(value, dict):
                for child in value.values():
                    check(child)
            elif isinstance(value, tuple):
                for child in value:
                    check(child)

        # Recursive dtype inspection is deliberately below the public test seam.
        check(result._fitted)
        assert result.members is not None
        for member in result.members:
            check(member._fitted)

    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    @pytest.mark.parametrize(
        "destination", ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
    )
    def test_selected_device_owns_all_retained_state(self, destination, kind):
        recipe, X, y, _ = resume_case(kind=kind)
        source = Network.fit(recipe, X, y, max_iter=2, tol=0, retain="members", n_inits=2)
        result = source.resume(X.to(destination), y.to(destination), max_iter=4, tol=0)
        assert result.device.type == destination
        assert result.members is not None
        for member in result.members:
            assert member.device.type == destination
            assert (
                member.initial_state.input_geometry.continuous_centroids.device.type == destination
            )
        assert result.predict(X.to(destination)).device.type == destination


class TestResumeCheckpointWarning:
    @pytest.mark.parametrize("members", [False, True])
    def test_changed_objective_warns_and_retains_message_even_without_steps(self, members):
        recipe, X, y, _ = resume_case()
        source = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            tol=0,
            n_inits=2 if members else 1,
            retain="members" if members else "winner",
        )
        before = source.diagnostics
        with pytest.warns(UserWarning, match="resume checkpoint loss differs") as seen:
            result = source.resume(X + 0.4, y, tol=1e9)
        assert len(seen) == (2 if members else 1)
        assert result.diagnostics.loss_history == before.loss_history
        assert result.diagnostics.n_iter == before.n_iter
        assert source.diagnostics == before
        continued = result.members if members else (result,)
        for member, warning in zip(continued, seen, strict=True):
            message = str(warning.message)
            assert warning.category is UserWarning
            assert message in member.diagnostics.warnings
            assert "stored=" in message and "recomputed=" in message and "threshold=" in message
        assert any("resume checkpoint loss differs" in msg for msg in result.diagnostics.warnings)

    @pytest.mark.parametrize("sign", [-1, 1])
    @pytest.mark.parametrize("relative_to_noise", [0.5, 2])
    def test_checkpoint_difference_uses_numerical_noise_not_fit_tolerance(
        self, monkeypatch, sign, relative_to_noise
    ):
        from entlearn.network import fit as fit_ops

        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        stored = source.diagnostics.loss_history[-1]
        noise = fit_ops.loss_noise_threshold(stored, DTYPE)
        # Inject precise objective differences at the public resume seam.
        monkeypatch.setattr(
            fit_ops, "_loss", lambda session: stored + sign * relative_to_noise * noise
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = source.resume(X, y, tol=1e9)
        checkpoint_warnings = [
            warning for warning in seen if "resume checkpoint loss differs" in str(warning.message)
        ]
        assert len(checkpoint_warnings) == (1 if relative_to_noise > 1 else 0)
        assert result.diagnostics.loss_history == source.diagnostics.loss_history

    def test_unchanged_checkpoint_does_not_warn(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            source.resume(X, y, tol=1e9)
        assert not seen

    def test_dtype_promotion_uses_the_less_precise_checkpoint_tolerance(self, monkeypatch):
        from entlearn.network import fit as fit_ops

        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X.float(), y, max_iter=2, tol=0)
        stored = source.diagnostics.loss_history[-1]
        noise = fit_ops.loss_noise_threshold(stored, torch.float32)
        monkeypatch.setattr(fit_ops, "_loss", lambda session: stored + noise / 2)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            source.resume(X, y, computation_dtype=torch.float64, tol=1e9)
        assert not seen

    def test_a_resumed_loss_increase_names_the_accumulated_iteration(self, monkeypatch):
        from entlearn.network import fit as fit_ops

        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        stored = source.diagnostics.loss_history[-1]
        calls = 0

        def rising(session):
            # The checkpoint recomputes the stored loss, then every step worsens it.
            nonlocal calls
            calls += 1
            return stored + max(0, calls - 1)

        monkeypatch.setattr(fit_ops, "_loss", rising)
        with pytest.warns(LossIncreaseWarning) as seen:
            result = source.resume(X, y, max_iter=4, tol=0)

        messages = [
            str(record.message) for record in seen if record.category is LossIncreaseWarning
        ]
        assert [message.split(" by ")[0] for message in messages] == [
            "loss increased at iteration 3",
            "loss increased at iteration 4",
        ]
        assert result.diagnostics.n_iter == 4
        assert all(message in result.diagnostics.warnings for message in messages)

    def test_checkpoint_warning_as_error_leaves_source_unchanged(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, tol=0)
        before = source.predict(X)
        diagnostics = source.diagnostics
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="resume checkpoint loss differs")
            with pytest.raises(UserWarning, match="resume checkpoint loss differs"):
                source.resume(X + 0.4, y, tol=1e9)
        assert source.diagnostics == diagnostics
        torch.testing.assert_close(source.predict(X), before, rtol=0, atol=0)
