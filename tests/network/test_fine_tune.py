"""Fine-tuning a fitted Network on new rows through the public lifecycle."""

import io
import logging
from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import active_widths, resume_case, squared_error

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, PredictConfig, Recipe

pytestmark = FIXED_BUDGET


def _assert_same_centroids(actual, expected):
    """Assert two initial states hold the same input centroids."""
    torch.testing.assert_close(
        actual.input_geometry.continuous_centroids,
        expected.input_geometry.continuous_centroids,
        rtol=0,
        atol=0,
    )


class TestFineTune:
    def test_fresh_stopping_controls_are_inherited_by_later_resume(self):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=5, tol=0)
        new_X, new_y = X[:15] + 0.05, y[:15]
        result = source.fine_tune(new_X, new_y, max_iter=2, tol=0)
        assert result.diagnostics.n_iter == 2
        assert (
            result.resume(new_X, new_y).diagnostics.loss_history == result.diagnostics.loss_history
        )
        resumed = result.resume(new_X, new_y, max_iter=4)
        uninterrupted = source.fine_tune(new_X, new_y, max_iter=4, tol=0)
        assert resumed.diagnostics.loss_history == uninterrupted.diagnostics.loss_history
        assert resumed.diagnostics.n_iter == 4

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("rows", [2, 30, 45])
    def test_rebuilds_rows_and_diagnostics_without_changing_source(
        self, task, kind, coupling, rows
    ):
        recipe, X, y, categories = resume_case(task, kind, coupling)
        source = Network.fit(recipe, X, y, X_cat=categories, max_iter=3, seed=5, tol=0)
        before = source.predict(X, X_cat=categories)
        indices = torch.arange(rows, device=X.device) % len(X)
        new_X, new_y = X[indices] + 0.01, y[indices]
        new_categories = tuple(c[indices] for c in categories)
        result = source.fine_tune(new_X, new_y, X_cat=new_categories, max_iter=2, tol=0)
        assert result.can_resume
        assert result.recipe == source.recipe
        assert result.schema.M == source.schema.M
        assert result.schema.M_cat == source.schema.M_cat
        assert result.diagnostics.n_iter == 2
        assert len(result.diagnostics.loss_history) == 3
        assert source.diagnostics.n_iter == 3
        assert result.inspect("training_affiliations")["input"].shape[0] == rows
        assert torch.isfinite(result.predict(new_X, X_cat=new_categories)).all()
        torch.testing.assert_close(source.predict(X, X_cat=categories), before, rtol=0, atol=0)

    # Two distinct rows prune the input from 5 clusters and cap the hidden block from 4.
    @pytest.mark.filterwarnings("ignore:hidden block:UserWarning")
    def test_a_pruned_hidden_block_continues_from_its_active_parameters(self):
        X = torch.tensor([[0.0]] * 3 + [[1.0]] * 3, dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 0, 1, 1, 1], device=DEVICE)
        recipe = Recipe.chain(
            Input(K=5), Hidden(K=4), ClassificationHead(coupling=Coupling.M), coupling=Coupling.M
        )
        source = Network.fit(recipe, X, y, max_iter=1, tol=0)
        assert dict(source.schema.K_active)["hidden_1"] < recipe.blocks[1].K
        rows = torch.tensor([0, 1, 3, 4], device=DEVICE)
        new_X, new_y = X[rows] + 0.1, y[rows]

        result = source.fine_tune(new_X, new_y, max_iter=3, tol=0)

        # A fresh fit from the captured active parameters is the independent route.
        expected = Network.fit(
            active_widths(recipe, source),
            new_X,
            new_y,
            initial_state=source.capture_current_state(),
            max_iter=3,
            tol=0,
        )
        assert result.schema.K_active == expected.schema.K_active
        assert result.diagnostics.loss_history == expected.diagnostics.loss_history
        for name in ("continuous_centroids", "transition_matrices", "training_affiliations"):
            for key, value in expected.inspect(name).items():
                torch.testing.assert_close(result.inspect(name)[key], value, rtol=0, atol=0)
        torch.testing.assert_close(result.predict(new_X), expected.predict(new_X), rtol=0, atol=0)

    def test_same_count_reordered_batch_has_no_position_bound_transfer(self):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2)
        weights = torch.arange(1, len(X) + 1, device=X.device, dtype=X.dtype)
        forward = source.fine_tune(X, y, sample_weights=weights, max_iter=2, tol=0)
        reverse = source.fine_tune(
            X.flip(0), y.flip(0), sample_weights=weights.flip(0), max_iter=2, tol=0
        )
        torch.testing.assert_close(forward.predict(X), reverse.predict(X))
        assert forward.diagnostics.n_iter == reverse.diagnostics.n_iter == 2

    def test_soft_categories_and_targets_keep_schema_when_codes_are_absent(self):
        recipe, X, y, cats = resume_case(kind="categorical")
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=2)
        soft_y = torch.nn.functional.one_hot(y[:5], source.schema.M).to(X.dtype)
        soft_cats = (torch.nn.functional.one_hot(cats[0][:5], source.schema.M_cat[0]).to(X.dtype),)
        result = source.fine_tune(X[:5], soft_y, X_cat=soft_cats, max_iter=2)
        assert result.schema.M == source.schema.M
        assert result.schema.M_cat == source.schema.M_cat
        assert torch.isfinite(result.predict(X, X_cat=cats)).all()

    def test_class_weights_reach_the_continued_fit(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        class_weights = torch.tensor([1.0, 4.0, 16.0], dtype=X.dtype, device=X.device)
        weighted = source.fine_tune(X, y, class_weights=class_weights, max_iter=3, tol=0)
        plain = source.fine_tune(X, y, max_iter=3, tol=0)
        assert weighted.diagnostics.loss_history != plain.diagnostics.loss_history
        assert not torch.equal(weighted.predict(X), plain.predict(X))


class TestFineTuneLogging:
    def test_logging_controls_follow_a_fresh_fit(self, capsys):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=1)
        stream = io.StringIO()
        logger = logging.Logger("caller", level=logging.DEBUG)  # noqa: LOG001
        logger.addHandler(logging.StreamHandler(stream))

        source.fine_tune(X, y, max_iter=2, tol=0, logger=logger, verbose=2)

        assert "loss" in stream.getvalue()
        # The logging controls are checked by the fit's owner; one refusal shows the route.
        with pytest.raises(ValueError, match="logger"):
            source.fine_tune(X, y, logger="caller")
        capsys.readouterr()
        source.fine_tune(X, y, max_iter=1)
        assert capsys.readouterr().err == ""


class TestFineTuneRetention:
    @pytest.mark.parametrize("retain", ["winner", "members"])
    def test_fine_tune_and_resume_preserve_the_selection_history(self, retain):
        recipe, X, y, _ = resume_case("regression")
        rows = torch.arange(len(X), device=X.device)
        source = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            n_inits=2,
            validation_pairs=((rows[:20], rows[20:]), (rows[10:], rows[:10])),
            selection_loss=squared_error,
            return_train_score=True,
            retain=retain,
        )
        outcomes = source.diagnostics.initialisation_outcomes
        assert len(outcomes) == 2
        assert all(outcome.train_score is not None for outcome in outcomes)
        result = source.fine_tune(X[:10], y[:10], max_iter=2, tol=0)
        resumed = result.resume(X[:10], y[:10], max_iter=4, tol=0)
        for continued in (result, resumed):
            assert continued.diagnostics.initialisation_outcomes == outcomes
            assert continued.diagnostics.selected_index == source.diagnostics.selected_index

    @pytest.mark.parametrize("retain", ["winner", "states", "members"])
    def test_retained_members_continue_independently(self, retain):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            n_inits=3,
            retain=retain,
        )
        calls = []

        def weights(target):
            calls.append(target.clone())
            return torch.ones(len(target), dtype=target.dtype, device=target.device)

        result = source.fine_tune(X[:10], y[:10], max_iter=2, task_weights=weights)
        assert len(calls) == 1
        assert (
            result.diagnostics.initialisation_outcomes == source.diagnostics.initialisation_outcomes
        )
        assert (result.initial_states is not None) == (retain != "winner")
        assert (result.members is not None) == (retain == "members")
        if result.initial_states is not None:
            winner = source.diagnostics.selected_index
            _assert_same_centroids(result.initial_state, result.initial_states[winner])
            for before, after in zip(source.initial_states, result.initial_states, strict=True):
                _assert_same_centroids(after, before)
        if result.members is not None:
            for i, (before, after) in enumerate(zip(source.members, result.members, strict=True)):
                expected = before.fine_tune(X[:10], y[:10], max_iter=2)
                _assert_same_centroids(after.initial_state, result.initial_states[i])
                assert after.members is None and after.initial_states is None
                assert after.diagnostics.loss_history == expected.diagnostics.loss_history
                torch.testing.assert_close(after.predict(X), expected.predict(X), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "policy",
        [PredictConfig(), PredictConfig(epsilon_P=0.2), PredictConfig(output_mode="arithmetic")],
    )
    def test_prediction_policy_and_scoring_reference_follow_new_fit(self, policy):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, predict_config=policy)
        new_X = X[:13] + 0.05
        result = source.fine_tune(new_X, y[:13], max_iter=3)
        assert (
            replace(result.predict_config, epsilon_P=source.predict_config.epsilon_P)
            == source.predict_config
        )
        if policy.epsilon_P is not None:
            assert result.predict_config.epsilon_P == policy.epsilon_P
        recovered = result.predict_with_details(
            new_X, details=("instance_weights",)
        ).instance_weights
        assert recovered is not None
        expected = torch.searchsorted(recovered.sort().values, recovered, right=True).to(
            new_X.dtype
        ) / len(new_X)
        torch.testing.assert_close(result.score_samples(new_X), expected, rtol=0, atol=0)
        assert result.resume(new_X, y[:13], max_iter=1).can_resume


class TestFineTuneValidation:
    @pytest.mark.parametrize(
        "invalid", ["features", "category_count", "new_category", "new_class", "regression_width"]
    )
    def test_schema_incompatibility_is_atomic(self, invalid):
        task = "regression" if invalid == "regression_width" else "classification"
        recipe, X, y, cats = resume_case(task, "categorical")
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=2)
        before = source.predict(X, X_cat=cats)
        new_X, new_y, new_cats = X, y, cats
        if invalid == "features":
            new_X = X[:, :1]
            message = "continuous feature count"
        elif invalid == "category_count":
            new_cats = ()
            message = "X_cat must contain 1 fitted"
        elif invalid == "new_category":
            new_cats = (cats[0] + 2,)
            message = r"X_cat\[0\] holds code 3"
        elif invalid == "new_class":
            new_y = y + 3
            message = "exceeds the declared number of classes"
        else:
            new_y = y[:, :1]
            message = "number of regression targets"
        with pytest.raises(ValueError, match=message):
            source.fine_tune(new_X, new_y, X_cat=new_cats)
        torch.testing.assert_close(source.predict(X, X_cat=cats), before, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"predict_config": PredictConfig()}, {"n_inits": 2}, {"seed": 5}, {"initial_state": None}],
    )
    def test_no_policy_or_initialisation_controls(self, kwargs):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=1)
        (name,) = kwargs
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
            source.fine_tune(X, y, **kwargs)
