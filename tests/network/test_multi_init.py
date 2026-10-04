"""Network-owned candidate fitting, selection and detached retention."""

import gc
import math
import weakref

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import blobs, classification_recipe, regression_blobs, regression_recipe

from entlearn import InitialState, Network, PredictConfig
from entlearn.network import selection
from entlearn.network.state import InitOutcome, _winning_outcome

NAN, INF = math.nan, math.inf


class _DerivedState(InitialState):
    """A subclass, which Network.fit refuses as an initial state."""


def outcomes(scores, train_scores=None):
    """Return one selection record per score, in candidate order."""
    return tuple(
        InitOutcome(
            index=index,
            seed=index,
            score=score,
            train_score=None if train_scores is None else train_scores[index],
            requested_backend="threads",
            effective_backend="serial",
        )
        for index, score in enumerate(scores)
    )


class TestSelectionRule:
    @pytest.mark.parametrize(
        "scores,expected",
        [
            ((3.0, 1.0, 5.0, 2.0), 1),
            ((NAN, 1.0, 5.0, 2.0), 1),
            ((1.0, 1.0, 1.0, 1.0), 0),
            ((NAN,) * 4, 0),
            ((INF, NAN, -INF, 0.0), 2),
        ],
    )
    def test_the_lowest_score_wins_with_nan_last_and_ties_in_candidate_order(
        self, scores, expected
    ):
        assert _winning_outcome(outcomes(scores)).index == expected

    def test_a_recorded_gap_breaks_only_exact_score_ties(self):
        # Gaps 0, 9, 1 and 1: candidate 0 has the smallest gap but not the lowest score.
        records = outcomes((6.0, 5.0, 5.0, 5.0), (6.0, -4.0, 4.0, 4.0))
        assert _winning_outcome(records).index == 2
        assert _winning_outcome(outcomes((NAN, NAN), (100.0, -100.0))).index == 0


class TestNetworkMultiInit:
    def test_default_selection_deploys_the_lowest_training_objective_candidate(self):
        X, codes = blobs(6, 0.15, rows_per_blob=5)
        y = codes % 3
        recipe = classification_recipe(K=3)

        selected = Network.fit(
            recipe,
            X,
            y,
            seed=13,
            n_inits=4,
            retain="members",
            max_iter=5,
        )

        outcomes = selected.diagnostics.initialisation_outcomes
        assert outcomes is not None
        assert len(outcomes) == 4
        for outcome, state, member in zip(
            outcomes, selected.initial_states, selected.members, strict=True
        ):
            replayed = Network.fit(recipe, X, y, seed=outcome.seed, max_iter=5)
            assert outcome.score == replayed.diagnostics.loss_history[-1]
            assert member.diagnostics.loss_history == replayed.diagnostics.loss_history
            assert member.predict_config == replayed.predict_config
            from_state = Network.fit(recipe, X, y, initial_state=state, max_iter=5)
            assert torch.equal(replayed.predict(X), from_state.predict(X))
        winner = min(outcomes, key=lambda outcome: outcome.score)
        replayed_winner = Network.fit(recipe, X, y, seed=winner.seed, max_iter=5)
        assert torch.equal(selected.predict(X), replayed_winner.predict(X))
        assert selected.diagnostics.loss_history == replayed_winner.diagnostics.loss_history

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_parallel_fit_matches_serial_and_retains_independent_inference_state(self, backend):
        X, y = blobs(3, 0.15)
        recipe = classification_recipe(K=3)
        options = dict(seed=12, n_inits=3, max_iter=4, retain="members")
        serial = Network.fit(recipe, X, y, **options)
        parallel = Network.fit(recipe, X, y, n_jobs=2, parallel_backend=backend, **options)
        assert torch.equal(serial.predict(X), parallel.predict(X))
        assert parallel.predict(X).dtype == DTYPE
        before = parallel.predict(X).clone()
        outcomes = parallel.diagnostics.initialisation_outcomes
        expected = serial.diagnostics.initialisation_outcomes
        assert outcomes is not None and expected is not None
        for outcome, reference, state, reference_state in zip(
            outcomes, expected, parallel.initial_states, serial.initial_states, strict=True
        ):
            assert (outcome.index, outcome.seed, outcome.score) == (
                reference.index,
                reference.seed,
                reference.score,
            )
            assert outcome.effective_backend == backend
            assert state.input_geometry.continuous_centroids.is_inference()
            assert torch.equal(
                state.input_geometry.continuous_centroids,
                reference_state.input_geometry.continuous_centroids,
            )
        assert torch.equal(serial.predict_all(X), parallel.predict_all(X))
        untouched = parallel.initial_states[1].input_geometry.continuous_centroids.clone()
        with torch.inference_mode():
            parallel.initial_states[0].input_geometry.continuous_centroids.zero_()
        assert torch.equal(
            parallel.initial_states[1].input_geometry.continuous_centroids, untouched
        )
        assert torch.equal(parallel.predict(X), before)

    def test_only_the_best_evaluation_is_kept_while_later_candidates_fit(self, monkeypatch):
        evaluate = selection._evaluate_candidate
        references, alive = [], []

        def recording(seed, **options):
            gc.collect()
            alive.append([reference() is not None for reference in references])
            evaluation = evaluate(seed, **options)
            references.append(weakref.ref(evaluation))
            return evaluation

        monkeypatch.setattr(selection, "_evaluate_candidate", recording)
        X, y = blobs(3, 0.1)
        scores = iter((1.0, 2.0, 3.0, 4.0))
        Network.fit(
            classification_recipe(K=3),
            X,
            y,
            n_inits=4,
            max_iter=3,
            selection_loss=lambda *args, **kwargs: next(scores),
        )
        # Candidate 0 stays the best; a later loser is released once the next one arrives.
        assert alive == [[], [True], [True, True], [True, False, True]]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("retain", ["winner", "states"])
    def test_supplied_state_retention_uses_the_computation_dtype(self, dtype, retain):
        X, y = blobs(2, 0.1)
        X_cat = (y,)
        source_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        recipe = classification_recipe()
        state = Network.initialise(recipe, X.to(source_dtype), y, X_cat=X_cat)
        network = Network.fit(
            recipe,
            X,
            y,
            X_cat=X_cat,
            initial_state=state,
            computation_dtype=dtype,
            retain=retain,
            max_iter=3,
        )
        outcomes = network.diagnostics.initialisation_outcomes
        assert outcomes is not None
        retained = network.initial_state
        assert retained.input_geometry.continuous_centroids.dtype == dtype
        assert retained.input_geometry.feature_weights.dtype == dtype
        assert all(
            centroids.dtype == dtype for centroids in retained.input_geometry.categorical_centroids
        )
        assert state.input_geometry.continuous_centroids.dtype == source_dtype
        replay = Network.fit(
            recipe, X.to(dtype), y, X_cat=X_cat, initial_state=retained, max_iter=3
        )
        assert torch.equal(
            network.predict(X.to(dtype), X_cat=X_cat), replay.predict(X.to(dtype), X_cat=X_cat)
        )

    def test_one_candidate_uses_root_seed_and_needs_no_parallel_dependency(self, without_joblib):
        X, y = blobs(2, 0.1)
        recipe = classification_recipe()
        one = Network.fit(
            recipe,
            X,
            y,
            seed=23,
            n_jobs=2,
            parallel_backend="processes",
            retain="states",
            max_iter=3,
        )
        outcomes = one.diagnostics.initialisation_outcomes
        assert outcomes is not None
        assert len(outcomes) == 1
        assert outcomes[0].seed == 23
        assert outcomes[0].effective_backend == "serial"
        state = Network.initialise(recipe, X, y, seed=23)
        replay = Network.fit(recipe, X, y, initial_state=state, max_iter=3)
        assert len(replay.diagnostics.initialisation_outcomes) == 1
        assert torch.equal(one.predict(X), replay.predict(X))
        with pytest.raises(ValueError, match=r"initial_state.*n_inits"):
            Network.fit(recipe, X, y, initial_state=state, n_inits=2)

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"n_inits": 0}, "n_inits"),
            ({"n_inits": True}, "n_inits"),
            ({"n_jobs": 0}, "n_jobs"),
            ({"n_jobs": True}, "n_jobs"),
            ({"n_jobs": 1.5}, "n_jobs"),
            ({"parallel_backend": "invalid"}, "parallel_backend"),
            ({"retain": True}, "retain"),
            ({"retain": "all"}, "retain"),
            ({"selection_loss": 1}, "selection_loss"),
            ({"validation_pairs": [()]}, "validation_pairs requires selection_loss"),
            ({"initial_state": object(), "init_rows": object()}, "mutually exclusive"),
            ({"initial_state": _DerivedState()}, "exact InitialState"),
        ],
    )
    def test_invalid_controls_fail_before_data_staging(self, options, message):
        with pytest.raises(ValueError, match=message):
            Network.fit(classification_recipe(), object(), object(), **options)


class TestSelectionLoss:
    def test_scores_calibrated_predictions_with_uncombined_weights(self):
        X, codes = blobs(3, 0.2)
        y = codes.clone()
        y[0] = -1
        sample_weights = torch.linspace(1.0, 4.0, len(y), device=DEVICE, dtype=DTYPE)
        class_weights = torch.tensor([1.0, 2.0, 3.0], device=DEVICE, dtype=DTYPE)
        labelled = y >= 0
        expected_target = torch.nn.functional.one_hot(y[labelled], 3).to(DTYPE)
        observed = []

        def selection_loss(prediction, target, **weights):
            assert torch.equal(target, expected_target)
            assert torch.equal(weights["sample_weights"], sample_weights[labelled])
            assert torch.equal(weights["class_weights"], class_weights)
            assert weights["task_weights"] is None
            assert weights["fold"] is None
            assert weights["partition"] == "training"
            observed.append(prediction.clone())
            return -prediction[:, 0].mean()

        recipe = classification_recipe(K=3)
        selected = Network.fit(
            recipe,
            X,
            y,
            seed=21,
            n_inits=3,
            max_iter=4,
            retain="members",
            sample_weights=sample_weights,
            class_weights=class_weights,
            selection_loss=selection_loss,
        )
        outcomes = selected.diagnostics.initialisation_outcomes
        assert outcomes is not None
        for outcome, prediction in zip(outcomes, observed, strict=True):
            replay = Network.fit(
                recipe,
                X,
                y,
                seed=outcome.seed,
                max_iter=4,
                sample_weights=sample_weights,
                class_weights=class_weights,
            )
            assert torch.equal(prediction, replay.predict(X)[labelled])
            assert outcome.score == float(-prediction[:, 0].mean())
            assert torch.equal(selected.predict_all(X)[outcome.index][labelled], prediction)
        winner = min(outcomes, key=lambda outcome: outcome.score)
        assert torch.equal(selected.predict(X)[labelled], observed[winner.index])

    @pytest.mark.parametrize(
        "config",
        [
            PredictConfig(output_mode="arithmetic"),
            PredictConfig(output_mode="geometric", epsilon_P=0.7),
        ],
    )
    def test_explicit_prediction_policy_is_used_before_scoring(self, config):
        X, y = blobs(2, 0.2)
        observed = []

        def loss(prediction, target, **weights):
            observed.append(prediction.clone())
            return 0.0

        network = Network.fit(
            classification_recipe(),
            X,
            y,
            n_inits=3,
            predict_config=config,
            selection_loss=loss,
            retain="members",
            max_iter=3,
        )
        outcomes = network.diagnostics.initialisation_outcomes
        assert outcomes is not None
        assert torch.equal(network.predict(X), observed[0])
        assert network.members is not None
        for member in network.members:
            assert member.predict_config == config

    @pytest.mark.parametrize(
        "backend,n_jobs", [("threads", None), ("threads", 2), ("processes", 2)]
    )
    def test_regression_weights_are_resolved_once_and_callback_minimises_its_loss(
        self, backend, n_jobs
    ):
        X, y, _ = regression_blobs(2)
        y[0] = float("nan")
        labelled = torch.isfinite(y).all(dim=1)
        sample_weights = torch.arange(1, len(y) + 1, dtype=DTYPE, device=DEVICE)
        calls = []
        expected_weights = torch.full((int(labelled.sum()),), 3.0, dtype=DTYPE, device=DEVICE)

        def task_weights(target):
            calls.append(target.clone())
            return expected_weights

        def loss(prediction, target, **weights):
            assert torch.equal(target, y[labelled])
            assert torch.equal(weights["sample_weights"], sample_weights[labelled])
            assert torch.equal(weights["task_weights"], expected_weights)
            assert weights["class_weights"] is None
            assert weights["fold"] is None
            assert weights["partition"] == "training"
            assert prediction.dtype == DTYPE
            assert prediction.is_inference()
            return ((prediction - target) ** 2).mean()

        options = dict(
            n_inits=3,
            max_iter=3,
            sample_weights=sample_weights,
            task_weights=task_weights,
            selection_loss=loss,
            retain="states",
        )
        network = Network.fit(
            regression_recipe(), X, y, n_jobs=n_jobs, parallel_backend=backend, **options
        )
        assert len(calls) == 1
        outcomes = network.diagnostics.initialisation_outcomes
        assert outcomes is not None
        winner = min(outcomes, key=lambda outcome: outcome.score)
        assert winner.score == float(((network.predict(X)[labelled] - y[labelled]) ** 2).mean())

    def test_soft_targets_and_callback_mutation_do_not_change_other_candidates(self):
        X, codes = blobs(2, 0.2)
        y = torch.nn.functional.one_hot(codes, 2).to(DTYPE) * 0.8 + 0.1
        weights = torch.ones(len(y), dtype=DTYPE, device=DEVICE)
        original_y = y.clone()

        def loss(prediction, target, **supplied):
            assert torch.equal(target, original_y)
            assert torch.equal(supplied["sample_weights"], weights)
            prediction.zero_()
            target.zero_()
            supplied["sample_weights"].zero_()
            return 0.0

        network = Network.fit(
            classification_recipe(),
            X,
            y,
            n_inits=3,
            sample_weights=weights,
            selection_loss=loss,
            retain="states",
            max_iter=3,
        )
        outcomes = network.diagnostics.initialisation_outcomes
        assert outcomes is not None
        replay = Network.fit(
            classification_recipe(),
            X,
            original_y,
            sample_weights=weights,
            seed=outcomes[0].seed,
            max_iter=3,
        )
        assert torch.equal(network.predict(X), replay.predict(X))
        assert torch.equal(y, original_y)
        assert torch.all(weights == 1)

    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            -float("inf"),
            True,
            [0.0],
            torch.tensor([0.0]),
            torch.tensor(True),
            torch.tensor(1j),
        ],
    )
    def test_non_finite_or_non_scalar_losses_are_rejected(self, value):
        X, y = blobs(2, 0.1)
        with pytest.raises(ValueError, match=r"selection_loss.*finite real scalar"):
            Network.fit(
                classification_recipe(),
                X,
                y,
                n_inits=2,
                max_iter=2,
                selection_loss=lambda *_args, **_kwargs: value,
            )

    def test_callback_exception_aborts_selection(self):
        X, y = blobs(2, 0.1)
        calls = []

        def loss(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("invalid metric")
            return 0.0

        with pytest.raises(RuntimeError, match="invalid metric"):
            Network.fit(classification_recipe(), X, y, n_inits=3, max_iter=2, selection_loss=loss)
        assert len(calls) == 2
