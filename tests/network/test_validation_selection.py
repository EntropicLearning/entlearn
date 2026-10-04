"""Explicit-fold selection through the fitted Network interface."""

from contextlib import nullcontext

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    blobs,
    classification_recipe,
    regression_blobs,
    regression_recipe,
    squared_error,
    validation_pairs,
)

from entlearn import Input, Network, PredictConfig


class TestValidationSelection:
    @pytest.mark.parametrize("with_folds", [False, True])
    def test_explicit_initialisation_rows_seed_only_the_requested_geometry(self, with_folds):
        X, y, _ = regression_blobs(2)
        recipe = regression_recipe()
        rows = torch.tensor([0, 4], dtype=torch.int64, device=DEVICE)
        selected = Network.fit(
            recipe,
            X,
            y,
            init_rows=rows,
            n_inits=3,
            max_iter=3,
            validation_pairs=validation_pairs() if with_folds else None,
            selection_loss=squared_error,
            retain="states",
        )
        for outcome, state in zip(
            selected.diagnostics.initialisation_outcomes, selected.initial_states, strict=True
        ):
            expected = Network.initialise(recipe, X[rows], y[rows], seed=outcome.seed)
            torch.testing.assert_close(
                state.input_geometry.continuous_centroids,
                expected.input_geometry.continuous_centroids,
            )
            assert state.connection_sub_seeds == expected.connection_sub_seeds

    @pytest.mark.parametrize(
        "pairs,message",
        [
            ([], "non-empty materialised"),
            ([(torch.tensor([0]),)], "training and validation indices"),
            ([(torch.tensor([[0]]), torch.tensor([1]))], "one-dimensional int64"),
            ([(torch.tensor([0.0]), torch.tensor([1]))], "one-dimensional int64"),
            ([(torch.tensor([True]), torch.tensor([1]))], "one-dimensional int64"),
            ([(torch.tensor([], dtype=torch.int64), torch.tensor([1]))], "indices in"),
            ([(torch.tensor([0, 0]), torch.tensor([1]))], "duplicate"),
            ([(torch.tensor([0, 1]), torch.tensor([1, 2]))], "disjoint"),
            ([(torch.tensor([-1]), torch.tensor([1]))], "indices in"),
            ([(torch.tensor([0]), torch.tensor([8]))], "indices in"),
        ],
    )
    def test_invalid_folds_fail_before_any_callback(self, pairs, message):
        X, y, _ = regression_blobs(2)
        pairs = [tuple(indices.to(DEVICE) for indices in pair) for pair in pairs]
        calls = []
        with pytest.raises(ValueError, match=rf"validation_pairs.*{message}"):
            Network.fit(
                regression_recipe(),
                X,
                y,
                validation_pairs=pairs,
                selection_loss=lambda *args, **kwargs: calls.append(1) or 0.0,
            )
        assert calls == []

    def test_validation_labels_do_not_change_a_fold_models_calibration(self):
        X, codes = blobs(2, 0.28, rows_per_blob=4)
        y = torch.nn.functional.one_hot(codes, 2).to(DTYPE) * 0.8 + 0.1
        recipe = classification_recipe(Input(K=3, epsilon=0.15))
        train, validation = validation_pairs()[0]
        observed = []

        def loss(prediction, target, **context):
            observed.append(prediction.clone())
            return (prediction - target).square().mean()

        options = dict(
            init_rows=train,
            validation_pairs=((train, validation),),
            selection_loss=loss,
            return_train_score=True,
            max_iter=5,
        )
        Network.fit(recipe, X, y, **options)
        original_records = observed.copy()
        observed.clear()
        changed = y.clone()
        changed[validation] = changed[validation].flip(dims=(1,))
        Network.fit(recipe, X, changed, **options)
        assert len(observed) == len(original_records) == 2
        for actual, expected in zip(observed, original_records, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        ("partition", "row_3", "row_7"),
        [
            pytest.param("validation", "label", "label", id="both-unlabelled"),
            pytest.param("training", "label", "task", id="unlabelled-and-zero-task"),
            pytest.param("validation", "label", "class", id="unlabelled-and-zero-class"),
            pytest.param("validation", None, None, id="scorable"),
        ],
    )
    def test_a_partition_needs_one_row_that_carries_every_weight_at_once(
        self, partition, row_3, row_7
    ):
        # Rows 3 and 7 form the named partition and every other row stays scorable. When the
        # two rows fail different conditions, each condition holds somewhere in the
        # partition, yet no single row can be scored.
        classification = row_7 == "class"
        if classification:
            X, codes = blobs(2, 0.25, rows_per_blob=4)
            y = torch.nn.functional.one_hot(codes, 2).to(DTYPE)
            # Row 7 carries only class 1, which this class weight removes.
            weights = {"class_weights": torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE)}
        else:
            X, y, _ = regression_blobs(2)
            # Task weights that are positive but never above one still count.
            weights = {"task_weights": torch.full((8,), 0.5, dtype=DTYPE, device=DEVICE)}
        for row, condition in ((3, row_3), (7, row_7)):
            if condition == "label":
                y[row] = 0 if classification else torch.nan
            elif condition == "task":
                weights["task_weights"][row] = 0
        train, validation = validation_pairs()[0]
        fold = (train, validation) if partition == "validation" else (validation, train)
        refusal = (
            nullcontext()
            if row_3 is None
            else pytest.raises(
                ValueError,
                match=rf"validation_pairs\[0\] {partition} needs positive-weight labelled rows",
            )
        )
        with refusal:
            Network.fit(
                classification_recipe() if classification else regression_recipe(),
                X,
                y,
                validation_pairs=(fold,),
                selection_loss=lambda *args, **kwargs: 0.0,
                max_iter=2,
                **weights,
            )

    @pytest.mark.parametrize("mode", ["single", "iterative"])
    def test_each_fold_and_full_data_member_match_independently_calibrated_fits(self, mode):
        X, codes = blobs(2, 0.25, rows_per_blob=4)
        y = torch.nn.functional.one_hot(codes, 2).to(DTYPE) * 0.8 + 0.1
        y[1] = 0
        labelled = y.sum(dim=1) > 0
        weights = torch.arange(1, 9, dtype=DTYPE, device=DEVICE)
        classes = torch.tensor([1.0, 2.0], dtype=DTYPE, device=DEVICE)
        recipe = classification_recipe(Input(K=3, epsilon=0.2))
        pairs = validation_pairs()
        policy = PredictConfig(predict_mode=mode, max_iter=2)
        observed = []

        def loss(prediction, target, *, fold, partition, sample_weights, class_weights, **other):
            observed.append((partition, prediction.clone()))
            # Selection need not use calibration's weighted log-loss.
            return (
                (prediction - target).square().sum(dim=1) * sample_weights
            ).sum() / sample_weights.sum()

        selected = Network.fit(
            recipe,
            X,
            y,
            n_inits=2,
            max_iter=5,
            sample_weights=weights,
            class_weights=classes,
            validation_pairs=pairs,
            predict_config=policy,
            selection_loss=loss,
            return_train_score=True,
            retain="members",
        )
        for index, member in enumerate(selected.members):
            for fold_index, (train, validation) in enumerate(pairs):
                fitted = Network.fit(
                    recipe,
                    X[train],
                    y[train],
                    sample_weights=weights[train],
                    class_weights=classes,
                    initial_state=member.initial_state,
                    max_iter=5,
                )
                for part_index, rows in enumerate((train, validation)):
                    prediction = fitted.predict(X[rows])[labelled[rows]]
                    torch.testing.assert_close(
                        prediction, observed[index * 4 + fold_index * 2 + part_index][1]
                    )
            replay = Network.fit(
                recipe,
                X,
                y,
                initial_state=member.initial_state,
                max_iter=5,
                sample_weights=weights,
                class_weights=classes,
                predict_config=policy,
            )
            assert member.predict_config == replay.predict_config
            torch.testing.assert_close(member.predict(X), replay.predict(X))

    @pytest.mark.parametrize(
        "backend,n_jobs", [("threads", None), ("threads", 2), ("processes", 2)]
    )
    @pytest.mark.parametrize("retain", ["winner", "states", "members"])
    def test_multiple_candidates_replay_each_geometry_and_retain_only_requested_values(
        self, backend, n_jobs, retain
    ):
        X, y, _ = regression_blobs(2)
        recipe = regression_recipe(K=3)
        pairs = validation_pairs()

        def loss(prediction, target, **context):
            return (prediction - target).square().mean()

        selected = Network.fit(
            recipe,
            X,
            y,
            n_inits=3,
            validation_pairs=pairs,
            selection_loss=loss,
            return_train_score=True,
            retain=retain,
            max_iter=3,
            n_jobs=n_jobs,
            parallel_backend=backend,
        )
        expected = []
        for outcome in selected.diagnostics.initialisation_outcomes:
            state = Network.initialise(recipe, X, y, seed=outcome.seed)
            losses, training_losses = [], []
            for train, validation in pairs:
                model = Network.fit(recipe, X[train], y[train], initial_state=state, max_iter=3)
                training_losses.append(float(loss(model.predict(X[train]), y[train])))
                losses.append(float(loss(model.predict(X[validation]), y[validation])))
            assert outcome.score == pytest.approx(sum(losses) / len(pairs))
            assert outcome.train_score == pytest.approx(sum(training_losses) / len(pairs))
            expected.append((outcome.score, outcome.score - outcome.train_score, outcome.index))
        index = min(expected)[2]
        winner = selected.diagnostics.initialisation_outcomes[index]
        assert selected.diagnostics.selected_outcome is winner
        replay = Network.fit(recipe, X, y, seed=winner.seed, max_iter=3)
        torch.testing.assert_close(selected.predict(X), replay.predict(X))
        assert (selected.initial_states is not None) == (retain != "winner")
        assert (selected.members is not None) == (retain == "members")
        if selected.initial_states is not None:
            assert selected.initial_state is selected.initial_states[index]
        if selected.members is not None:
            assert selected.initial_states is not None
            for member, state, outcome in zip(
                selected.members,
                selected.initial_states,
                selected.diagnostics.initialisation_outcomes,
                strict=True,
            ):
                assert member.initial_state is state
                assert member.diagnostics.selected_outcome == outcome
                assert member.members is None and member.initial_states is None
                member_replay = Network.fit(recipe, X, y, initial_state=state, max_iter=3)
                torch.testing.assert_close(member.predict(X), member_replay.predict(X))

    def test_exact_validation_ties_prefer_the_smaller_training_gap_then_original_order(self):
        X, y, _ = regression_blobs(2)
        # Per-fold training losses under a constant validation loss of 10. Candidate 1's
        # fold gaps cancel: the gap is a difference of means, not a mean of absolutes.
        loss, calls = tied_loss(((8.0, 10.0), (0.0, 20.0), (9.5, 10.5)))
        selected = Network.fit(
            regression_recipe(),
            X,
            y,
            n_inits=3,
            max_iter=3,
            validation_pairs=validation_pairs(),
            selection_loss=loss,
            return_train_score=True,
            retain="states",
        )
        outcomes = selected.diagnostics.initialisation_outcomes
        assert selected.initial_state is selected.initial_states[1]
        assert selected.diagnostics.selected_outcome is outcomes[1]
        assert [item.score for item in outcomes] == [10.0] * 3
        assert [item.train_score for item in outcomes] == [9.0, 10.0, 10.0]
        assert calls == {"training": 6, "validation": 6}
        continued = selected.resume(X, y, max_iter=1)
        assert continued.diagnostics.initialisation_outcomes == outcomes
        assert continued.diagnostics.selected_outcome == outcomes[1]

    def test_the_tie_key_is_the_gap_that_a_saved_model_recomputes(self, tmp_path):
        X, y, _ = regression_blobs(2)
        # The training scores differ, but both gaps round to exactly 10.0. Ranking by
        # training score alone would pick candidate 1, which loading then rejects.
        loss, _ = tied_loss(((1e-17, 1e-17), (2e-17, 2e-17)))
        selected = Network.fit(
            regression_recipe(),
            X,
            y,
            n_inits=2,
            max_iter=3,
            validation_pairs=validation_pairs(),
            selection_loss=loss,
            return_train_score=True,
        )
        outcomes = selected.diagnostics.initialisation_outcomes
        assert outcomes[0].train_score < outcomes[1].train_score
        assert [item.score - item.train_score for item in outcomes] == [10.0, 10.0]
        assert selected.diagnostics.selected_outcome is outcomes[0]
        path = tmp_path / "model.safetensors"
        selected.save(path)
        assert Network.load(path).diagnostics.selected_outcome == outcomes[0]

    def test_without_training_scores_exact_ties_go_to_original_order(self):
        X, y, _ = regression_blobs(2)
        loss, calls = tied_loss(((8.0, 10.0), (0.0, 20.0), (9.5, 10.5)))
        selected = Network.fit(
            regression_recipe(),
            X,
            y,
            n_inits=3,
            max_iter=3,
            validation_pairs=validation_pairs(),
            selection_loss=loss,
            retain="states",
        )
        outcomes = selected.diagnostics.initialisation_outcomes
        assert selected.initial_state is selected.initial_states[0]
        assert selected.diagnostics.selected_outcome is outcomes[0]
        assert [item.train_score for item in outcomes] == [None] * 3
        assert calls == {"training": 0, "validation": 6}

    def test_the_flag_is_a_boolean_and_records_nothing_in_sample(self):
        X, y, _ = regression_blobs(2)
        with pytest.raises(ValueError, match="return_train_score must be a boolean"):
            Network.fit(regression_recipe(), X, y, return_train_score=1)
        selected = Network.fit(
            regression_recipe(), X, y, n_inits=2, max_iter=3, return_train_score=True
        )
        outcomes = selected.diagnostics.initialisation_outcomes
        assert [item.train_score for item in outcomes] == [None, None]

    def test_one_shared_state_scores_each_partition_and_refits_on_all_rows(self):
        X, y, _ = regression_blobs(2)
        recipe = regression_recipe()
        pairs = validation_pairs()
        state = Network.initialise(recipe, X, y)
        observed = []

        def loss(prediction, target, *, fold, partition, **weights):
            train, validation = fold
            rows = train if partition == "training" else validation
            assert torch.equal(target, y[rows])
            fold_fit = Network.fit(recipe, X[train], y[train], initial_state=state, max_iter=3)
            torch.testing.assert_close(prediction, fold_fit.predict(X[rows]))
            assert all(value is None for value in weights.values())
            observed.append((partition, len(rows)))
            return float(len(rows))

        selected = Network.fit(
            recipe,
            X,
            y,
            validation_pairs=pairs,
            selection_loss=loss,
            return_train_score=True,
            max_iter=3,
        )
        assert sorted(observed) == [
            ("training", 4),
            ("training", 6),
            ("validation", 2),
            ("validation", 4),
        ]
        # Equal fold influence, despite different row counts.
        assert selected.diagnostics.initialisation_outcomes[0].score == 3.0
        assert selected.diagnostics.initialisation_outcomes[0].train_score == 5.0
        replay = Network.fit(recipe, X, y, initial_state=state, max_iter=3)
        torch.testing.assert_close(selected.predict(X), replay.predict(X))
        assert selected.diagnostics.loss_history == replay.diagnostics.loss_history

    def test_classification_weights_not_supplied_reach_the_callback_as_none(self):
        X, y = blobs(2, 0.25, rows_per_blob=4)
        observed = []

        def loss(prediction, target, *, fold, partition, **weights):
            observed.append(weights)
            return 0.0

        Network.fit(classification_recipe(), X, y, selection_loss=loss, max_iter=2)
        assert observed == [dict(sample_weights=None, class_weights=None, task_weights=None)]

    def test_validation_requires_an_explicit_callback(self):
        X, y, _ = regression_blobs(2)
        with pytest.raises(ValueError, match=r"validation.*selection_loss"):
            Network.fit(regression_recipe(), X, y, validation_pairs=validation_pairs())


def tied_loss(training_losses):
    """Return a constant validation loss and per-candidate, per-fold training losses."""
    calls = {"training": 0, "validation": 0}

    def loss(prediction, target, *, partition, **context):
        count = calls[partition]
        calls[partition] += 1
        if partition == "validation":
            return 10.0
        return training_losses[count // 2][count % 2]

    return loss, calls
