"""Validation selection across tensor modalities and caller-controlled failures."""

import logging
import warnings
from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    blobs,
    regression_blobs,
    regression_recipe,
    squared_error,
    validation_pairs,
)

from entlearn import (
    ClassificationHead,
    ConvergenceWarning,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    Network,
    PredictConfig,
    Recipe,
    RegressionHead,
)


class TestValidationContract:
    def test_callback_failure_stops_before_the_next_fold_fit(self, caplog):
        X, y = blobs(2, 0.25, rows_per_blob=4)
        recipe = Recipe.chain(Input(K=3, epsilon=0.2), ClassificationHead(coupling=Coupling.M))
        logger = logging.getLogger("validation-callback-failure")
        caplog.set_level(logging.INFO, logger=logger.name)

        def loss(prediction, target, **context):
            completed = [
                record for record in caplog.records if record.message.startswith("fit finished:")
            ]
            assert len(completed) == 1
            raise RuntimeError("selection metric failed")

        with pytest.raises(RuntimeError, match="selection metric failed"):
            Network.fit(
                recipe,
                X,
                y,
                validation_pairs=validation_pairs(),
                selection_loss=loss,
                max_iter=3,
                logger=logger,
                verbose=1,
            )

    @pytest.mark.parametrize("mode", ["geometric", "arithmetic", "regression"])
    def test_iterative_deployment_preserves_single_pass_selection_and_temperature(self, mode):
        X, codes = blobs(2, 0.25, rows_per_blob=4)
        target = codes.to(DTYPE) if mode == "regression" else codes
        head = RegressionHead() if mode == "regression" else ClassificationHead(coupling=Coupling.M)
        recipe = Recipe.chain(Input(K=3, epsilon=0.2), head)
        observed = []

        def loss(prediction, target, **context):
            observed.append(prediction.clone())
            return (prediction - target).square().mean()

        options = dict(
            validation_pairs=validation_pairs(),
            selection_loss=loss,
            n_inits=2,
            max_iter=5,
            retain="members",
        )
        config = PredictConfig(output_mode=None if mode == "regression" else mode, max_iter=2)
        single = Network.fit(recipe, X, target, predict_config=config, **options)
        single_records = observed.copy()
        observed.clear()
        iterative = Network.fit(
            recipe,
            X,
            target,
            predict_config=replace(config, predict_mode="iterative"),
            **options,
        )
        assert (
            single.diagnostics.initialisation_outcomes
            == iterative.diagnostics.initialisation_outcomes
        )
        assert len(observed) == len(single_records)
        for actual, expected in zip(observed, single_records, strict=True):
            torch.testing.assert_close(actual, expected)
        assert single.predict_config.epsilon_P == iterative.predict_config.epsilon_P
        assert iterative.predict_config.predict_mode == "iterative"
        assert iterative.members is not None
        for member in iterative.members:
            replay = Network.fit(
                recipe,
                X,
                target,
                initial_state=member.initial_state,
                predict_config=member.predict_config,
                max_iter=5,
            )
            torch.testing.assert_close(member.predict(X), replay.predict(X))
            details = member.predict_with_details(X, details=("diagnostics",))
            assert details.n_iter > 0
            assert details.n_iter == len(details.loss_history)
            assert isinstance(details.converged, bool)
            torch.testing.assert_close(details.prediction, member.predict(X))

    def test_positive_class_support_survives_weight_product_underflow(self):
        X, _, _ = regression_blobs(2)
        X = X.to(torch.float32)
        target = torch.tensor([1e-20, 1.0], dtype=torch.float32, device=DEVICE).repeat(8, 1)
        class_weights = torch.tensor([1e-30, 0.0], dtype=torch.float32, device=DEVICE)
        recipe = Recipe.chain(Input(K=2), ClassificationHead(coupling=Coupling.M))
        fitted = Network.fit(
            recipe,
            X,
            target,
            class_weights=class_weights,
            validation_pairs=validation_pairs(),
            max_iter=3,
            selection_loss=squared_error,
        )
        assert torch.isfinite(fitted.predict(X)).all()

    def test_regression_row_weights_survive_weight_product_underflow(self):
        X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32, device=DEVICE)
        target = torch.tensor([float("nan"), 1.0, 4.0], dtype=torch.float32, device=DEVICE)
        sample_weights = torch.tensor([1.0, 1e-30, 1e-30], dtype=torch.float32, device=DEVICE)
        task_weights = torch.tensor([1.0, 1e-20, 2e-20], dtype=torch.float32, device=DEVICE)
        recipe = Recipe.chain(Input(K=1), RegressionHead())

        fitted = Network.fit(
            recipe,
            X,
            target,
            sample_weights=sample_weights,
            task_weights=task_weights,
            max_iter=3,
            seed=0,
        )

        # Each labelled row's sample-task product underflows to zero in float32, so only
        # weights recovered in the logarithmic domain carry the intended 1:2 ratio. A
        # single cluster gathers both labelled rows into one output centroid, making the
        # prediction their row-weight-weighted mean.
        expected = torch.full(
            (3, 1),
            1.0 / 3.0 * 1.0 + 2.0 / 3.0 * 4.0,
            dtype=torch.float32,
            device=DEVICE,
        )
        torch.testing.assert_close(fitted.predict(X), expected)

    @pytest.mark.parametrize("with_folds", [False, True])
    @pytest.mark.parametrize("override", [False, True])
    @pytest.mark.parametrize("field", ["continuous_centroids", "feature_weights"])
    def test_supplied_state_is_validated_before_any_dtype_conversion(
        self, with_folds, override, field
    ):
        X, y, _ = regression_blobs(2)
        X = X.to(torch.float64)
        recipe = regression_recipe()
        original = Network.initialise(recipe, X, y)
        state = replace(
            original,
            input_geometry=replace(
                original.input_geometry,
                **{field: getattr(original.input_geometry, field).to(torch.float32)},
            ),
        )
        options = dict(
            initial_state=state,
            computation_dtype=torch.float64 if override else None,
            validation_pairs=validation_pairs() if with_folds else None,
            selection_loss=squared_error,
            max_iter=3,
        )
        if override:
            fitted = Network.fit(recipe, X, y, **options)
            assert getattr(fitted.initial_state.input_geometry, field).dtype == torch.float64
            assert getattr(state.input_geometry, field).dtype == torch.float32
        else:
            with pytest.raises(ValueError, match="InitialState"):
                Network.fit(recipe, X, y, **options)

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_parallel_calibration_and_retained_members_match_serial(self, backend):
        X, codes = blobs(2, 0.25, rows_per_blob=4)
        recipe = Recipe.chain(Input(K=3, epsilon=0.2), ClassificationHead(coupling=Coupling.M))
        options = dict(
            n_inits=3,
            max_iter=5,
            retain="members",
            validation_pairs=validation_pairs(),
            selection_loss=squared_error,
        )
        serial = Network.fit(recipe, X, codes, **options)
        parallel = Network.fit(recipe, X, codes, n_jobs=2, parallel_backend=backend, **options)
        torch.testing.assert_close(serial.predict_all(X), parallel.predict_all(X))
        assert serial.predict_config == parallel.predict_config
        for a, b in zip(
            serial.diagnostics.initialisation_outcomes,
            parallel.diagnostics.initialisation_outcomes,
            strict=True,
        ):
            assert (a.index, a.seed, a.score) == (b.index, b.seed, b.score)
            assert b.effective_backend == backend
        assert parallel.initial_states is not None
        assert all(
            state.input_geometry.continuous_centroids.is_inference()
            for state in parallel.initial_states
        )

    def test_callable_task_weights_are_resolved_once_before_fold_dispatch(self):
        X, y, _ = regression_blobs(2)
        y[1] = torch.nan
        labelled = torch.isfinite(y).all(dim=1)
        calls = []
        weights = torch.arange(1, 8, device=DEVICE, dtype=DTYPE)
        all_weights = torch.ones(8, device=DEVICE, dtype=DTYPE)
        all_weights[labelled] = weights

        def weighting(target):
            calls.append(target.clone())
            return weights

        def loss(prediction, target, *, fold, partition, task_weights, **context):
            rows = fold[0] if partition == "training" else fold[1]
            rows = rows[labelled[rows]]
            assert torch.equal(task_weights, all_weights[rows])
            return (prediction - target).square().mean()

        Network.fit(
            regression_recipe(),
            X,
            y,
            task_weights=weighting,
            n_inits=2,
            n_jobs=2,
            validation_pairs=validation_pairs(),
            selection_loss=loss,
            return_train_score=True,
            max_iter=3,
        )
        assert len(calls) == 1
        assert torch.equal(calls[0], y[labelled])

    def test_fold_training_rows_restage_missing_targets_and_task_weights(self):
        """A fold's own staging keeps its unlabelled rows unlabelled and its row weights."""
        X = torch.linspace(0, 1, 8, dtype=DTYPE, device=DEVICE)[:, None]
        y = torch.tensor(
            [2.0, 4.0, 6.0, 8.0, 10.0, torch.nan, 14.0, 16.0], dtype=DTYPE, device=DEVICE
        )
        sample_weights = torch.arange(1.0, 9.0, dtype=DTYPE, device=DEVICE)
        task_weights = torch.tensor(
            [1.0, 2.0, 4.0, 8.0, 16.0, 1.0, 32.0, 64.0], dtype=DTYPE, device=DEVICE
        )
        labelled = torch.isfinite(y)
        widths = []

        def loss(prediction, target, *, fold, partition, **context):
            if partition == "training":
                # A single cluster predicts the weighted mean of its own labelled fit rows.
                rows = fold[0][labelled[fold[0]]]
                combined = sample_weights[rows] * task_weights[rows]
                expected = (y[rows] * combined).sum() / combined.sum()
                torch.testing.assert_close(prediction[:, 0], expected.expand(len(rows)))
                widths.append(len(rows))
            return float((prediction - target).square().mean())

        Network.fit(
            regression_recipe(K=1),
            X,
            y,
            sample_weights=sample_weights,
            task_weights=task_weights,
            validation_pairs=validation_pairs(),
            selection_loss=loss,
            return_train_score=True,
            max_iter=1,
        )
        assert widths == [5, 4]

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["continuous", "categorical", "distribution", "manifold"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_supported_modalities_preserve_schema_weights_and_original_state(
        self, task, kind, coupling
    ):
        X, codes = blobs(2, 0.2, rows_per_blob=4)
        categorical = None
        if kind == "categorical":
            # The last validation row has a category absent from one training partition.
            categorical = (torch.tensor([0, 0, 0, 1, 1, 1, 1, 2], device=DEVICE),)
        elif kind == "distribution":
            categorical = (torch.nn.functional.one_hot(codes, 2).to(DTYPE) * 0.8 + 0.1,)
        if categorical is not None:
            X = X[:, :0]
        input_block = (
            ManifoldInput(K=2, subspace_dimension=1, epsilon=0.2)
            if kind == "manifold"
            else Input(K=2, epsilon=0.2)
        )
        head = (
            ClassificationHead(coupling=coupling) if task == "classification" else RegressionHead()
        )
        recipe = Recipe.chain(input_block, Hidden(K=2, epsilon=0.2), head, coupling=coupling)
        y = codes.clone() if task == "classification" else codes.to(DTYPE)[:, None]
        y[1] = -1 if task == "classification" else torch.nan
        labelled = codes >= 0
        labelled[1] = False
        sample_weights = torch.arange(1, 9, dtype=DTYPE, device=DEVICE)
        task_weights = torch.linspace(0.5, 2.0, 8, dtype=DTYPE, device=DEVICE)
        class_weights = torch.tensor([2.0, 1.0], dtype=DTYPE, device=DEVICE)
        options = dict(
            sample_weights=sample_weights,
            class_weights=class_weights if task == "classification" else None,
            task_weights=task_weights if task == "regression" else None,
            X_cat=categorical,
        )

        def loss(prediction, target, *, fold, partition, **weights):
            rows = fold[0] if partition == "training" else fold[1]
            rows = rows[labelled[rows]]
            expected = (
                torch.nn.functional.one_hot(codes[rows], 2).to(DTYPE)
                if task == "classification"
                else y[rows]
            )
            assert torch.equal(target, expected)
            assert torch.equal(weights["sample_weights"], sample_weights[rows])
            if task == "regression":
                assert torch.equal(weights["task_weights"], task_weights[rows])
            else:
                assert torch.equal(weights["class_weights"], class_weights)
            return (prediction - target).square().mean()

        selected = Network.fit(
            recipe,
            X,
            y,
            **options,
            validation_pairs=validation_pairs(),
            selection_loss=loss,
            return_train_score=True,
            n_inits=2,
            retain="members",
            max_iter=3,
        )
        expected_width = 2 if task == "classification" else 1
        assert expected_width == selected.schema.M
        if kind == "categorical":
            assert selected.schema.M_cat == (3,)
        assert selected.members is not None
        for member in selected.members:
            replay = Network.fit(
                recipe,
                X,
                y,
                **options,
                initial_state=member.initial_state,
                predict_config=member.predict_config,
                max_iter=3,
            )
            torch.testing.assert_close(
                member.predict(X, X_cat=categorical), replay.predict(X, X_cat=categorical)
            )

    @pytest.mark.parametrize(
        "config", [PredictConfig(epsilon_P=0.7), PredictConfig(output_mode="arithmetic")]
    )
    def test_fixed_classification_policy_is_not_calibrated(self, config):
        X, codes = blobs(2, 0.2, rows_per_blob=4)
        recipe = Recipe.chain(Input(K=2), ClassificationHead(coupling=Coupling.M))
        selected = Network.fit(
            recipe,
            X,
            codes,
            validation_pairs=validation_pairs(),
            predict_config=config,
            selection_loss=squared_error,
            n_inits=2,
            retain="members",
            max_iter=3,
        )
        assert selected.members is not None
        for member in selected.members:
            assert member.predict_config.epsilon_P == config.epsilon_P

    def test_callback_mutation_cannot_change_fold_definitions_or_the_refit(self):
        X, y, _ = regression_blobs(2)
        pairs = validation_pairs()
        original = tuple((train.clone(), validation.clone()) for train, validation in pairs)
        recipe = regression_recipe()
        sample_weights = torch.arange(1, 9, dtype=DTYPE, device=DEVICE)

        def loss(prediction, target, *, fold, partition, **weights):
            prediction.zero_()
            target.zero_()
            weights["sample_weights"].zero_()
            for indices in fold:
                indices.zero_()
            return 0.0

        selected = Network.fit(
            recipe,
            X,
            y,
            validation_pairs=pairs,
            selection_loss=loss,
            sample_weights=sample_weights,
            retain="states",
            n_inits=2,
            max_iter=3,
        )
        for actual, expected in zip(pairs, original, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert selected.initial_state is selected.initial_states[0]
        replay = Network.fit(
            recipe,
            X,
            y,
            initial_state=selected.initial_state,
            sample_weights=sample_weights,
            max_iter=3,
        )
        torch.testing.assert_close(selected.predict(X), replay.predict(X))

    @pytest.mark.parametrize("partition", ["training", "validation"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True, torch.tensor([1.0])])
    def test_invalid_callback_result_aborts_selection(self, partition, value):
        X, y, _ = regression_blobs(2)

        def loss(*args, **context):
            return value if context["partition"] == partition else 0.0

        with pytest.raises(ValueError, match="finite real scalar"):
            Network.fit(
                regression_recipe(),
                X,
                y,
                validation_pairs=validation_pairs(),
                selection_loss=loss,
                return_train_score=True,
                max_iter=3,
            )

    @pytest.mark.parametrize("retain,refits", [("winner", 1), ("states", 1), ("members", 3)])
    @pytest.mark.parametrize(
        "n_jobs,backend", [(None, "threads"), (2, "threads"), (2, "processes")]
    )
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_fit_cost_and_non_convergence_are_reported_once(
        self, caplog, retain, refits, n_jobs, backend, task
    ):
        X, codes = blobs(2, 0.25, rows_per_blob=4)
        y = codes if task == "classification" else codes.to(DTYPE)
        head = (
            ClassificationHead(coupling=Coupling.M)
            if task == "classification"
            else RegressionHead()
        )
        recipe = Recipe.chain(Input(K=3, epsilon=0.2), head)
        logger = logging.getLogger("validation-fit-test")
        caplog.set_level(logging.INFO, logger=logger.name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            selected = Network.fit(
                recipe,
                X,
                y,
                n_inits=3,
                validation_pairs=validation_pairs(),
                selection_loss=squared_error,
                retain=retain,
                n_jobs=n_jobs,
                parallel_backend=backend,
                max_iter=1,
                tol=0.0,
                logger=logger,
                verbose=1,
            )
        summaries = [
            record for record in caplog.records if record.message.startswith("fit finished:")
        ]
        assert len(summaries) == 3 * 2 + refits
        failures = [warning for warning in caught if warning.category is ConvergenceWarning]
        assert len(failures) == 1
        message = str(failures[0].message)
        assert "fold 0" in message and "full-data refit" in message
        assert all(f"init {index}" in message for index in range(3))
        assert message in selected.diagnostics.warnings

    def test_warning_as_error_prevents_publication(self):
        X, y, _ = regression_blobs(2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(ConvergenceWarning, match="fold"):
                Network.fit(
                    regression_recipe(),
                    X,
                    y,
                    validation_pairs=validation_pairs(),
                    selection_loss=lambda *args, **kwargs: 0.0,
                    max_iter=1,
                    tol=0.0,
                )

    @pytest.mark.parametrize("rows", [[], [-1, 4], [0, 8], [0, 0]])
    def test_invalid_initialisation_rows_raise(self, rows):
        X, y, _ = regression_blobs(2)
        with pytest.raises(ValueError):
            Network.fit(
                regression_recipe(),
                X,
                y,
                init_rows=torch.tensor(rows, dtype=torch.int64, device=DEVICE),
            )

    def test_one_initialisation_row_caps_the_geometry_and_still_fits_all_rows(self):
        X, y, _ = regression_blobs(2)
        with pytest.warns(UserWarning, match=r"K=2.*1.*reducing"):
            fitted = Network.fit(
                regression_recipe(),
                X,
                y,
                init_rows=torch.tensor([0], dtype=torch.int64, device=DEVICE),
            )
        assert fitted.initial_state.input_geometry.K_active == 1
        torch.testing.assert_close(fitted.initial_state.input_geometry.continuous_centroids, X[:1])
        torch.testing.assert_close(fitted.predict(X), y.mean(dim=0).expand_as(y))
