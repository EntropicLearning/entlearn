"""Retained fit warnings and the absence of alternate empty-cluster policies."""

from __future__ import annotations

import io
import logging
import warnings

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    collapsed_reference_model,
    connection_initialisation_data,
    deep_classification_recipe,
    diverging_regression,
)

from entlearn import (
    ClassificationHead,
    ConvergenceWarning,
    Coupling,
    Input,
    LossIncreaseWarning,
    Network,
    Recipe,
)
from entlearn.network import fit as fit_ops


class TestFitWarnings:
    @pytest.mark.parametrize("n_inits", [1, 3])
    @pytest.mark.parametrize("max_iter", [1, 2])
    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    def test_non_converged_fits_warn_once_before_publication(self, n_inits, max_iter, execution):
        X, y = connection_initialisation_data()
        with pytest.warns(ConvergenceWarning, match="did not converge") as seen:
            model = Network.fit(
                deep_classification_recipe((4, 2)),
                X,
                y,
                n_inits=n_inits,
                max_iter=max_iter,
                tol=0,
                n_jobs=execution[0],
                parallel_backend=execution[1],
            )
        assert len(seen) == 1
        assert not model.diagnostics.converged
        message = str(seen[0].message)
        assert message in model.diagnostics.warnings
        for record in model.diagnostics.initialisation_outcomes:
            assert f"init {record.index} (seed={record.seed})" in message

    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    @pytest.mark.parametrize("retain", ["winner", "members"])
    def test_converged_winner_still_reports_non_converged_losers(self, execution, retain):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((4, 2))
        with pytest.warns(ConvergenceWarning):
            reference = Network.fit(
                recipe, X, y, n_inits=4, max_iter=3, tol=0.01, seed=1, retain="members"
            )
        assert reference.members is not None
        converged = [member.diagnostics.converged for member in reference.members]
        assert any(converged) and not all(converged)
        with pytest.warns(ConvergenceWarning, match="did not converge") as seen:
            model = Network.fit(
                recipe,
                X,
                y,
                n_inits=4,
                max_iter=3,
                tol=0.01,
                seed=1,
                n_jobs=execution[0],
                parallel_backend=execution[1],
                retain=retain,
            )
        assert len(seen) == 1
        assert model.diagnostics.converged
        message = str(seen[0].message)
        for index in range(4):
            assert (f"init {index} (" in message) is not converged[index]
        if retain == "members":
            assert model.members is not None
            assert [member.diagnostics.converged for member in model.members] == converged
        assert message in model.diagnostics.warnings

    @pytest.mark.parametrize("n_inits", [1, 3])
    def test_warning_as_error_preserves_the_previous_network(self, n_inits):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((4, 2))
        model = Network.fit(recipe, X, y, tol=0.01, max_iter=100)
        previous = model
        prediction = model.predict(X)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(ConvergenceWarning):
                model = Network.fit(recipe, X, y, n_inits=n_inits, max_iter=1, tol=0)
        assert model is previous
        torch.testing.assert_close(model.predict(X), prediction)

    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    def test_lifecycle_warnings_are_attributed_to_the_calling_line(self, execution):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((4, 2))
        diverging, X_large, y_large = diverging_regression()
        collapsed = collapsed_reference_model()
        query = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        dispatch = {"n_inits": 2, "n_jobs": execution[0], "parallel_backend": execution[1]}
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            # The process pool reports its own restarted workers; that notice is not ours.
            warnings.filterwarnings("ignore", message="A worker stopped")
            network = Network.fit(recipe, X, y, max_iter=1, tol=0, **dispatch)
            network.resume(X, y, max_iter=2, tol=0)
            network.fine_tune(X, y, max_iter=1, tol=0)
            Network.fit(diverging, X_large, y_large, max_iter=2, **dispatch)
            collapsed.score_samples(query)
        assert [record.category for record in seen[:3]] == [ConvergenceWarning] * 3
        assert LossIncreaseWarning in {record.category for record in seen[3:-1]}
        assert "collapsed" in str(seen[-1].message)
        assert [record.filename for record in seen] == [__file__] * len(seen)

    def test_converged_candidates_emit_no_convergence_warning(self):
        X, y = connection_initialisation_data()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            model = Network.fit(deep_classification_recipe((4, 2)), X, y, n_inits=3, tol=0.01)
        assert model.diagnostics.converged
        assert seen == []

    def test_width_one_hidden_block_warns_once_and_records_the_warning(self):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((2, 1))
        with pytest.warns(UserWarning, match="hidden block 'hidden_1' has K = 1") as seen:
            network = Network.fit(recipe, X, y)
        assert len(seen) == 1
        assert network.diagnostics.warnings == (str(seen[0].message),)
        assert torch.isfinite(network.predict(X)).all()

    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    def test_a_candidate_warning_reaches_the_caller_on_every_backend(self, execution):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((2, 1))
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            network = Network.fit(
                recipe,
                X,
                y,
                n_inits=2,
                n_jobs=execution[0],
                parallel_backend=execution[1],
            )
        records = [
            record for record in seen if "hidden block 'hidden_1' has K = 1" in str(record.message)
        ]
        assert len(records) == 2
        assert all(record.category is UserWarning for record in records)
        assert network.diagnostics.warnings == (str(records[0].message),)

    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    def test_an_initialisation_warning_reaches_the_calling_line_on_every_backend(self, execution):
        X, y = connection_initialisation_data()
        recipe = Recipe.chain(Input(K=12), ClassificationHead(Coupling.M))
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message="A worker stopped")
            Network.fit(
                recipe,
                X,
                y,
                n_inits=2,
                max_iter=2,
                n_jobs=execution[0],
                parallel_backend=execution[1],
            )
        capped = [record for record in seen if "reducing the initial input" in str(record.message)]
        assert len(capped) == 2
        assert all(record.category is UserWarning for record in capped)
        assert [record.filename for record in seen] == [__file__] * len(seen)

    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    def test_a_candidate_warning_keeps_its_category_and_the_calling_line(self, execution):
        X, y = connection_initialisation_data()

        def selection_loss(prediction, target, **_):
            warnings.warn("selection loss checked", RuntimeWarning, stacklevel=1)
            return float(((prediction - target) ** 2).mean())

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message="A worker stopped")
            Network.fit(
                deep_classification_recipe((4, 2)),
                X,
                y,
                n_inits=3,
                tol=0.01,
                selection_loss=selection_loss,
                n_jobs=execution[0],
                parallel_backend=execution[1],
            )
        checked = [record for record in seen if "selection loss checked" in str(record.message)]
        assert [record.category for record in checked] == [RuntimeWarning] * 3
        assert [record.filename for record in checked] == [__file__] * 3

    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    def test_a_candidate_warning_as_error_aborts_publication_on_every_backend(self, execution):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((2, 1))
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="hidden block")
            with pytest.raises(UserWarning, match="hidden block 'hidden_1' has K = 1"):
                Network.fit(
                    recipe,
                    X,
                    y,
                    n_inits=2,
                    n_jobs=execution[0],
                    parallel_backend=execution[1],
                )

    def test_loss_increase_is_promoted_to_an_error_by_pytest(self):
        with pytest.raises(LossIncreaseWarning, match="unexpected increase"):
            warnings.warn("unexpected increase", LossIncreaseWarning, stacklevel=2)

    @pytest.mark.parametrize(
        "controls",
        [
            {"starved_tol": 0.0},
            {"move_retry_tol": 0.0},
            {"on_empty": "reseed"},
            {"on_empty": "split"},
            {"gamma_first": True},
        ],
    )
    def test_removed_empty_cluster_and_order_controls_are_rejected(self, controls):
        X, y = connection_initialisation_data()
        with pytest.raises(TypeError, match="unexpected keyword"):
            Network.fit(deep_classification_recipe((2, 2)), X, y, **controls)

    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_ordinary_fits_do_not_emit_removed_initialisation_or_revival_warnings(self, coupling):
        X, y = connection_initialisation_data()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            network = Network.fit(deep_classification_recipe((4, 2), coupling=coupling), X, y)
        assert seen == []
        assert network.diagnostics.warnings == ()


class TestFitLogging:
    @pytest.mark.parametrize("verbose", [0, 1, 2])
    @pytest.mark.parametrize("n_inits", [1, 2, 3])
    @pytest.mark.parametrize("execution", [(1, "threads"), (2, "threads"), (2, "processes")])
    def test_injected_logger_obeys_network_verbosity_without_reconfiguration(
        self, verbose, n_inits, execution, caplog
    ):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((2, 2))
        stream = io.StringIO()
        logger = logging.Logger("caller", level=logging.DEBUG)  # noqa: LOG001
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        before = (logger.level, tuple(logger.handlers), logger.propagate, logger.disabled)
        with caplog.at_level(logging.DEBUG):
            Network.fit(
                recipe,
                X,
                y,
                max_iter=2,
                tol=0.0,
                logger=logger,
                verbose=verbose,
                n_inits=n_inits,
                n_jobs=execution[0],
                parallel_backend=execution[1],
            )
        assert caplog.records == []
        output = stream.getvalue()
        assert ("fit finished" in output) is (verbose >= 1)
        assert output.count("fit finished") == (n_inits if verbose >= 1 else 0)
        assert ("iter 1: loss" in output) is (verbose >= 2)
        assert ("multi-init: selected init" in output) is (verbose >= 1 and n_inits > 1)
        assert ("multi-init: per-init scores" in output) is (verbose >= 1 and n_inits > 1)
        assert (logger.level, tuple(logger.handlers), logger.propagate, logger.disabled) == before
        # A later silent fit must not inherit the preceding Network's verbosity.
        Network.fit(recipe, X, y, max_iter=1, logger=logger)
        assert stream.getvalue() == output

    @pytest.mark.parametrize("n_inits", [1, 3])
    def test_default_verbose_logging_is_private_and_does_not_leak(self, capsys, caplog, n_inits):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((2, 2))
        loggers = [logging.getLogger(), logging.getLogger("entlearn")]
        before = [(log.level, tuple(log.handlers), log.propagate) for log in loggers]
        with caplog.at_level(logging.INFO):
            Network.fit(recipe, X, y, max_iter=1, verbose=2, n_inits=n_inits)
        assert caplog.records == []
        assert "iter 1: loss" in capsys.readouterr().err
        with caplog.at_level(logging.INFO):
            Network.fit(recipe, X, y, max_iter=1, n_inits=n_inits)
        assert caplog.records == []
        assert capsys.readouterr().err == ""
        assert [(log.level, tuple(log.handlers), log.propagate) for log in loggers] == before

    @pytest.mark.parametrize("verbose", [-1, True, 1.5, "debug"])
    def test_invalid_verbosity_is_rejected(self, verbose):
        X, y = connection_initialisation_data()
        with pytest.raises(ValueError, match="verbose"):
            Network.fit(deep_classification_recipe((2, 2)), X, y, verbose=verbose)

    def test_invalid_logger_is_rejected(self):
        X, y = connection_initialisation_data()
        with pytest.raises(ValueError, match="logger"):
            Network.fit(deep_classification_recipe((2, 2)), X, y, logger="entlearn")

    def test_nested_and_failed_fits_leave_the_callers_logger_untouched(self, monkeypatch):
        X, y = connection_initialisation_data()
        recipe = deep_classification_recipe((2, 2))
        records = []
        logger = logging.Logger("caller", level=logging.DEBUG)  # noqa: LOG001

        class NestedFit(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())
                if len(records) == 1:
                    Network.fit(recipe, X, y, max_iter=1, logger=logger, verbose=0)

        handler = NestedFit()
        logger.addHandler(handler)
        before = (logger.level, tuple(logger.handlers), logger.propagate)
        Network.fit(recipe, X, y, max_iter=1, logger=logger, verbose=2)
        assert len(records) == 3  # initial loss, complete iteration, summary; no nested records
        assert (logger.level, tuple(logger.handlers), logger.propagate) == before

        def fail(session):
            raise RuntimeError("interrupted iteration")

        monkeypatch.setattr(fit_ops, "_fit_iteration_", fail)
        with pytest.raises(RuntimeError, match="interrupted iteration"):
            Network.fit(recipe, X, y, max_iter=1, logger=logger, verbose=1)
        assert (logger.level, tuple(logger.handlers), logger.propagate) == before
        assert len(records) == 3
