"""The Network record holds one selection history, apart from every fitted trajectory."""

import logging
import math
import pickle

import pytest
from conftest import DEVICE
from network._fixtures import resume_case

from entlearn import FitDiagnostics, Network
from entlearn.network.continuation import continue_fit
from entlearn.network.persistence.snapshot import decode_snapshot, encode_snapshot
from entlearn.network.state import InitOutcome, _NetworkRecord, _SelectionHistory


def history(network):
    """Return the published records and index, comparable when a score is NaN."""
    diagnostics = network.diagnostics
    return repr((diagnostics.initialisation_outcomes, diagnostics.selected_index))


class TestSelectionHistory:
    def test_every_route_publishes_one_history_to_the_winner_and_each_member(self):
        recipe, X, y, _ = resume_case()
        fitted = Network.fit(recipe, X, y, max_iter=2, tol=0, n_inits=3, retain="members")
        source = fitted._record()
        outcomes = tuple(
            InitOutcome(index, 10 + index, score, train_score, "threads", "serial")
            for index, (score, train_score) in enumerate(((2.0, 1.0), (math.nan, None), (1.0, 0.5)))
        )
        original = source.members[2].trajectory.warnings
        record = _NetworkRecord(
            source.members[2],
            _SelectionHistory(outcomes, 2, ("selection warning",), len(original)),
            source.states,
            source.members,
        )

        def continued(**replacement):
            return continue_fit(
                record,
                X,
                y,
                X_cat=None,
                sample_weights=None,
                class_weights=None,
                task_weights=None,
                computation_dtype=None,
                max_iter=4,
                tol=0,
                logger=logging.getLogger(__name__),
                verbose=0,
                **replacement,
            )

        routes = {
            "published": record,
            "resumed": continued(),
            "fine-tuned": continued(replacement=recipe),
            "decoded": decode_snapshot(*encode_snapshot(record), device=X.device),
        }
        for route, result in routes.items():
            network = Network._publish(result)
            winner = result.members[2].trajectory
            assert history(network) == repr((outcomes, 2)), route
            assert network.diagnostics.loss_history == winner.loss_history, route
            # The selection's warnings follow the fit's own; fine-tuning resets both.
            expected = (
                winner.warnings
                if route == "fine-tuned"
                else (*original, "selection warning", *winner.warnings[len(original) :])
            )
            assert network.diagnostics.warnings == expected, route
            for index, member in enumerate(network.members):
                assert history(member) == repr(((outcomes[index],), index)), route
                assert member.diagnostics.warnings == result.members[index].trajectory.warnings

    def test_a_selected_index_without_its_record_has_no_selected_outcome(self):
        outcome = InitOutcome(0, 10, 1.0, None, "threads", "serial")
        diagnostics = FitDiagnostics((1.0, 0.5), 1, True, (), (outcome,), selected_index=1)
        with pytest.raises(ValueError, match="no initialisation outcome records the selected"):
            _ = diagnostics.selected_outcome


class TestWarningOrder:
    @pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
    @pytest.mark.filterwarnings("ignore:hidden block:UserWarning")
    @pytest.mark.parametrize("retain", ["winner", "members"])
    def test_the_selection_warning_keeps_its_place_through_resume_and_transport(
        self, tmp_path, retain
    ):
        recipe, X, y, _ = resume_case()
        recipe = recipe.replace_block("hidden_1", K=1)
        fitted = Network.fit(recipe, X, y, max_iter=1, tol=0, n_inits=2, retain=retain)
        own, selection = fitted.diagnostics.warnings
        assert own.startswith("hidden block") and selection.startswith("the solver did not")
        path = tmp_path / "fitted.safetensors"
        fitted.save(path, resumable=True)
        routes = {
            "resumed": fitted.resume(X, y, max_iter=2),
            "loaded, then resumed": Network.load(path, device=DEVICE).resume(X, y, max_iter=2),
        }
        resumed = routes["resumed"]
        routes["pickled"] = pickle.loads(pickle.dumps(resumed))
        resumed.save(path, resumable=True)
        routes["resumed, then loaded"] = Network.load(path, device=DEVICE)
        *_, resume = resumed.diagnostics.warnings
        assert resume.startswith("the solver has not converged")
        for route, network in routes.items():
            assert network.diagnostics.warnings == (own, selection, resume), route
            for member in network.members or ():
                assert member.diagnostics.warnings == (own, resume), route
        # Fine-tuning starts a new record, so the selection's warning and its place both go.
        assert fitted.fine_tune(X, y).diagnostics.warnings == ()
