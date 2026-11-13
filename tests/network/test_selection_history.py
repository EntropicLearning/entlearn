"""The Network record holds one selection history, apart from every fitted trajectory."""

import pickle

import pytest
from conftest import DEVICE, FIXED_BUDGET
from network._fixtures import resume_case, round_trip_network

from entlearn import FitDiagnostics, Network
from entlearn.network.state import InitOutcome


def history(network):
    """Return the published initialisation outcomes and selected index."""
    diagnostics = network.diagnostics
    return diagnostics.initialisation_outcomes, diagnostics.selected_index


class TestSelectionHistory:
    @FIXED_BUDGET
    def test_every_route_publishes_one_history_to_the_winner_and_each_member(self, tmp_path):
        recipe, X, y, _ = resume_case()
        fits = (
            Network.fit(recipe, X, y, max_iter=2, tol=0, n_inits=3, retain="members", seed=seed)
            for seed in range(10)
        )
        # A winner after the first candidate tells the winner's record from member 0's. Which
        # candidate wins depends on the device's random stream, so take the first such seed.
        fitted = next((fit for fit in fits if fit.diagnostics.selected_index != 0), None)
        assert fitted is not None, "candidate 0 won for every seed from 0 to 9"
        outcomes, selected = history(fitted)
        assert len(outcomes) == 3
        routes = {
            "published": fitted,
            "resumed": fitted.resume(X, y, max_iter=4),
            "fine-tuned": fitted.fine_tune(X, y, max_iter=4),
            "loaded": round_trip_network(fitted, tmp_path / "fitted.safetensors", resumable=True),
        }
        for route, network in routes.items():
            assert history(network) == (outcomes, selected), route
            assert network.members is not None
            winner = network.members[selected].diagnostics
            assert network.diagnostics.loss_history == winner.loss_history, route
            for index, member in enumerate(network.members):
                assert history(member) == ((outcomes[index],), index), route

    def test_a_selected_index_without_its_record_has_no_selected_outcome(self):
        outcome = InitOutcome(0, 10, 1.0, None, "threads", "serial")
        diagnostics = FitDiagnostics((1.0, 0.5), 1, True, (), (outcome,), selected_index=1)
        with pytest.raises(ValueError, match="selected index"):
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
