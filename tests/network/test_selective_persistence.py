"""Portable starting groups retain selective reuse and snapshot ownership."""

import math
import pickle

import pytest
import torch
from conftest import DEVICE
from network._fixtures import (
    assert_same_values,
    corrupt_saved_tensor,
    read_saved_metadata,
    replace_saved_tensors,
    resume_case,
    set_metadata,
    write_saved_metadata,
)
from safetensors.torch import load_file

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    Network,
    Recipe,
    RegressionHead,
)


def round_trip_provided_groups(tmp_path, case, selected, resumable):
    """Save a fit started from ``selected`` donor groups, reload it and check the groups.

    The reloaded original and current states must match the source's in separate storage,
    and the reloaded original state must start the same new fit. Returns the source and the
    loaded Network.
    """
    recipe, X, y, cats = case
    donor = Network.fit(recipe, X, y, X_cat=cats, max_iter=2)
    provided = donor.capture_current_state(blocks=selected)
    source = Network.fit(recipe, X, y, X_cat=cats, initial_state=provided, max_iter=2, seed=9)
    path = tmp_path / "groups.safetensors"
    source.save(path, resumable=resumable)
    loaded = Network.load(path, device=DEVICE)
    assert_same_values(loaded.initial_state, source.initial_state)
    for blocks in (None, ("input",), ("hidden_1",), ("output",), ()):
        assert_same_values(
            loaded.capture_current_state(blocks=blocks),
            source.capture_current_state(blocks=blocks),
        )
    expected = Network.fit(
        recipe, X, y, X_cat=cats, initial_state=source.initial_state, max_iter=2, seed=17
    )
    actual = Network.fit(
        recipe, X, y, X_cat=cats, initial_state=loaded.initial_state, max_iter=2, seed=17
    )
    assert actual.diagnostics == expected.diagnostics
    torch.testing.assert_close(
        actual.predict(X, X_cat=cats), expected.predict(X, X_cat=cats), rtol=0, atol=0
    )
    return source, loaded


class TestSelectivePersistence:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_original_parameter_groups_round_trip_and_start_another_fit(
        self, tmp_path, task, kind, coupling
    ):
        round_trip_provided_groups(
            tmp_path, resume_case(task, kind, coupling), selected=None, resumable=False
        )

    @pytest.mark.parametrize("selected", [None, ("hidden_1", "output"), ("input",)])
    @pytest.mark.parametrize("resumable", [False, True])
    def test_selected_groups_keep_the_saved_capability(self, tmp_path, selected, resumable):
        case = resume_case(kind="manifold")
        source, loaded = round_trip_provided_groups(tmp_path, case, selected, resumable)
        _, X, y, cats = case
        assert loaded.can_resume == resumable
        if resumable:
            assert not source.diagnostics.converged
            result = loaded.resume(X, y, X_cat=cats, max_iter=4, tol=0)
            expected = source.resume(X, y, X_cat=cats, max_iter=4, tol=0)
            assert result.diagnostics.n_iter > source.diagnostics.n_iter
            assert result.diagnostics == expected.diagnostics
            torch.testing.assert_close(
                result.predict(X, X_cat=cats), expected.predict(X, X_cat=cats), rtol=0, atol=0
            )

    def test_capture_after_loading_supports_a_new_task(self, tmp_path):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "task.safetensors"
        source.save(path)
        loaded = Network.load(path, device=DEVICE)
        target = Recipe.chain(
            loaded.capture_current_state().input_geometry.input,
            recipe.blocks[1],
            RegressionHead(),
            coupling=Coupling.M,
        )
        with pytest.warns(UserWarning, match="output.*incompatible"):
            adapted = Network.fit(
                target, X, X[:, :2], initial_state=loaded.capture_current_state(), max_iter=2
            )
        adapted.save(path)
        result = Network.load(path, device=DEVICE)
        assert result.schema.task == "regression"
        torch.testing.assert_close(result.predict(X), adapted.predict(X), rtol=0, atol=0)

    @pytest.mark.parametrize("field", ["max_iter", "tol"])
    def test_stopping_controls_are_required_metadata(self, tmp_path, field):
        recipe, X, y, _ = resume_case()
        path = tmp_path / "controls.safetensors"
        Network.fit(recipe, X, y, max_iter=3, tol=0.02).save(path)
        metadata = read_saved_metadata(path)
        assert metadata["models"][0]["max_iter"] == 3
        assert metadata["models"][0]["tol"] == 0.02
        del metadata["models"][0][field]
        write_saved_metadata(path, metadata)
        with pytest.raises(ValueError):
            Network.load(path)

    @pytest.mark.parametrize("resumable", [False, True])
    def test_the_file_preserves_the_selection_history_of_every_network(self, tmp_path, resumable):
        recipe, X, y, _ = resume_case()
        pairs = ((torch.arange(20, device=DEVICE), torch.arange(20, 30, device=DEVICE)),)

        def loss(prediction, target, **kwargs):
            return (prediction - target).square().mean()

        source = Network.fit(
            recipe,
            X,
            y,
            max_iter=2,
            n_inits=2,
            retain="members",
            validation_pairs=pairs,
            selection_loss=loss,
            return_train_score=True,
        )
        outcomes = source.diagnostics.initialisation_outcomes
        assert all(outcome.train_score is not None for outcome in outcomes)
        path = tmp_path / "scores.safetensors"
        source.save(path, resumable=resumable)
        loaded = Network.load(path, device=DEVICE)
        assert loaded.diagnostics == source.diagnostics
        for actual, expected in zip(loaded.members, source.members, strict=True):
            assert actual.diagnostics == expected.diagnostics

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_the_file_preserves_each_outcomes_parallel_backends(self, tmp_path, backend):
        recipe, X, y, _ = resume_case()
        source = Network.fit(
            recipe, X, y, max_iter=2, n_inits=2, n_jobs=2, parallel_backend=backend
        )
        path = tmp_path / "parallel.safetensors"
        source.save(path)
        outcomes = Network.load(path, device=DEVICE).diagnostics.initialisation_outcomes
        assert [(o.requested_backend, o.effective_backend) for o in outcomes] == [
            (backend, backend)
        ] * 2

    @pytest.mark.parametrize("fixed", [None, (0.25, 0.75)])
    def test_original_regression_groups_preserve_implicit_and_fixed_weights(self, tmp_path, fixed):
        recipe, X, y, _ = resume_case("regression")
        recipe = recipe.replace_block("output", epsilon_M=math.inf, W_M=fixed)
        donor = Network.fit(recipe, X, y, max_iter=2)
        source = Network.fit(
            recipe,
            X,
            y,
            initial_state=donor.capture_current_state(blocks=("output",)),
            max_iter=2,
            retain="members",
        )
        path = tmp_path / "weights.safetensors"
        source.save(path)
        loaded = Network.load(path, device=DEVICE)
        assert_same_values(loaded.initial_state, source.initial_state)
        transported = pickle.loads(pickle.dumps(loaded))
        assert_same_values(transported.initial_state, loaded.initial_state)
        assert transported.initial_state is transported.members[0].initial_state

    @pytest.mark.parametrize(
        "blocks",
        [
            (Input(K=1), Hidden(K=1), ClassificationHead(Coupling.M)),
            (Input(K=1), ClassificationHead(Coupling.M)),
            (Input(K=1), RegressionHead()),
            (Input(K=2), ClassificationHead(Coupling.M, n_classes=1)),
        ],
        ids=("hidden-source", "classification-source", "regression-source", "one-class"),
    )
    def test_narrowest_starting_groups_survive_the_round_trip(self, tmp_path, blocks):
        """A source width or class count of one is a width the decoder must accept."""
        _, X, codes, _ = resume_case()
        head = blocks[-1]
        if isinstance(head, RegressionHead):
            y = X[:, :1]
        elif head.n_classes == 1:
            y = torch.zeros_like(codes)
        else:
            y = codes
        recipe = Recipe.chain(*blocks, coupling=Coupling.M if len(blocks) > 2 else None)
        donor = Network.fit(recipe, X, y, max_iter=2)
        source = Network.fit(recipe, X, y, initial_state=donor.capture_current_state(), max_iter=2)
        path = tmp_path / "narrow.safetensors"
        source.save(path)
        loaded = Network.load(path, device=DEVICE)
        assert_same_values(loaded.initial_state, source.initial_state)
        torch.testing.assert_close(loaded.predict(X), source.predict(X), rtol=0, atol=0)

    def test_post_load_fine_tune_stopping_controls_survive_the_next_checkpoint(self, tmp_path):
        recipe, X, y, _ = resume_case("regression")
        path = tmp_path / "stopping.safetensors"
        Network.fit(recipe, X, y, max_iter=4, tol=0.1).save(path)
        loaded = Network.load(path, device=DEVICE)
        adapted = loaded.fine_tune(X, y, max_iter=2, tol=0)
        assert adapted.resume(X, y).diagnostics.n_iter == 2
        assert adapted.resume(X, y, max_iter=4).diagnostics.n_iter == 4
        adapted.save(path)
        Network.load(path, device=DEVICE).save(path)
        metadata = read_saved_metadata(path)
        assert metadata["models"][0]["max_iter"] == 2
        assert metadata["models"][0]["tol"] == 0


@pytest.fixture
def captured_artefact(tmp_path):
    recipe, X, y, _ = resume_case()
    donor = Network.fit(recipe, X, y, max_iter=2)
    source = Network.fit(recipe, X, y, initial_state=donor.capture_current_state(), max_iter=2)
    path = tmp_path / "captured.safetensors"
    source.save(path)
    return path


class TestSelectivePersistenceValidation:
    @pytest.mark.parametrize("index", [0, 1])
    @pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nan", "simplex"])
    def test_invalid_original_transitions_are_rejected(self, captured_artefact, index, damage):
        corrupt_saved_tensor(captured_artefact, f"states.0.parameters.{index}.theta", damage)
        with pytest.raises(ValueError):
            Network.load(captured_artefact)

    @pytest.mark.parametrize(
        "location,value",
        [
            (("states", 0, "parameters", 0, "source_width"), True),
            (("states", 0, "parameters", 0, "coupling"), "other"),
            (("states", 0, "parameters", 0, "description", "kind"), "Input"),
            (("models", 0, "max_iter"), 0),
            (("models", 0, "max_iter"), True),
            (("models", 0, "tol"), -1),
            (("models", 0, "tol"), True),
            (("selection", "selected_index"), 1),
            (("selection", "selected_index"), True),
            (("selection", "selected_index"), -1),
            (("selection", "initialisation_outcomes", 0, "train_score"), True),
            (("selection", "initialisation_outcomes", 0, "train_score"), "nan"),
        ],
    )
    def test_invalid_group_and_control_metadata(self, captured_artefact, location, value):
        set_metadata(captured_artefact, location, value)
        with pytest.raises(ValueError):
            Network.load(captured_artefact)

    @pytest.mark.parametrize("value", [{}, None])
    def test_a_group_without_a_block_description_is_rejected(self, captured_artefact, value):
        set_metadata(captured_artefact, ("states", 0, "parameters", 0), value)
        with pytest.raises(ValueError, match="require a block description"):
            Network.load(captured_artefact)

    @pytest.mark.parametrize(
        "task,index,field",
        [
            ("classification", 0, "source_width"),
            ("classification", 1, "source_width"),
            ("regression", 1, "source_width"),
            ("regression", 1, "output_width"),
        ],
    )
    def test_saved_group_widths_are_rejected_before_the_tensor_manifest(
        self, tmp_path, task, index, field
    ):
        """A declared width of zero fails on its own domain, not on a shape mismatch."""
        recipe, X, y, _ = resume_case(task)
        donor = Network.fit(recipe, X, y, max_iter=2)
        source = Network.fit(recipe, X, y, initial_state=donor.capture_current_state(), max_iter=2)
        path = tmp_path / "widths.safetensors"
        source.save(path)
        set_metadata(path, ("states", 0, "parameters", index, field), 0)
        with pytest.raises(ValueError, match="expected an integer >= 1"):
            Network.load(path)

    def test_original_geometry_must_retain_fitted_feature_meanings(self, captured_artefact):
        # Original geometry takes its feature dimensions from the fitted schema.
        path = captured_artefact
        tensors = load_file(path)
        centroids = tensors["states.0.continuous_centroids"]
        D = centroids.shape[1] + 1
        tensors["states.0.continuous_centroids"] = torch.cat((centroids, centroids[:, :1]), dim=1)
        tensors["states.0.feature_weights"] = centroids.new_full((D,), 1 / D)
        replace_saved_tensors(tensors, path)
        with pytest.raises(ValueError, match=r"incompatible saved tensor 'states\.0\.continuous"):
            Network.load(path)

    def test_every_original_state_requires_input_geometry(self, captured_artefact):
        set_metadata(captured_artefact, ("states", 0, "input_geometry"), None)
        with pytest.raises(ValueError, match="original state 0 has no input geometry"):
            Network.load(captured_artefact)

    @pytest.mark.parametrize("damage", ["name", "order", "source_width"])
    def test_original_groups_must_match_the_resolved_graph(self, captured_artefact, damage):
        path = captured_artefact
        metadata = read_saved_metadata(path)
        groups = metadata["states"][0]["parameters"]
        tensors = load_file(path)
        if damage == "name":
            groups[0]["description"]["description"]["name"] = "alien"
        elif damage == "order":
            groups.reverse()
            tensors["states.0.parameters.0.theta"], tensors["states.0.parameters.1.theta"] = (
                tensors["states.0.parameters.1.theta"],
                tensors["states.0.parameters.0.theta"],
            )
        else:
            groups[0]["source_width"] += 1
            theta = tensors["states.0.parameters.0.theta"]
            tensors["states.0.parameters.0.theta"] = torch.cat((theta, theta[:, :1]), dim=1)
        replace_saved_tensors(tensors, path)
        write_saved_metadata(path, metadata)
        with pytest.raises(ValueError):
            Network.load(path)
