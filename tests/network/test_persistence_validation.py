"""Malformed portable artefacts fail before a Network is returned."""

import math
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    change_metadata,
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
    Input,
    Network,
    Recipe,
)


@pytest.fixture
def artefact(tmp_path):
    recipe, X, y, cats = resume_case(kind="categorical")
    network = Network.fit(recipe, X, y, X_cat=cats, max_iter=2, n_inits=2, retain="members")
    path = tmp_path / "model.safetensors"
    network.save(path)
    return path


def single_model_artefact(tmp_path, retain):
    """Save a two-candidate fit that retains one fitted model."""
    recipe, X, y, _ = resume_case()
    path = tmp_path / f"{retain}.safetensors"
    Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain=retain).save(path)
    return path


def select(data, target, *, train_scores=None, scores=(1.0, 1.0)):
    """Record ``target`` as the winner, optionally after rewriting every candidate's scores."""
    if train_scores is not None:
        for index, outcome in enumerate(data["selection"]["initialisation_outcomes"]):
            outcome.update(score=scores[index], train_score=train_scores[index])
    data["selection"]["selected_index"] = target


class TestPersistenceValidation:
    @pytest.mark.parametrize(
        "criterion,train_scores,target,scores",
        [
            ("scores", None, None, None),
            # Tied scores without training scores: candidate order prefers 0.
            ("order", (None, None), 1, (1.0, 1.0)),
            # Tied scores whose recorded training scores give candidate 1 the smaller gap.
            ("train_scores", (0.0, 1.0), 0, (1.0, 1.0)),
            # A diverged score ranks last, below any finite score.
            ("nan", (None, None), 0, ("nan", 1.0)),
        ],
    )
    def test_winner_must_agree_with_historical_selection(
        self, artefact, criterion, train_scores, target, scores
    ):
        def damage(data):
            select(
                data,
                1 - data["selection"]["selected_index"] if target is None else target,
                train_scores=train_scores,
                scores=scores,
            )

        change_metadata(artefact, damage)
        with pytest.raises(ValueError, match="winner contradicts historical selection"):
            Network.load(artefact)

    def test_a_stored_winner_field_is_rejected(self, artefact):
        def damage(data):
            data["winner"] = data["selection"]["selected_index"]

        change_metadata(artefact, damage)
        with pytest.raises(ValueError, match="expected metadata fields"):
            Network.load(artefact)

    @pytest.mark.parametrize("retain", ["states", "members"])
    def test_a_selected_index_without_its_original_state_is_rejected(self, tmp_path, retain):
        recipe, X, y, _ = resume_case()
        path = tmp_path / "one.safetensors"
        Network.fit(recipe, X, y, max_iter=2, retain=retain).save(path)

        def damage(data):
            data["selection"]["initialisation_outcomes"][0]["index"] = 1
            data["selection"]["selected_index"] = 1

        change_metadata(path, damage)
        with pytest.raises(ValueError, match="missing winning original state"):
            Network.load(path)

    @pytest.mark.parametrize("retain", ["winner", "states"])
    def test_a_duplicated_selection_record_is_rejected(self, tmp_path, retain):
        path = single_model_artefact(tmp_path, retain)
        change_metadata(
            path, lambda data: data["models"][0]["diagnostics"].update(data["selection"])
        )
        with pytest.raises(ValueError, match="expected metadata fields"):
            Network.load(path)

    def test_a_duplicated_member_trajectory_is_rejected(self, artefact):
        def damage(data):
            selected = data["models"][data["selection"]["selected_index"]]["diagnostics"]
            for field in ("loss_history", "n_iter", "converged"):
                data["selection"][field] = selected[field]

        change_metadata(artefact, damage)
        with pytest.raises(ValueError, match="expected metadata fields"):
            Network.load(artefact)

    def test_a_member_bound_to_another_members_original_state_is_rejected(self, artefact):
        def damage(data):
            first, second = data["models"]
            second.update(initial_state=0, connection_sub_seeds=first["connection_sub_seeds"])

        change_metadata(artefact, damage)
        with pytest.raises(ValueError, match="inconsistent member original-state identity"):
            Network.load(artefact)

    def test_an_original_state_narrower_than_the_fitted_input_is_rejected(self, tmp_path):
        path = single_model_artefact(tmp_path, "winner")
        set_metadata(path, ("states", 0, "input_geometry", "K_active"), 1)
        tensors = load_file(path)
        tensors["states.0.continuous_centroids"] = tensors["states.0.continuous_centroids"][:1]
        replace_saved_tensors(tensors, path)
        with pytest.raises(
            ValueError, match="original state is incompatible with the fitted graph"
        ):
            Network.load(path)

    def test_a_fitted_graph_with_other_connection_seeds_is_rejected(self, tmp_path):
        path = single_model_artefact(tmp_path, "winner")

        def damage(data):
            data["models"][0]["connection_sub_seeds"][0][1] += 1

        change_metadata(path, damage)
        with pytest.raises(
            ValueError, match="original state is incompatible with the fitted graph"
        ):
            Network.load(path)

    def test_a_losing_original_state_with_other_seed_names_is_rejected(self, tmp_path):
        path = single_model_artefact(tmp_path, "states")

        def damage(data):
            loser = 1 - data["selection"]["selected_index"]
            data["states"][loser]["connection_sub_seeds"][0][0] = "other"

        change_metadata(path, damage)
        with pytest.raises(ValueError, match="seed names disagree with the fitted graph"):
            Network.load(path)

    def test_a_scoring_reference_without_instance_weight_recovery_is_rejected(self, tmp_path):
        X = torch.linspace(0, 1, 16, dtype=DTYPE, device=DEVICE).reshape(8, 2)
        y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=DEVICE)
        path = tmp_path / "no-recovery.safetensors"
        recipe = Recipe.chain(Input(K=2), ClassificationHead(Coupling.M))
        Network.fit(recipe, X, y, max_iter=2).save(path)
        set_metadata(path, ("models", 0, "has_scoring_reference"), True)
        tensors = load_file(path)
        tensors["models.0.Wt_ref"] = torch.ones(8, dtype=DTYPE)
        replace_saved_tensors(tensors, path)
        with pytest.raises(ValueError, match="scoring reference requires instance-weight recovery"):
            Network.load(path)

    def test_a_head_width_that_disagrees_with_the_schema_is_rejected(self, artefact):
        set_metadata(artefact, ("models", 0, "schema", "K_active", -1, 1), 5)
        with pytest.raises(ValueError, match="number of head outputs does not match"):
            Network.load(artefact)

    @pytest.mark.parametrize(
        ("damage", "message"),
        [
            (
                lambda data: data["states"][0]["connection_sub_seeds"][0].pop(),
                "invalid connection seed entry",
            ),
            (
                lambda data: [
                    outcome.update(index=1 - outcome["index"])
                    for outcome in data["selection"]["initialisation_outcomes"]
                ],
                "selection outcomes must retain candidate order",
            ),
            (
                lambda data: data["selection"]["initialisation_outcomes"][0].update(
                    effective_backend="gpu"
                ),
                "invalid initialisation outcome backend",
            ),
            (
                lambda data: data["states"][0]["input_geometry"].update(
                    input=deepcopy(data["models"][0]["recipe"]["blocks"][1])
                ),
                "original geometry requires an input description",
            ),
            (
                lambda data: data["models"][0].update(tol=-1.0),
                "saved tolerance must be non-negative",
            ),
            (lambda data: data.update(states=[]), "missing fitted model or original state"),
            (lambda data: data["models"].pop(), "inconsistent retention collection sizes"),
            (
                lambda data: data["models"][0].update(initial_state=len(data["states"])),
                "missing original state reference",
            ),
            (
                lambda data: data["models"][0]["predict_config"].update(output_mode=None),
                "saved prediction policy is not resolved",
            ),
        ],
        ids=[
            "seed-pair",
            "outcome-order",
            "backend",
            "geometry-description",
            "negative-tol",
            "no-states",
            "member-count",
            "state-index",
            "unresolved-policy",
        ],
    )
    def test_tampered_records_are_rejected_by_name(self, artefact, damage, message):
        change_metadata(artefact, damage)
        with pytest.raises(ValueError, match=message):
            Network.load(artefact)

    def test_a_winner_file_holds_exactly_one_original_state(self, tmp_path):
        path = single_model_artefact(tmp_path, "states")
        set_metadata(path, ("retention",), "winner")
        with pytest.raises(ValueError, match="inconsistent retention collection sizes"):
            Network.load(path)

    @pytest.mark.parametrize(
        ("value", "message"),
        [(-1.0, "invalid scoring reference"), (0.0, None)],
        ids=["negative", "zero"],
    )
    def test_a_scoring_reference_may_hold_zero_but_no_negative_weight(
        self, artefact, value, message
    ):
        tensors = load_file(artefact)
        tensors["models.0.Wt_ref"][0] = value
        replace_saved_tensors(tensors, artefact)
        with nullcontext() if message is None else pytest.raises(ValueError, match=message):
            Network.load(artefact)

    def test_a_file_needs_the_selection_record(self, artefact):
        set_metadata(artefact, ("selection",), None)
        with pytest.raises(ValueError, match="expected metadata fields"):
            Network.load(artefact)

    @pytest.mark.parametrize(
        "place,message",
        [(-1, "integer >= 0"), (1, "selection warnings are placed outside")],
        ids=("negative", "beyond-the-winners-own"),
    )
    def test_the_selection_warnings_must_follow_some_of_the_winners_own(
        self, artefact, place, message
    ):
        def damage(data):
            record = data["selection"]
            own = data["models"][record["selected_index"]]["diagnostics"]["warnings"]
            record["warnings_at"] = place if place < 0 else len(own) + place

        change_metadata(artefact, damage)
        with pytest.raises(ValueError, match=message):
            Network.load(artefact)

    @pytest.mark.parametrize("retain", ["winner", "states"])
    def test_a_single_retained_models_record_is_verified(self, tmp_path, retain):
        path = single_model_artefact(tmp_path, retain)

        def damage(data):
            record = data["selection"]
            for outcome in record["initialisation_outcomes"]:
                outcome["score"] = 1e9 if outcome["index"] == record["selected_index"] else -1e9

        change_metadata(path, damage)
        with pytest.raises(ValueError, match="winner contradicts historical selection"):
            Network.load(path)

    def test_an_outcome_field_outside_the_record_is_rejected(self, artefact):
        set_metadata(artefact, ("selection", "initialisation_outcomes", 0, "payload"), None)
        with pytest.raises(ValueError, match="expected metadata fields"):
            Network.load(artefact)

    def test_a_training_gap_tie_break_is_restored_and_verified(self, artefact):
        change_metadata(artefact, lambda data: select(data, 1, train_scores=(0.0, 1.0)))
        loaded = Network.load(artefact)
        outcomes = loaded.diagnostics.initialisation_outcomes
        assert [outcome.train_score for outcome in outcomes] == [0.0, 1.0]
        assert loaded.diagnostics.selected_outcome is outcomes[1]
        assert loaded.initial_state is loaded.initial_states[1]
        assert loaded.members[1].diagnostics.selected_outcome == outcomes[1]

    @pytest.mark.parametrize(
        "field,value",
        [
            ("format", "unknown"),
            ("version", 0),
            ("version", 2),
            ("version", 3),
            ("version", 4),
            ("version", True),
            ("package_version", None),
            ("package_version", 1),
            ("retention", "unknown"),
            ("states", []),
            ("states", {}),
            ("models", []),
            ("models", [None]),
        ],
    )
    def test_invalid_envelope(self, artefact, field, value):
        set_metadata(artefact, (field,), value)
        with pytest.raises(ValueError):
            Network.load(artefact)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("schema", None),
            ("recipe", {}),
            ("predict_config", {}),
            ("diagnostics", {}),
            ("epsilon_P_source", "unknown"),
            ("epsilon_P_source", None),
            ("initial_state", True),
            ("initial_state", 100),
            ("training_rows", 0),
            ("training_rows", True),
            ("connection_sub_seeds", []),
            ("blocks", []),
            ("has_scoring_reference", False),
            ("has_scoring_reference", "yes"),
            ("capability", "resumable"),
        ],
    )
    def test_invalid_model_fields(self, artefact, field, value):
        set_metadata(artefact, ("models", 0, field), value)
        with pytest.raises(ValueError):
            Network.load(artefact)

    @pytest.mark.parametrize(
        "section,field,value",
        [
            ("schema", "computation_dtype", "float16"),
            ("schema", "D_cont", True),
            ("schema", "M_cat", [3]),
            ("schema", "M", 0),
            ("schema", "K_active", [["wrong_name", 3]]),
            ("predict_config", "max_iter", True),
            ("predict_config", "epsilon_P", None),
            ("predict_config", "tol", float("nan")),
            ("diagnostics", "n_iter", True),
            ("diagnostics", "warnings", [12]),
            ("diagnostics", "converged", 1),
            ("diagnostics", "loss_history", [1.0]),
        ],
    )
    def test_invalid_structured_fields(self, artefact, section, field, value):
        set_metadata(artefact, ("models", 0, section, field), value)
        with pytest.raises(ValueError):
            Network.load(artefact)

    @pytest.mark.parametrize(
        "key",
        [
            "states.0.continuous_centroids",
            "states.0.feature_weights",
            "states.0.categorical_centroids.0",
            "models.0.blocks.0.continuous_centroids",
            "models.0.blocks.0.feature_weights",
            "models.0.blocks.0.categorical_centroids.0",
            "models.0.blocks.1.theta",
            "models.0.blocks.2.theta",
            "models.0.Wt_ref",
        ],
    )
    @pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nan"])
    def test_tensor_manifest_and_finiteness(self, artefact, key, damage):
        corrupt_saved_tensor(artefact, key, damage)
        with pytest.raises(ValueError):
            Network.load(artefact)

    @pytest.mark.parametrize(
        "key",
        ["models.0.blocks.1.theta", "models.0.blocks.2.theta", "states.0.feature_weights"],
    )
    @pytest.mark.parametrize("value", [-0.1, 2.0])
    def test_invalid_probability_parameters(self, artefact, key, value):
        tensors = load_file(artefact)
        tensors[key] = torch.full_like(tensors[key], value)
        replace_saved_tensors(tensors, artefact)
        with pytest.raises(ValueError):
            Network.load(artefact)

    @pytest.mark.parametrize("field", ["gamma", "instance_weights", "unknown"])
    def test_extra_tensors_are_not_ignored(self, artefact, field):
        tensors = load_file(artefact)
        tensors[f"models.0.blocks.0.{field}"] = torch.ones(30)
        replace_saved_tensors(tensors, artefact)
        with pytest.raises(ValueError, match="unexpected"):
            Network.load(artefact)

    def test_every_required_metadata_field_is_required(self, artefact):
        metadata = read_saved_metadata(artefact)
        sections = [
            (),
            ("models", 0),
            ("models", 0, "recipe"),
            ("models", 0, "schema"),
            ("models", 0, "predict_config"),
            ("models", 0, "diagnostics"),
            ("selection",),
            ("selection", "initialisation_outcomes", 0),
            ("models", 0, "blocks", 0),
            ("models", 0, "blocks", 1),
            ("states", 0),
            ("states", 0, "input_geometry"),
            ("states", 0, "input_geometry", "input", "description"),
        ]
        for location in sections:
            section = metadata
            for key in location:
                section = section[key]
            for missing in section:
                damaged = deepcopy(metadata)
                target = damaged
                for key in location:
                    target = target[key]
                del target[missing]
                write_saved_metadata(artefact, damaged)
                with pytest.raises(ValueError):
                    Network.load(artefact)

    @pytest.mark.parametrize(
        "path,value",
        [
            (("models", 0, "blocks", 0, "soft_assignments"), False),
            (("models", 0, "blocks", 1, "initial_widths"), [100, 100]),
            (("models", 0, "recipe", "blocks", 2, "description", "n_classes"), 4),
            (("models", 0, "connection_sub_seeds", 0, 1), 2**64),
            (("states", 0, "input_geometry", "captured"), 1),
            (("states", 0, "input_geometry", "input", "description", "name"), "other"),
            (("selection", "initialisation_outcomes"), []),
            (("selection", "initialisation_outcomes", 0, "seed"), 2**64),
            (("models", 1, "recipe", "connections", 0, "delta"), 2.0),
        ],
    )
    def test_cross_record_invariants(self, artefact, path, value):
        set_metadata(artefact, path, value)
        with pytest.raises(ValueError):
            Network.load(artefact)

    @pytest.mark.parametrize(
        "path,value,message",
        [
            (("recipe", "blocks", 0, "description", "K"), True, "K must be a positive integer"),
            (("recipe", "blocks", 0, "description", "name"), "", "names must be non-empty"),
            (("recipe", "blocks", 0, "description", "centroid_strategy"), None, "centroid_str"),
            (("recipe", "connections", 0, "source"), [["input"]], "target must be non-empty"),
            (("recipe", "connections", 0, "delta"), {"positive_infinity": True}, "delta must"),
            (("predict_config", "predict_mode"), ["single"], "predict_mode must be"),
        ],
    )
    def test_malformed_descriptions_fail_with_the_owner_wording(
        self, artefact, path, value, message
    ):
        set_metadata(artefact, ("models", 0, *path), value)
        with pytest.raises(ValueError, match=message):
            Network.load(artefact)

    def test_an_original_state_name_is_validated_by_the_original_state_owner(self, artefact):
        set_metadata(artefact, ("states", 0, "input_geometry", "input", "description", "name"), 5)
        with pytest.raises(ValueError, match="unique non-empty block names"):
            Network.load(artefact)

    @pytest.mark.parametrize(
        "marker,message",
        [
            ({"positive_infinity": False}, "invalid infinite temperature"),
            ({"positive_infinity": "yes"}, "expected a metadata boolean"),
            ({"negative_infinity": True}, "expected metadata fields: positive_infinity"),
        ],
    )
    def test_malformed_infinite_temperature_marker(self, tmp_path, marker, message):
        recipe, X, y, _ = resume_case()
        recipe = recipe.replace_block("input", epsilon_T=math.inf)
        path = tmp_path / "infinite.safetensors"
        Network.fit(recipe, X, y, max_iter=2).save(path)

        def damage(data):
            for model in data["models"]:
                model["recipe"]["blocks"][0]["description"]["epsilon_T"] = marker

        change_metadata(path, damage)
        with pytest.raises(ValueError, match=message):
            Network.load(path)

    def test_unretained_member_original_geometry_is_still_validated(self, tmp_path):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2, n_inits=2, retain="states")
        path = tmp_path / "states.safetensors"
        source.save(path)
        data = read_saved_metadata(path)
        other = 1 - data["selection"]["selected_index"]
        tensors = load_file(path)
        tensors[f"states.{other}.continuous_centroids"] = torch.full_like(
            tensors[f"states.{other}.continuous_centroids"], torch.nan
        )
        replace_saved_tensors(tensors, path)
        with pytest.raises(ValueError):
            Network.load(path)

    @pytest.mark.parametrize(
        "key", ["states.0.manifold_projectors", "models.0.blocks.0.manifold_projectors"]
    )
    def test_invalid_manifold_basis_is_rejected(self, tmp_path, key):
        recipe, X, y, _ = resume_case(kind="manifold")
        path = tmp_path / "manifold.safetensors"
        Network.fit(recipe, X, y, max_iter=2).save(path)
        tensors = load_file(path)
        tensors[key] = torch.zeros_like(tensors[key])
        replace_saved_tensors(tensors, path)
        with pytest.raises(ValueError, match="orthonormal"):
            Network.load(path)

    def test_missing_learned_output_weights_cannot_change_the_objective(self, tmp_path):
        recipe, X, y, _ = resume_case("regression")
        path = tmp_path / "regression.safetensors"
        Network.fit(recipe, X, y, max_iter=2).save(path)
        tensors = load_file(path)
        del tensors["models.0.blocks.2.W_M"]
        replace_saved_tensors(tensors, path)
        set_metadata(path, ("models", 0, "blocks", 2, "has_output_weights"), False)
        with pytest.raises(ValueError):
            Network.load(path)

    def test_a_saved_iteration_count_must_be_a_complete_iteration(self, artefact):
        def damage(data):
            record = data["models"][data["selection"]["selected_index"]]["diagnostics"]
            record["n_iter"] = 0
            record["loss_history"] = record["loss_history"][:1]

        change_metadata(artefact, damage)
        with pytest.raises(ValueError):
            Network.load(artefact)

    def test_a_single_class_head_survives_the_round_trip(self, tmp_path):
        # One declared class is a legal head width, so decoding must not require two.
        X = torch.rand(8, 2, dtype=torch.float64)
        recipe = Recipe.chain(Input(K=2), ClassificationHead(coupling=Coupling.M, n_classes=1))
        source = Network.fit(recipe, X, torch.zeros(8, dtype=torch.int64), max_iter=2)
        path = tmp_path / "one_class.safetensors"
        source.save(path)

        assert Network.load(path).recipe == recipe

    def test_a_network_fitted_on_one_row_survives_the_round_trip(self, tmp_path):
        # One training row is a legal fit, so decoding must not require two.
        X = torch.full((1, 2), 0.5, dtype=DTYPE, device=DEVICE)
        y = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        recipe = Recipe.chain(Input(K=1), ClassificationHead(coupling=Coupling.M, n_classes=1))
        path = tmp_path / "one_row.safetensors"
        Network.fit(recipe, X, y, max_iter=2).save(path, resumable=True)

        assert Network.load(path, device=DEVICE).can_resume

    def test_a_default_coupling_is_saved_explicitly(self, tmp_path):
        recipe, X, y, _ = resume_case()
        default = Recipe.chain(*recipe.blocks[:2], ClassificationHead(), theta_alpha=1.1)
        assert default == recipe
        path = tmp_path / "default.safetensors"
        Network.fit(default, X, y, max_iter=2).save(path)

        saved = read_saved_metadata(path)["models"][0]["recipe"]
        assert saved["connections"][0]["coupling"] == "M"
        assert saved["blocks"][2]["description"]["coupling"] == "M"
        assert Network.load(path).recipe == recipe

    def test_a_cleared_saved_coupling_cannot_turn_an_s_model_into_m(self, tmp_path):
        recipe, X, y, _ = resume_case(coupling=Coupling.S)
        path = tmp_path / "s.safetensors"
        Network.fit(recipe, X, y, max_iter=2).save(path)
        set_metadata(path, ("models", 0, "recipe", "connections", 0, "coupling"), None)
        # The cleared coupling reads as M, which the saved S transition then fails.
        with pytest.raises(ValueError, match=r"input_to_hidden_1\.theta .* sum to one"):
            Network.load(path)


class TestPersistenceIO:
    @pytest.mark.parametrize("device", ["cpu:0", torch.device("cpu:0")])
    def test_indexed_cpu_is_canonicalised(self, artefact, device):
        restored = Network.load(artefact, device=device)
        assert restored.device == torch.device("cpu")

    @pytest.mark.parametrize("kind", ["cuda", "mps"])
    def test_explicit_accelerator_spellings_or_unavailable_error(self, tmp_path, kind):
        recipe, X, y, _ = resume_case()
        X = X.to(torch.float32)
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "device.safetensors"
        source.save(path)
        available = (
            torch.cuda.is_available() if kind == "cuda" else torch.backends.mps.is_available()
        )
        if not available:
            with pytest.raises(ValueError, match="unavailable"):
                Network.load(path, device=kind)
            return
        expected = source.predict(X).cpu()
        for device in (kind, f"{kind}:0", torch.device(kind)):
            loaded = Network.load(path, device=device)
            assert loaded.device.type == kind
            actual = loaded.predict(X.to(loaded.device)).cpu()
            tolerance = 512 * torch.finfo(torch.float32).eps
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_mps_loading_requires_a_float32_save(self, tmp_path):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS is unavailable")
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X.to(torch.float64), y, max_iter=2)
        path = tmp_path / "float64.safetensors"
        source.save(path)
        with pytest.raises(ValueError, match="MPS loading requires a saved float32"):
            Network.load(path, device="mps")

    @pytest.mark.parametrize("suffix", [".pt", ".pkl", ".json", ""])
    def test_no_extension_selected_pickle_or_state_dictionary_reader(
        self, tmp_path, suffix, monkeypatch
    ):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=1)
        path = tmp_path / ("model" + suffix)
        with pytest.raises(ValueError, match="safetensors"):
            source.save(path)
        torch.save({"metadata": {"schema_version": 1}, "tensors": {}}, path)
        monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle reader was called"))
        with pytest.raises(ValueError, match="safetensors"):
            Network.load(path)
        for name in ("state_dict", "load_state_dict", "from_state_dict"):
            assert not hasattr(Network, name)

    @pytest.mark.parametrize("resumable", [1, "yes", None])
    def test_unsupported_save_controls_leave_existing_files_unchanged(self, artefact, resumable):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=1)
        before = artefact.read_bytes()
        with pytest.raises(ValueError):
            source.save(artefact, resumable=resumable)
        assert artefact.read_bytes() == before

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ('{"format": 1, "format": 2}', "duplicate metadata field 'format'"),
            ("[]", "expected metadata fields"),
            ("{", "Expecting property name"),
            ('{"x": NaN}', "non-standard JSON number 'NaN'"),
        ],
    )
    def test_json_is_strict(self, artefact, content, message):
        write_saved_metadata(artefact, content)
        with pytest.raises(ValueError, match=message):
            Network.load(artefact)

    def test_missing_bundle_is_a_file_error(self, artefact):
        artefact.unlink()
        with pytest.raises(FileNotFoundError):
            Network.load(artefact)

    def test_invalid_tensor_container_is_not_read_as_pickle(self, artefact, monkeypatch):
        torch.save({"metadata": {"schema_version": 1}}, artefact)
        monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle reader was called"))
        with pytest.raises(ValueError):
            Network.load(artefact)

    @pytest.mark.parametrize("device", ["meta", "not-a-device", 0, None])
    def test_invalid_device_is_rejected(self, artefact, device):
        with pytest.raises(ValueError):
            Network.load(artefact, device=device)
