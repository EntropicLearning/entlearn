"""A hidden block's affiliation temperature given as a ratio to its incoming ``delta``."""

import pickle
from typing import Any, cast

import pytest
import torch
from conftest import FIXED_BUDGET
from network._fixtures import (
    assert_same_checkpoint,
    change_metadata,
    link,
    read_saved_metadata,
    resume_case,
    round_trip_network,
)

from entlearn import ClassificationHead, Hidden, Input, ManifoldInput, Network, Recipe

_RATIO, _DELTA = 1.5, 0.4
_INCOMING = "input_to_hidden_1"
_FITTED = ("transition_matrices", "head_parameters", "training_affiliations")


def _ratio_case() -> tuple[Recipe, Recipe, torch.Tensor, torch.Tensor]:
    """Return one Recipe that gives the ratio, its twin that gives ``epsilon``, and data."""
    recipe, X, y, _ = resume_case()
    recipe = recipe.replace_connection(_INCOMING, delta=_DELTA)
    explicit = recipe.replace_block("hidden_1", epsilon=_RATIO * _DELTA)
    ratio = recipe.replace_block("hidden_1", epsilon=None, epsilon_ratio=_RATIO)
    return ratio, explicit, X, y


def _assert_same_fit(actual: Network, expected: Network) -> None:
    """Compare two fits of Recipes that differ only in how they state a temperature."""
    assert actual.diagnostics.loss_history == expected.diagnostics.loss_history
    for name in _FITTED:
        torch.testing.assert_close(actual.inspect(name), expected.inspect(name), rtol=0, atol=0)


def _hidden_descriptions(node: object) -> list[dict[str, Any]]:
    """Collect every saved ``Hidden`` description, in the Recipes and the initial states."""
    if isinstance(node, dict):
        if node.get("kind") == "Hidden":
            return [node["description"]]
        node = list(node.values())
    if isinstance(node, list):
        return [found for item in node for found in _hidden_descriptions(item)]
    return []


class TestRatioDescription:
    def test_an_omitted_temperature_defaults_to_zero_unless_a_ratio_is_given(self):
        assert Hidden().epsilon == 0.0
        assert Hidden(epsilon=None).epsilon == 0.0
        assert Hidden(epsilon_ratio=_RATIO).epsilon is None
        assert Hidden(epsilon=None, epsilon_ratio=0).epsilon is None

    def test_the_resolved_temperature_is_the_ratio_times_the_incoming_delta(self):
        ratio, explicit, _, _ = _ratio_case()

        assert ratio.blocks[1] == Hidden(K=3, epsilon_ratio=_RATIO, name="hidden_1")
        assert ratio.resolve_ratios() == explicit
        assert explicit.resolve_ratios() is explicit

    def test_the_resolved_temperature_follows_the_incoming_delta_alone(self):
        ratio, _, _, _ = _ratio_case()

        weaker = ratio.replace_connection(_INCOMING, delta=0.1).resolve_ratios()
        outgoing = ratio.replace_connection("hidden_1_to_output", delta=7.0)

        assert weaker.blocks[1] == Hidden(K=3, epsilon=_RATIO * 0.1, name="hidden_1")
        assert outgoing.resolve_ratios().blocks[1].epsilon == _RATIO * _DELTA

    def test_a_zero_ratio_is_a_hard_block(self):
        recipe = Recipe.chain(Input(), Hidden(epsilon_ratio=0), ClassificationHead())

        assert recipe.resolve_ratios().blocks[1] == Hidden(name="hidden_1")

    @pytest.mark.parametrize(
        "value",
        [-0.1, float("nan"), float("inf"), True, "1.5", 10**10000],
        ids=["negative", "nan", "inf", "bool", "str", "oversized"],
    )
    def test_rejects_a_ratio_that_is_not_finite_and_non_negative(self, value):
        with pytest.raises(ValueError, match=r"epsilon_ratio.*non-negative"):
            Hidden(epsilon_ratio=value)

    @pytest.mark.parametrize("epsilon", [0.0, 0.1], ids=["zero", "positive"])
    @pytest.mark.parametrize("ratio", [0.0, _RATIO], ids=["zero-ratio", "positive-ratio"])
    def test_rejects_a_ratio_combined_with_an_explicit_temperature(self, epsilon, ratio):
        with pytest.raises(ValueError, match="cannot be combined with an explicit epsilon"):
            Hidden(epsilon=epsilon, epsilon_ratio=ratio)

    def test_a_value_and_a_ratio_are_exchanged_together(self):
        ratio, _, _, _ = _ratio_case()

        with pytest.raises(ValueError, match="cannot be combined with an explicit epsilon"):
            ratio.replace_block("hidden_1", epsilon=0.1)
        replaced = ratio.replace_block("hidden_1", epsilon=0.1, epsilon_ratio=None)
        assert replaced.blocks[1] == Hidden(K=3, epsilon=0.1, name="hidden_1")
        with pytest.raises(ValueError, match="cannot be combined with an explicit epsilon"):
            replaced.replace_block("hidden_1", epsilon_ratio=_RATIO)
        assert replaced.replace_block("hidden_1", epsilon=None, epsilon_ratio=_RATIO) == ratio

    @pytest.mark.parametrize("description", [Input, ManifoldInput])
    def test_an_input_block_has_no_ratio(self, description):
        with pytest.raises(TypeError, match="epsilon_ratio"):
            cast(Any, description)(epsilon_ratio=_RATIO)

    @pytest.mark.parametrize("description", [Input, ManifoldInput])
    def test_an_input_temperature_remains_numeric(self, description):
        assert description().epsilon == 0.0
        with pytest.raises(ValueError, match=r"epsilon.*non-negative"):
            description(epsilon=None)

    def test_rejects_a_ratio_on_a_block_with_several_incoming_connections(self):
        def build(second: Hidden) -> Recipe:
            return Recipe(
                blocks=(
                    Input(name="input"),
                    Hidden(name="first"),
                    second,
                    ClassificationHead(name="output"),
                ),
                connections=(
                    link("input", "first"),
                    link("input", "second"),
                    link("first", "second"),
                    link("second", "output"),
                ),
            )

        build(Hidden(name="second"))
        with pytest.raises(ValueError, match=r"epsilon_ratio.*exactly one incoming"):
            build(Hidden(name="second", epsilon_ratio=_RATIO))

    def test_rejects_a_ratio_whose_temperature_overflows(self):
        with pytest.raises(ValueError, match=r"epsilon_ratio.*finite"):
            Recipe.chain(Input(), Hidden(epsilon_ratio=1e200), ClassificationHead(), delta=1e200)


@FIXED_BUDGET
class TestRatioFit:
    def test_a_fit_equals_the_fit_with_the_explicit_temperature(self):
        ratio, explicit, X, y = _ratio_case()
        controls = dict(seed=5, max_iter=6, tol=0)

        fitted = Network.fit(ratio, X, y, **controls)
        expected = Network.fit(explicit, X, y, **controls)
        hard = Network.fit(explicit.replace_block("hidden_1", epsilon=0.0), X, y, **controls)

        assert fitted.recipe == ratio
        _assert_same_fit(fitted, expected)
        assert hard.diagnostics.loss_history != expected.diagnostics.loss_history

    def test_fine_tuning_the_incoming_delta_moves_the_temperature_with_it(self):
        ratio, explicit, X, y = _ratio_case()
        controls = dict(seed=5, max_iter=3, tol=0)
        weaker = ratio.replace_connection(_INCOMING, delta=0.2)
        explicit_weaker = explicit.replace_connection(_INCOMING, delta=0.2).replace_block(
            "hidden_1", epsilon=_RATIO * 0.2
        )

        tuned = Network.fit(ratio, X, y, **controls).fine_tune(X, y, recipe=weaker, max_iter=3)
        expected = Network.fit(explicit, X, y, **controls).fine_tune(
            X, y, recipe=explicit_weaker, max_iter=3
        )
        restated = Network.fit(explicit, X, y, **controls).fine_tune(
            X, y, recipe=weaker, max_iter=3
        )

        assert tuned.recipe == restated.recipe == weaker
        _assert_same_fit(tuned, expected)
        _assert_same_fit(restated, expected)


@FIXED_BUDGET
class TestRatioPersistence:
    def test_a_saved_or_pickled_fit_keeps_the_ratio_and_resumes(self, tmp_path):
        ratio, explicit, X, y = _ratio_case()
        controls = dict(seed=5, max_iter=3, tol=0)
        fitted = Network.fit(ratio, X, y, **controls)
        expected = Network.fit(explicit, X, y, **controls).resume(X, y, max_iter=6, tol=0)

        loaded = round_trip_network(fitted, tmp_path / "ratio.safetensors", resumable=True)
        unpickled = pickle.loads(pickle.dumps(fitted))

        assert _hidden_descriptions(read_saved_metadata(tmp_path / "ratio.safetensors")) == [
            {"K": 3, "epsilon": None, "epsilon_ratio": _RATIO, "name": "hidden_1"}
        ]
        for restored in (loaded, unpickled):
            assert_same_checkpoint(restored, fitted)
            torch.testing.assert_close(restored.predict(X), fitted.predict(X), rtol=0, atol=0)
            resumed = restored.resume(X, y, max_iter=6, tol=0)
            assert resumed.recipe == ratio
            _assert_same_fit(resumed, expected)

    def test_rejects_a_saved_ratio_with_an_explicit_temperature(self, tmp_path):
        ratio, _, X, y = _ratio_case()
        path = tmp_path / "conflicting.safetensors"
        Network.fit(ratio, X, y, seed=5, max_iter=3, tol=0).save(path)

        change_metadata(
            path, lambda metadata: _hidden_descriptions(metadata)[0].update(epsilon=0.0)
        )

        with pytest.raises(ValueError, match="cannot be combined with an explicit epsilon"):
            Network.load(path)

    @pytest.mark.parametrize(
        "damage",
        [lambda description: description.pop("epsilon"), lambda d: d.update(epsilon_scale=1.0)],
        ids=["missing-epsilon", "unknown-field"],
    )
    def test_a_saved_description_must_hold_every_field(self, tmp_path, damage):
        ratio, _, X, y = _ratio_case()
        path = tmp_path / "damaged.safetensors"
        Network.fit(ratio, X, y, seed=5, max_iter=3, tol=0).save(path)

        change_metadata(path, lambda metadata: damage(_hidden_descriptions(metadata)[0]))

        with pytest.raises(ValueError, match="expected metadata fields: K epsilon"):
            Network.load(path)
