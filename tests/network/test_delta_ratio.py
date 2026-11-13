"""A connection's ``delta`` given as a ratio to the ``delta`` of the previous connection."""

import pickle

import pytest
import torch
from conftest import FIXED_BUDGET
from network._fixtures import (
    assert_same_checkpoint,
    blobs,
    link,
    read_saved_metadata,
    round_trip_network,
)

from entlearn import ClassificationHead, Connection, Hidden, Input, Network, Recipe

_FIRST, _RATIO, _EPSILON_RATIO = 0.4, 3.0, 1.5
_HEAD = ClassificationHead()
_CONTROLS = dict(seed=5, max_iter=6, tol=0)
_FITTED = ("transition_matrices", "head_parameters", "training_affiliations")


def _ratio_case() -> tuple[Recipe, Recipe, torch.Tensor, torch.Tensor]:
    """Return a chain of ratios, its twin with the values written out, and data.

    The second hidden block also gives ``epsilon_ratio``, against a ``delta`` that is a ratio.
    """
    X, y = blobs(3, 0.2, rows_per_blob=10, n_features=3, seed=11)
    first = Input(K=4, epsilon=0.1, epsilon_D=0.2)
    ratio = Recipe.chain(
        first,
        Hidden(K=3, epsilon=0.15),
        Hidden(K=2, epsilon_ratio=_EPSILON_RATIO),
        ClassificationHead(),
        delta=_FIRST,
        delta_ratio=_RATIO,
        theta_alpha=1.1,
    )
    explicit = Recipe.chain(
        first,
        Hidden(K=3, epsilon=0.15),
        Hidden(K=2, epsilon=_EPSILON_RATIO * (_FIRST * _RATIO)),
        ClassificationHead(),
        delta=[_FIRST, _FIRST * _RATIO, _FIRST * _RATIO * _RATIO],
        theta_alpha=1.1,
    )
    return ratio, explicit, X, y


def _dag(last: Connection) -> Recipe:
    """Return a graph whose block ``second`` has two incoming connections."""
    return Recipe(
        blocks=(
            Input(name="input"),
            Hidden(name="first"),
            Hidden(name="second"),
            ClassificationHead(name="output"),
        ),
        connections=(
            link("input", "first"),
            link("input", "second"),
            link("first", "second"),
            last,
        ),
    )


def _assert_same_fit(actual: Network, expected: Network) -> None:
    """Compare two fits of Recipes that differ only in how they state their values."""
    assert actual.diagnostics.loss_history == expected.diagnostics.loss_history
    for name in _FITTED:
        torch.testing.assert_close(actual.inspect(name), expected.inspect(name), rtol=0, atol=0)


def _saved_connections(node: object) -> list[dict]:
    """Collect every saved connection description, in the Recipes and the initial states."""
    if isinstance(node, dict):
        if "delta_ratio" in node:
            return [node]
        node = list(node.values())
    if isinstance(node, list):
        return [found for item in node for found in _saved_connections(item)]
    return []


class TestDeltaRatioDescription:
    def test_a_chain_of_ratios_resolves_from_the_first_delta(self):
        ratio, explicit, _, _ = _ratio_case()

        assert [(c.delta, c.delta_ratio) for c in ratio.connections] == [
            (_FIRST, None),
            (None, _RATIO),
            (None, _RATIO),
        ]
        assert ratio.blocks[2] == Hidden(K=2, epsilon_ratio=_EPSILON_RATIO, name="hidden_2")
        assert ratio.resolve_ratios() == explicit
        assert explicit.resolve_ratios() is explicit

    def test_the_resolved_deltas_follow_a_changed_upstream_delta(self):
        ratio, _, _, _ = _ratio_case()

        first = ratio.replace_connection("input_to_hidden_1", delta=0.1).resolve_ratios()
        middle = ratio.replace_connection("hidden_1_to_hidden_2", delta_ratio=2.0).resolve_ratios()

        assert [c.delta for c in first.connections] == [0.1, 0.1 * _RATIO, 0.1 * _RATIO * _RATIO]
        assert first.blocks[2].epsilon == _EPSILON_RATIO * (0.1 * _RATIO)
        assert [c.delta for c in middle.connections] == [
            _FIRST,
            _FIRST * 2.0,
            _FIRST * 2.0 * _RATIO,
        ]
        assert middle.blocks[2].epsilon == _EPSILON_RATIO * (_FIRST * 2.0)

    @pytest.mark.parametrize(
        ("deltas", "ratios"),
        [
            ([0.1, 0.2, 0.8], None),
            ([0.1, None, None], [None, 2, 4]),
            ([0.1, None, 0.8], [None, 2, None]),
        ],
        ids=["absolute-values", "different-ratios", "mixed"],
    )
    def test_each_connection_can_have_its_own_value_or_ratio(self, deltas, ratios):
        recipe = Recipe.chain(Input(), Hidden(), Hidden(), _HEAD, delta=deltas, delta_ratio=ratios)

        assert [c.delta for c in recipe.resolve_ratios().connections] == [0.1, 0.2, 0.8]

    def test_chain_sequences_align_with_the_connections(self):
        mixed = Recipe.chain(
            Input(), Hidden(), Hidden(), _HEAD, delta=[0.1, None, 2.0], delta_ratio=[None, 3, None]
        )

        assert [(c.delta, c.delta_ratio) for c in mixed.connections] == [
            (0.1, None),
            (None, 3.0),
            (2.0, None),
        ]
        assert [c.delta for c in mixed.resolve_ratios().connections] == [0.1, 0.1 * 3, 2.0]

    def test_a_value_and_a_ratio_are_exchanged_together(self):
        ratio, _, _, _ = _ratio_case()

        with pytest.raises(ValueError, match="cannot be combined with an explicit delta"):
            ratio.replace_connection("hidden_1_to_hidden_2", delta=0.5)
        restored = ratio.replace_connection("hidden_1_to_hidden_2", delta=0.5, delta_ratio=None)

        assert [c.delta for c in restored.resolve_ratios().connections] == [
            _FIRST,
            0.5,
            0.5 * _RATIO,
        ]
        assert link("a", "b").delta == 1.0

    @pytest.mark.parametrize(
        "value",
        [0, -0.1, float("nan"), float("inf"), True, "3", 10**10000],
        ids=["zero", "negative", "nan", "inf", "bool", "str", "oversized"],
    )
    def test_rejects_a_ratio_that_is_not_finite_and_positive(self, value):
        with pytest.raises(ValueError, match=r"delta_ratio.*positive"):
            link("a", "b", delta_ratio=value)

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            pytest.param(
                lambda: link("a", "b", delta=0.5, delta_ratio=_RATIO),
                "cannot be combined with an explicit delta",
                id="explicit-delta",
            ),
            pytest.param(
                lambda: Recipe.chain(
                    Input(), Hidden(), _HEAD, delta=[0.1, 0.3], delta_ratio=[None, _RATIO]
                ),
                "cannot be combined with an explicit delta",
                id="chain-delta-beside-a-ratio",
            ),
            pytest.param(
                lambda: Recipe.chain(Input(), Hidden(), _HEAD, delta_ratio=[_RATIO, None]),
                "requires a previous connection",
                id="first-connection",
            ),
            pytest.param(
                lambda: Recipe.chain(Input(), _HEAD, delta_ratio=_RATIO),
                r"one connection.*delta_ratio",
                id="one-connection",
            ),
            pytest.param(
                lambda: Recipe.chain(Input(), Hidden(), _HEAD, delta_ratio=[None, 3, None]),
                r"delta_ratio.*one value per",
                id="long-sequence",
            ),
            pytest.param(
                lambda: _dag(link("second", "output", delta_ratio=_RATIO)),
                r"delta_ratio.*exactly one incoming",
                id="several-incoming",
            ),
            pytest.param(
                lambda: Recipe.chain(Input(), Hidden(), _HEAD, delta=1e200, delta_ratio=1e200),
                r"delta_ratio.*finite and positive",
                id="overflow",
            ),
            pytest.param(
                lambda: Recipe.chain(Input(), Hidden(), _HEAD, delta=1e-200, delta_ratio=1e-200),
                r"delta_ratio.*finite and positive",
                id="underflow",
            ),
        ],
    )
    def test_rejects_a_ratio_the_graph_cannot_resolve(self, build, message):
        _dag(link("second", "output"))
        with pytest.raises(ValueError, match=message):
            build()


@FIXED_BUDGET
class TestDeltaRatioFit:
    def test_a_fit_equals_the_fit_with_explicit_values(self):
        ratio, explicit, X, y = _ratio_case()

        fitted = Network.fit(ratio, X, y, **_CONTROLS)
        expected = Network.fit(explicit, X, y, **_CONTROLS)
        flat = Network.fit(
            explicit.replace_connection("hidden_2_to_output", delta=_FIRST), X, y, **_CONTROLS
        )

        assert fitted.recipe == ratio
        _assert_same_fit(fitted, expected)
        assert flat.diagnostics.loss_history != expected.diagnostics.loss_history

    @pytest.mark.parametrize(
        ("name", "change"),
        [("input_to_hidden_1", {"delta": 0.2}), ("hidden_1_to_hidden_2", {"delta_ratio": 2.0})],
        ids=["first-delta", "ratio"],
    )
    def test_fine_tuning_moves_every_delta_downstream(self, name, change):
        ratio, explicit, X, y = _ratio_case()
        tuned_recipe = ratio.replace_connection(name, **change)
        controls = dict(seed=5, max_iter=3, tol=0)

        tuned = Network.fit(ratio, X, y, **controls).fine_tune(
            X, y, recipe=tuned_recipe, max_iter=3
        )
        expected = Network.fit(explicit, X, y, **controls).fine_tune(
            X, y, recipe=tuned_recipe.resolve_ratios(), max_iter=3
        )

        assert tuned.recipe == tuned_recipe
        _assert_same_fit(tuned, expected)


@FIXED_BUDGET
class TestDeltaRatioPersistence:
    def test_a_saved_or_pickled_fit_keeps_the_ratios_and_resumes(self, tmp_path):
        ratio, explicit, X, y = _ratio_case()
        controls = dict(seed=5, max_iter=3, tol=0)
        fitted = Network.fit(ratio, X, y, **controls)
        expected = Network.fit(explicit, X, y, **controls).resume(X, y, max_iter=6, tol=0)

        loaded = round_trip_network(fitted, tmp_path / "ratio.safetensors", resumable=True)
        unpickled = pickle.loads(pickle.dumps(fitted))

        saved = _saved_connections(read_saved_metadata(tmp_path / "ratio.safetensors"))
        assert [(c["delta"], c["delta_ratio"]) for c in saved] == [
            (_FIRST, None),
            (None, _RATIO),
            (None, _RATIO),
        ]
        for restored in (loaded, unpickled):
            assert_same_checkpoint(restored, fitted)
            torch.testing.assert_close(restored.predict(X), fitted.predict(X), rtol=0, atol=0)
            resumed = restored.resume(X, y, max_iter=6, tol=0)
            assert resumed.recipe == ratio
            _assert_same_fit(resumed, expected)
