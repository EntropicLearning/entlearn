"""Public contract tests for immutable Recipe block descriptions."""

import pytest

from entlearn import (
    ClassificationHead,
    Connection,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    Recipe,
)


class TestRecipeDescriptions:
    @pytest.mark.parametrize(
        ("build", "message"),
        [
            pytest.param(
                lambda: Input(delta_cat=float("nan")), "delta_cat must be finite", id="delta_cat"
            ),
            pytest.param(
                lambda: Input(centroid_strategy="kmeans"),
                "centroid_strategy must be 'kmeans",
                id="centroid_strategy",
            ),
            pytest.param(
                lambda: Input(greedy_candidates=2),
                "greedy_candidates must be None",
                id="greedy_candidates-without-greedy",
            ),
            pytest.param(
                lambda: Input(centroid_strategy="greedy-kmeans++", greedy_candidates=True),
                "greedy_candidates must be an integer",
                id="greedy_candidates-bool",
            ),
            pytest.param(
                lambda: ManifoldInput(balanced=1), "balanced must be a bool", id="balanced"
            ),
            pytest.param(
                lambda: ClassificationHead(coupling=Coupling.M, n_classes=True),
                "n_classes must be a positive",
                id="n_classes",
            ),
        ],
    )
    def test_rejects_invalid_description_fields(self, build, message):
        with pytest.raises(ValueError, match=message):
            build()

    def test_a_greedy_strategy_leaves_the_candidate_count_to_its_default(self):
        assert ManifoldInput(centroid_strategy="greedy-kmeans++").greedy_candidates is None

    def test_coerces_string_couplings_and_rejects_other_values(self):
        assert ClassificationHead(coupling="M").coupling is Coupling.M
        assert Connection(name="c", source="a", target="b", coupling="S").coupling is Coupling.S
        chain = Recipe.chain(
            Input(), Hidden(), Hidden(), ClassificationHead(coupling="M"), coupling="S"
        )
        assert tuple(connection.coupling for connection in chain.connections[:2]) == (
            Coupling.S,
            Coupling.S,
        )
        listed = Recipe.chain(
            Input(),
            Hidden(),
            Hidden(),
            ClassificationHead(coupling="M"),
            coupling=["M", "S"],
        )
        assert tuple(connection.coupling for connection in listed.connections[:2]) == (
            Coupling.M,
            Coupling.S,
        )
        for invalid in ("X", "m", None):
            with pytest.raises(ValueError, match="'M' or 'S'"):
                ClassificationHead(coupling=invalid)
        with pytest.raises(ValueError, match="'M' or 'S'"):
            Connection(name="c", source="a", target="b", coupling=True)

    def test_a_classification_head_without_coupling_stores_m(self):
        head = ClassificationHead(n_classes=3)

        assert head.coupling is Coupling.M
        assert head == ClassificationHead(coupling=Coupling.M, n_classes=3)
