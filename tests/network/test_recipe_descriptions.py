"""Public contract tests for immutable Recipe block descriptions."""

from typing import Any, cast

import pytest

from entlearn import (
    ClassificationHead,
    Connection,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    Recipe,
    RegressionHead,
)


class TestRecipeDescriptions:
    def test_descriptions_are_immutable(self):
        descriptions = (
            Input(K=3),
            ManifoldInput(K=3, subspace_dimension=1, alpha=0.0),
            Hidden(K=2),
            ClassificationHead(coupling=Coupling.M),
            RegressionHead(),
            Connection(name="input_to_hidden", source="input", target="hidden"),
        )

        for description in descriptions:
            original_name = description.name
            with pytest.raises(AttributeError):
                description.name = "other"
            assert description.name == original_name

    def test_carries_input_identity_fields_and_declared_class_counts(self):
        standard = Input(
            delta_cat=0.5,
            centroid_strategy="greedy-kmeans++",
            greedy_candidates=2,
            balanced=True,
        )
        manifold = ManifoldInput(
            centroid_strategy="greedy-kmeans++",
            greedy_candidates=2,
            balanced=True,
        )
        classification = ClassificationHead(coupling=Coupling.M, n_classes=3)
        regression = RegressionHead(W_M=[2.0, 3.0])

        assert standard.delta_cat == 0.5
        assert manifold.centroid_strategy == "greedy-kmeans++"
        assert classification.n_classes == 3
        assert regression.W_M == (2.0, 3.0)

        with pytest.raises(ValueError):
            Input(delta_cat=float("nan"))
        with pytest.raises(ValueError):
            Input(greedy_candidates=2)
        assert ManifoldInput(centroid_strategy="greedy-kmeans++").greedy_candidates is None
        with pytest.raises(ValueError):
            Input(
                centroid_strategy="greedy-kmeans++",
                greedy_candidates=True,
            )
        with pytest.raises(ValueError):
            ManifoldInput(balanced=1)
        with pytest.raises(ValueError):
            ClassificationHead(coupling=Coupling.M, n_classes=True)
        regression_head_constructor = cast(Any, RegressionHead)
        with pytest.raises(TypeError):
            regression_head_constructor(n_classes=2)

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
            with pytest.raises(ValueError, match="coupling must be 'M' or 'S'"):
                ClassificationHead(coupling=invalid)
        with pytest.raises(ValueError, match="coupling must be 'M' or 'S'"):
            Connection(name="c", source="a", target="b", coupling=True)

    def test_a_classification_head_without_coupling_stores_m(self):
        head = ClassificationHead(n_classes=3)

        assert head.coupling is Coupling.M
        assert head == ClassificationHead(coupling=Coupling.M, n_classes=3)
