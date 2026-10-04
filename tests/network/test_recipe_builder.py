"""Public contract tests for the private Recipe builder, through Network."""

import pytest
import torch
from network._fixtures import link
from network._recipe_stubs import UnreadableData

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    Network,
    Recipe,
)


class TestRecipeBuilder:
    @pytest.mark.parametrize(
        "recipe",
        (
            Recipe(
                blocks=(
                    Input(name="input", K=3),
                    Hidden(name="hidden", K=2),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(
                    link("input", "hidden", coupling=Coupling.M, theta_alpha=1.0),
                    link("hidden", "output"),
                    link("input", "output"),
                ),
            ),
            Recipe(
                blocks=(
                    Input(name="input", K=3),
                    Hidden(name="left", K=2),
                    Hidden(name="right", K=2),
                    Hidden(name="join", K=2),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(
                    link("input", "left", coupling=Coupling.M, theta_alpha=1.0),
                    link("input", "right", coupling=Coupling.M, theta_alpha=1.0),
                    link("left", "join", coupling=Coupling.M, theta_alpha=1.0),
                    link("right", "join", coupling=Coupling.M, theta_alpha=1.0),
                    link("join", "output"),
                ),
            ),
            Recipe(
                blocks=(
                    Input(name="input_a", K=3),
                    Input(name="input_b", K=3),
                    Hidden(name="hidden", K=2),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(
                    link("input_a", "hidden", coupling=Coupling.M, theta_alpha=1.0),
                    link("input_b", "hidden", coupling=Coupling.M, theta_alpha=1.0),
                    link("hidden", "output"),
                ),
            ),
            Recipe(
                blocks=(
                    Input(name="input", K=3),
                    ClassificationHead(name="output_a", coupling=Coupling.M),
                    ClassificationHead(name="output_b", coupling=Coupling.S),
                ),
                connections=(link("input", "output_a"), link("input", "output_b")),
            ),
            Recipe(
                blocks=(
                    Input(name="input_a", K=3),
                    ClassificationHead(name="output_a", coupling=Coupling.M),
                    Input(name="input_b", K=3),
                    ClassificationHead(name="output_b", coupling=Coupling.S),
                ),
                connections=(link("input_a", "output_a"), link("input_b", "output_b")),
            ),
        ),
        ids=("skip", "join", "multiple-inputs", "multiple-heads", "disconnected"),
    )
    def test_rejects_future_topologies_before_data_access(self, recipe):
        with pytest.raises(ValueError, match="the builder only supports one connected chain"):
            Network.initialise(recipe, UnreadableData(), UnreadableData())

    def test_builds_a_non_topological_recipe_with_stable_public_identity(self):
        blocks = (
            Hidden(name="downstream", K=2),
            ClassificationHead(name="output", coupling=Coupling.M),
            Input(name="input", K=3),
            Hidden(name="upstream", K=4),
        )
        connections = (
            link("downstream", "output"),
            link("input", "upstream", coupling=Coupling.M, theta_alpha=1.0),
            link("upstream", "downstream", coupling=Coupling.S, theta_alpha=1.0),
        )
        recipe = Recipe(blocks=blocks, connections=connections)
        X_cont = torch.arange(12, dtype=torch.float64).reshape(6, 2) / 16
        y = torch.tensor([0, 1, 0, 1, 0, 1])

        state = Network.initialise(recipe, X_cont, y, seed=5)

        assert recipe.blocks == blocks
        assert state.input_geometry.input == blocks[2]
        assert state.input_geometry.K_active == 3
        assert tuple(name for name, _ in state.connection_sub_seeds) == tuple(
            connection.name for connection in connections
        )
