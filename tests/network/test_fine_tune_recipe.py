"""Replacing a fitted Network's Recipe objective when fine-tuning."""

import math
from dataclasses import replace

import pytest
import torch
from conftest import FIXED_BUDGET
from network._fixtures import resume_case

from entlearn import (
    Connection,
    Coupling,
    Hidden,
    ManifoldInput,
    Network,
    Recipe,
)

pytestmark = FIXED_BUDGET


class TestObjectiveReplacement:
    @pytest.mark.parametrize(
        "kind,index,field,value",
        [
            ("standard", 0, "epsilon", 0.3),
            ("standard", 0, "epsilon_D", 0.3),
            ("standard", 0, "epsilon_T", 0.4),
            ("categorical", 0, "delta_cat", 0.0),
            ("manifold", 0, "epsilon", 0.3),
            ("manifold", 0, "epsilon_T", 0.4),
            ("manifold", 0, "alpha", 0.4),
            ("standard", 1, "epsilon", 0.3),
            ("standard", -1, "epsilon_M", 0.4),
        ],
    )
    def test_each_allowed_block_objective_replaces_completely(self, kind, index, field, value):
        recipe, X, y, cats = resume_case("regression", kind)
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=2)
        fitted = source.recipe
        blocks = list(fitted.blocks)
        blocks[index] = replace(blocks[index], **{field: value})
        new = replace(fitted, blocks=tuple(blocks))
        result = source.fine_tune(X, y, X_cat=cats, recipe=new, max_iter=2)
        unchanged = source.fine_tune(X, y, X_cat=cats, max_iter=2)
        # The replaced value must reach the objective, so the losses part from the start.
        assert result.diagnostics.loss_history[0] != unchanged.diagnostics.loss_history[0]
        assert source.recipe == fitted
        assert result.can_resume

    # The first connection's delta is test_a_replaced_connection_strength_reaches_the_fitted_parameters.
    @pytest.mark.parametrize("index,field,value", [(0, "theta_alpha", 1.2), (1, "delta", 1.2)])
    def test_connection_objectives_are_replaceable(self, index, field, value):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        connections = list(recipe.connections)
        connections[index] = replace(connections[index], **{field: value})
        new = replace(recipe, connections=tuple(connections))
        result = source.fine_tune(X, y, recipe=new, max_iter=2)
        unchanged = source.fine_tune(X, y, max_iter=2)
        assert result.diagnostics.loss_history[0] != unchanged.diagnostics.loss_history[0]

    def test_the_head_prior_is_replaceable(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        new = recipe.replace_block("output", theta_alpha=1.5)
        result = source.fine_tune(X, y, recipe=new, max_iter=2)
        unchanged = source.fine_tune(X, y, max_iter=2)
        assert result.diagnostics.loss_history[0] != unchanged.diagnostics.loss_history[0]
        assert result.recipe == new

    def test_a_replaced_connection_strength_reaches_the_fitted_parameters(self):
        # The owned graph keeps its fitted weights, so only the refreshed connection
        # descriptions can make a changed strength show in the refit.
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        connections = list(recipe.connections)
        connections[0] = replace(connections[0], delta=1.5)
        stronger = replace(recipe, connections=tuple(connections))

        unchanged = source.fine_tune(X, y, recipe=recipe, max_iter=3, tol=0)
        tuned = source.fine_tune(X, y, recipe=stronger, max_iter=3, tol=0)

        assert not torch.allclose(tuned.predict(X), unchanged.predict(X))

    def test_replacement_freezes_transferred_weights_and_new_row_weights(self):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=3, tol=0, seed=5)
        new = replace(
            recipe,
            blocks=(
                replace(recipe.blocks[0], epsilon_D=math.inf, epsilon_T=math.inf),
                recipe.blocks[1],
                replace(recipe.blocks[-1], epsilon_M=math.inf),
            ),
        )
        weights = torch.arange(1, 11, dtype=X.dtype, device=X.device)
        result = source.fine_tune(
            X[:10], y[:10], recipe=new, sample_weights=weights, max_iter=2, tol=0
        )
        assert result.recipe == new and source.recipe == recipe
        torch.testing.assert_close(
            result.inspect("feature_weights")["input"],
            source.inspect("feature_weights")["input"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.inspect("head_parameters")["output"]["W_M"],
            source.inspect("head_parameters")["output"]["W_M"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.inspect("training_instance_weights")["input"],
            weights / weights.sum(),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("fixed", [(0.8, 0.2), None])
    def test_output_weight_freezing_and_unfreezing(self, fixed):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=3, tol=0)
        frozen_recipe = replace(
            recipe,
            blocks=(*recipe.blocks[:-1], replace(recipe.blocks[-1], epsilon_M=math.inf, W_M=fixed)),
        )
        frozen = source.fine_tune(X[:10], y[:10], recipe=frozen_recipe, max_iter=2)
        expected = (
            source.inspect("head_parameters")["output"]["W_M"]
            if fixed is None
            else torch.tensor(fixed, dtype=X.dtype, device=X.device)
        )
        torch.testing.assert_close(
            frozen.inspect("head_parameters")["output"]["W_M"], expected, rtol=0, atol=0
        )
        unfrozen = frozen.fine_tune(X[:15], y[:15], recipe=recipe, max_iter=2)
        assert not torch.equal(
            unfrozen.inspect("head_parameters")["output"]["W_M"],
            frozen.inspect("head_parameters")["output"]["W_M"],
        )

    def test_uniform_output_weights_become_learnable(self):
        recipe, X, y, _ = resume_case("regression")
        uniform = replace(
            recipe, blocks=(*recipe.blocks[:-1], replace(recipe.blocks[-1], epsilon_M=math.inf))
        )
        source = Network.fit(uniform, X, y, max_iter=3, tol=0, seed=5)
        assert "W_M" not in source.inspect("head_parameters")["output"]

        result = source.fine_tune(X, y, recipe=recipe, max_iter=3, tol=0)
        fixed = source.fine_tune(X, y, recipe=uniform, max_iter=3, tol=0)

        W_M = result.inspect("head_parameters")["output"]["W_M"]
        assert W_M.shape == (y.shape[1],)
        assert bool((W_M >= 0).all())
        torch.testing.assert_close(W_M.sum(), torch.ones((), dtype=W_M.dtype, device=W_M.device))
        assert not torch.allclose(W_M, torch.full_like(W_M, 1 / W_M.shape[0]))
        # A uniform start adds only its entropy term, -epsilon_M log M, to the start loss.
        entropy = recipe.blocks[-1].epsilon_M * math.log(y.shape[1])
        assert result.diagnostics.loss_history[0] == pytest.approx(
            fixed.diagnostics.loss_history[0] - entropy, abs=4 * torch.finfo(X.dtype).eps
        )

    def test_fixed_output_weights_must_keep_the_number_of_targets(self):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2)
        new = replace(
            recipe,
            blocks=(
                *recipe.blocks[:-1],
                replace(recipe.blocks[-1], epsilon_M=math.inf, W_M=(1.0,)),
            ),
        )
        with pytest.raises(ValueError, match=r"W_M has length 1 .* regression targets is 2"):
            source.fine_tune(X, y, recipe=new)


class TestReplacementWhitelist:
    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    @pytest.mark.parametrize(
        "field,value", [("K", 5), ("balanced", True), ("centroid_strategy", "greedy-kmeans++")]
    )
    def test_geometry_controls_cannot_change(self, kind, field, value):
        recipe, X, y, _ = resume_case(kind=kind)
        source = Network.fit(recipe, X, y, max_iter=2)
        new = replace(
            recipe, blocks=(replace(recipe.blocks[0], **{field: value}), *recipe.blocks[1:])
        )
        with pytest.raises(ValueError, match=field):
            source.fine_tune(X, y, recipe=new)

    @pytest.mark.parametrize("field,value", [("W_std", 0.1), ("greedy_candidates", 4)])
    def test_standard_initialisation_controls_cannot_change(self, field, value):
        recipe, X, y, _ = resume_case()
        recipe = replace(
            recipe,
            blocks=(
                replace(recipe.blocks[0], centroid_strategy="greedy-kmeans++", greedy_candidates=3),
                *recipe.blocks[1:],
            ),
        )
        source = Network.fit(recipe, X, y, max_iter=2)
        new = replace(
            recipe, blocks=(replace(recipe.blocks[0], **{field: value}), *recipe.blocks[1:])
        )
        with pytest.raises(ValueError, match=field):
            source.fine_tune(X, y, recipe=new)

    def test_manifold_dimension_is_structural(self):
        recipe, X, y, _ = resume_case(kind="manifold")
        source = Network.fit(recipe, X, y, max_iter=2)
        new = replace(
            recipe, blocks=(replace(recipe.blocks[0], subspace_dimension=2), *recipe.blocks[1:])
        )
        with pytest.raises(ValueError, match="subspace_dimension"):
            source.fine_tune(X, y, recipe=new)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ("hidden_width", "cannot change 'hidden_1' parameter 'K'"),
            ("head_width", "cannot change 'output' parameter 'n_classes'"),
            ("head_coupling", "cannot change 'output' parameter 'coupling'"),
            ("connection_coupling", "cannot change 'input_to_hidden_1' parameter 'coupling'"),
            ("names", "cannot change 'hidden_1' parameter 'name'"),
            ("kind", "cannot change block or connection kinds"),
            ("order", "cannot change block or connection kinds"),
            ("topology", "cannot change topology"),
            ("skip", "cannot change topology"),
        ],
    )
    def test_structural_fields_cannot_change(self, change, message):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        blocks, connections = list(recipe.blocks), list(recipe.connections)
        if change == "hidden_width":
            blocks[1] = replace(blocks[1], K=4)
        elif change == "head_width":
            blocks[-1] = replace(blocks[-1], n_classes=3)
        elif change == "head_coupling":
            blocks[-1] = replace(blocks[-1], coupling=Coupling.S)
        elif change == "connection_coupling":
            connections[0] = replace(connections[0], coupling=Coupling.S)
        elif change == "names":
            blocks[1] = replace(blocks[1], name="renamed")
            connections[0] = replace(connections[0], target="renamed", name="renamed_in")
            connections[1] = replace(connections[1], source="renamed")
        elif change == "kind":
            blocks[0] = ManifoldInput(name="input", K=4)
        elif change == "order":
            blocks[0], blocks[1] = blocks[1], blocks[0]
        elif change == "skip":
            connections.append(Connection("skip", "input", "output"))
        else:
            blocks.insert(2, Hidden(name="extra"))
            connections.insert(
                1, Connection("extra_in", "hidden_1", "extra", coupling=Coupling.M, theta_alpha=1)
            )
            connections[-1] = replace(connections[-1], source="extra")
        with pytest.raises(ValueError, match=message):
            source.fine_tune(X, y, recipe=Recipe(tuple(blocks), tuple(connections)))
