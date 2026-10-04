"""Public contract tests for Recipe graph structure and scalar validation."""

import pytest
from network._fixtures import link
from network._recipe_stubs import MutableConnection, MutableInput

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

_INPUT = Input(name="input", K=3)
_HIDDEN = Hidden(name="hidden", K=2)
_OUTPUT = ClassificationHead(name="output", coupling=Coupling.M)
_HEAD = ClassificationHead(coupling=Coupling.M)
_OVERSIZED = 10**10000


class TestRecipe:
    def test_replace_block_returns_a_new_valid_immutable_recipe(self):
        original = Recipe.chain(Input(K=2), ClassificationHead(coupling=Coupling.M))

        replacement = original.replace_block("input", K=3)

        assert replacement is not original
        assert replacement.blocks[0].K == 3
        assert original.blocks[0].K == 2
        assert replacement.connections == original.connections
        with pytest.raises(AttributeError):
            replacement.blocks = original.blocks
        with pytest.raises(KeyError, match="no block named 'missing'"):
            original.replace_block("missing", K=3)
        with pytest.raises(ValueError, match="block names must be unique"):
            original.replace_block("input", name="output")

    def test_replace_connection_replaces_one_connection(self):
        original = Recipe.chain(
            Input(K=3),
            Hidden(K=2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
        )

        replacement = original.replace_connection(
            "input_to_hidden_1", delta=0.5, coupling=Coupling.S
        )

        assert replacement.connections[0].delta == 0.5
        assert replacement.connections[0].coupling is Coupling.S
        assert replacement.connections[1] == original.connections[1]
        assert original.connections[0].delta == 1.0
        assert replacement.blocks == original.blocks
        with pytest.raises(KeyError, match="no connection named 'missing'"):
            original.replace_connection("missing", delta=0.5)
        with pytest.raises(ValueError, match="head-target connection cannot carry"):
            original.replace_connection("hidden_1_to_output", coupling=Coupling.S)

    def test_stores_descriptions_as_tuples_in_declared_order(self):
        input_block = Input(name="input", K=3)
        output_block = ClassificationHead(name="output", coupling=Coupling.M)
        connection = link("input", "output")

        recipe = Recipe(
            blocks=[input_block, output_block],
            connections=[connection],
        )

        assert recipe.blocks == (input_block, output_block)
        assert recipe.connections == (connection,)

    def test_rejects_a_mutable_connection(self):
        with pytest.raises(ValueError, match="connections must be valid Recipe connection"):
            Recipe(
                blocks=(
                    Input(name="input", K=3),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(
                    MutableConnection(name="input_to_output", source="input", target="output"),
                ),
            )

    def test_rejects_a_block_subclass_with_mutable_state(self):
        with pytest.raises(ValueError, match="blocks must be valid Recipe block"):
            Recipe(
                blocks=(
                    MutableInput(name="input", K=3),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(link("input", "output"),),
            )

    def test_rejects_names_containing_the_parameter_path_token(self):
        with pytest.raises(ValueError, match="block names cannot contain"):
            Recipe(
                blocks=(
                    Input(name="in__put", K=3),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(link("in__put", "output"),),
            )
        with pytest.raises(ValueError, match="connection names cannot contain"):
            Recipe(
                blocks=(
                    Input(name="input", K=3),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(link("input", "output", name="input__output"),),
            )

    @pytest.mark.parametrize("name", ["_input", "input_"])
    def test_rejects_names_whose_boundary_underscore_forms_the_path_token(self, name):
        with pytest.raises(ValueError, match="block names cannot start or end"):
            Recipe(
                blocks=(
                    Input(name=name, K=3),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(link(name, "output", name="link"),),
            )
        with pytest.raises(ValueError, match="connection names cannot start or end"):
            Recipe(
                blocks=(
                    Input(name="input", K=3),
                    ClassificationHead(name="output", coupling=Coupling.M),
                ),
                connections=(link("input", "output", name=name),),
            )

    def test_chain_reports_a_boundary_underscore_against_the_block_the_caller_named(self):
        with pytest.raises(ValueError, match="block names cannot start or end with '_'"):
            Recipe.chain(
                Input(name="input_", K=3),
                ClassificationHead(name="output", coupling=Coupling.M),
            )

    @pytest.mark.parametrize(
        ("blocks", "connections", "message"),
        [
            pytest.param(
                (_INPUT, Input(name="input"), _OUTPUT),
                (),
                "block names must be unique",
                id="duplicate-name",
            ),
            pytest.param(
                (_INPUT, _OUTPUT),
                (link("input", "missing"),),
                "endpoints must name Recipe blocks",
                id="missing-endpoint",
            ),
            pytest.param(
                (_INPUT, _HIDDEN, _OUTPUT),
                (link("hidden", "hidden"),),
                "cannot join a block to itself",
                id="self-loop",
            ),
            pytest.param((_HIDDEN, _OUTPUT), (), "at least one input block", id="no-input"),
            pytest.param((_INPUT, _HIDDEN), (), "at least one head", id="no-head"),
            pytest.param(
                (_INPUT, _OUTPUT),
                (link("input", "output", coupling=Coupling.M),),
                "head-target connection cannot carry",
                id="head-target-with-coupling",
            ),
            pytest.param(
                (_INPUT, _OUTPUT),
                (link("input", "output", theta_alpha=1.0),),
                "head-target connection cannot carry",
                id="head-target-with-theta-alpha",
            ),
        ],
    )
    def test_validates_graph_structure_and_connection_endpoints(self, blocks, connections, message):
        with pytest.raises(ValueError, match=message):
            Recipe(blocks=blocks, connections=connections)

    def test_a_hidden_target_connection_without_coupling_stores_m(self):
        blocks = (_INPUT, _HIDDEN, _OUTPUT)
        declared = Recipe(
            blocks=blocks,
            connections=(link("input", "hidden", theta_alpha=1.0), link("hidden", "output")),
        )
        explicit = Recipe(
            blocks=blocks,
            connections=(
                link("input", "hidden", coupling=Coupling.M, theta_alpha=1.0),
                link("hidden", "output"),
            ),
        )

        assert declared.connections[0].coupling is Coupling.M
        assert declared.connections[1].coupling is None
        assert declared == explicit
        assert repr(declared) == repr(explicit)
        switched = declared.replace_connection("input_to_hidden", coupling=Coupling.S)
        assert switched.replace_connection("input_to_hidden", coupling=None) == explicit

    def test_a_hidden_target_connection_without_coupling_or_theta_alpha_takes_the_defaults(self):
        blocks = (_INPUT, _HIDDEN, _OUTPUT)
        head_link = link("hidden", "output")
        declared = Recipe(blocks=blocks, connections=(link("input", "hidden"), head_link))

        assert declared.connections[0].coupling is Coupling.M
        assert declared.connections[0].theta_alpha == 1.0
        assert declared.connections[1] is head_link
        assert declared == Recipe.chain(*blocks)
        assert repr(declared) == repr(Recipe.chain(*blocks))
        tuned = declared.replace_connection("input_to_hidden", theta_alpha=2.0)
        assert tuned.connections[0].theta_alpha == 2.0
        assert tuned.replace_connection("input_to_hidden", theta_alpha=None) == declared
        explicit = link("input", "hidden", coupling=Coupling.S, theta_alpha=2.0)
        assert Recipe(blocks=blocks, connections=(explicit, head_link)).connections[0] is explicit

    def test_accepts_a_structurally_valid_future_dag(self):
        recipe = Recipe(
            blocks=(
                Input(name="input_a", K=3),
                Input(name="input_b", K=3),
                Hidden(name="join", K=2),
                ClassificationHead(name="output_a", coupling=Coupling.M),
                ClassificationHead(name="output_b", coupling=Coupling.S),
            ),
            connections=(
                link("input_a", "join", coupling=Coupling.M, theta_alpha=1.0),
                link("input_b", "join", coupling=Coupling.S, theta_alpha=2.0),
                link("join", "output_a"),
                link("join", "output_b"),
            ),
        )

        assert tuple(block.name for block in recipe.blocks) == (
            "input_a",
            "input_b",
            "join",
            "output_a",
            "output_b",
        )

    def test_chain_assigns_canonical_names_and_hidden_connection_fields(self):
        recipe = Recipe.chain(
            Input(K=3),
            Hidden(name="middle", K=2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.S,
        )

        assert tuple(block.name for block in recipe.blocks) == ("input", "middle", "output")
        assert recipe.connections == (
            Connection(
                name="input_to_middle",
                source="input",
                target="middle",
                coupling=Coupling.S,
                theta_alpha=1.0,
            ),
            Connection(name="middle_to_output", source="middle", target="output"),
        )

    def test_chain_applies_connection_scalars(self):
        recipe = Recipe.chain(
            Input(K=3),
            Hidden(K=2),
            _HEAD,
            coupling=Coupling.S,
            delta=[0.5, 2.0],
            theta_alpha=3.0,
        )

        assert tuple(connection.delta for connection in recipe.connections) == (0.5, 2.0)
        assert recipe.connections[0].theta_alpha == 3.0

    @pytest.mark.parametrize("n_hidden", [1, 3])
    def test_chain_without_coupling_gives_every_hidden_target_connection_m(self, n_hidden):
        blocks = (Input(K=3), *(Hidden(K=2) for _ in range(n_hidden)), _HEAD)

        recipe = Recipe.chain(*blocks)

        assert tuple(connection.coupling for connection in recipe.connections) == (
            *(Coupling.M,) * n_hidden,
            None,
        )
        assert recipe == Recipe.chain(*blocks, coupling=Coupling.M)

    @pytest.mark.parametrize(
        ("blocks", "fields", "message"),
        [
            pytest.param(
                (Input(), Hidden(), _HEAD),
                {"coupling": "X"},
                "coupling must be 'M' or 'S'",
                id="invalid-coupling",
            ),
            pytest.param(
                (Input(), Hidden(), Hidden(), _HEAD),
                {"coupling": [Coupling.M]},
                "one Coupling value per hidden block",
                id="short-coupling",
            ),
            pytest.param(
                (Hidden(), Input(), _HEAD),
                {},
                "must start with an input block",
                id="hidden-first",
            ),
            pytest.param(
                (Input(), Input(), _HEAD),
                {},
                "cannot contain an interior input block or head",
                id="interior-input",
            ),
            pytest.param(
                (Input(), _HEAD, _HEAD),
                {},
                "cannot contain an interior input block or head",
                id="interior-head",
            ),
            pytest.param(
                (Input(), "not a block", _HEAD),
                {},
                "chain requires Recipe block descriptions",
                id="not-a-block",
            ),
            pytest.param(
                (Input(), Hidden(), _HEAD),
                {"coupling": Coupling.M, "delta": [1.0]},
                "delta must provide one value",
                id="short-delta",
            ),
            pytest.param(
                (Input(), Hidden(), _HEAD),
                {"coupling": Coupling.M, "theta_alpha": [1.0, 2.0]},
                "theta_alpha must provide one value",
                id="long-theta-alpha",
            ),
            pytest.param(
                (Input(), _HEAD),
                {"theta_alpha": 0.0},
                "theta_alpha must be finite",
                id="theta-alpha-below-one",
            ),
            pytest.param(
                (Input(), _HEAD),
                {"coupling": Coupling.M},
                "without hidden blocks cannot receive coupling",
                id="coupling-without-hidden",
            ),
        ],
    )
    def test_chain_rejects_invalid_arguments(self, blocks, fields, message):
        with pytest.raises(ValueError, match=message):
            Recipe.chain(*blocks, **fields)

    def test_preserves_declared_direct_recipe_order(self):
        blocks = (
            Hidden(name="hidden", K=2),
            Input(name="input", K=3),
            ClassificationHead(name="output", coupling=Coupling.M),
        )

        recipe = Recipe(
            blocks=blocks,
            connections=(
                link("input", "hidden", coupling=Coupling.M, theta_alpha=1.0),
                link("hidden", "output"),
            ),
        )

        assert recipe.blocks == blocks

    def test_stable_order_follows_topology_then_declared_order(self):
        chain = Recipe(
            blocks=(
                Hidden(name="downstream", K=2),
                ClassificationHead(name="output", coupling=Coupling.M),
                Input(name="input", K=3),
                Hidden(name="upstream", K=4),
            ),
            connections=(
                link("downstream", "output"),
                link("input", "upstream", coupling=Coupling.M, theta_alpha=1.0),
                link("upstream", "downstream", coupling=Coupling.S, theta_alpha=1.0),
            ),
        )
        tied = Recipe(
            blocks=(
                Input(name="second", K=3),
                Input(name="first", K=3),
                Hidden(name="join", K=2),
                ClassificationHead(name="output", coupling=Coupling.M),
            ),
            connections=(
                link("first", "join", coupling=Coupling.M, theta_alpha=1.0),
                link("second", "join", coupling=Coupling.M, theta_alpha=1.0),
                link("join", "output"),
            ),
        )

        assert chain.stable_order() == ("input", "upstream", "downstream", "output")
        assert tied.stable_order() == ("second", "first", "join", "output")

    # Validation scans the blocks in declared order and raises on the first violation, so
    # in hidden-without-incoming the unconnected input fails the input-block rule before
    # the sourceless hidden block is reached. A head or hidden block without an incoming
    # connection fails the graph-source rule (head-without-incoming), and a hidden block
    # without an outgoing one the graph-sink rule.
    @pytest.mark.parametrize(
        ("blocks", "connections", "message"),
        [
            pytest.param(
                (_INPUT, _HIDDEN, _OUTPUT),
                (link("hidden", "output"),),
                "input block must have an outgoing",
                id="hidden-without-incoming",
            ),
            pytest.param(
                (_INPUT, Input(name="input_b", K=3), _OUTPUT),
                (link("input", "output"),),
                "input block must have an outgoing",
                id="input-without-outgoing",
            ),
            pytest.param(
                (_INPUT, _HIDDEN, _OUTPUT),
                (
                    link("input", "hidden", coupling=Coupling.M, theta_alpha=1.0),
                    link("input", "output"),
                ),
                "graph output must be a head",
                id="hidden-without-outgoing",
            ),
            pytest.param(
                (_INPUT, _OUTPUT, ClassificationHead(name="output_b", coupling=Coupling.M)),
                (link("input", "output"),),
                "graph source must be an input block",
                id="head-without-incoming",
            ),
        ],
    )
    def test_rejects_misplaced_sources_and_sinks(self, blocks, connections, message):
        with pytest.raises(ValueError, match=message):
            Recipe(blocks=blocks, connections=connections)

    def test_validates_sources_and_sinks_without_rejecting_future_dags(self):
        first_input = Input(name="first_input", K=3)
        second_input = Input(name="second_input", K=3)
        first_output = ClassificationHead(name="first_output", coupling=Coupling.M)
        second_output = ClassificationHead(name="second_output", coupling=Coupling.M)

        recipe = Recipe(
            blocks=(first_input, second_input, first_output, second_output),
            connections=(
                link("first_input", "first_output"),
                link("second_input", "second_output"),
            ),
        )
        assert recipe.blocks == (first_input, second_input, first_output, second_output)

    def test_rejects_balanced_input_for_regression(self):
        with pytest.raises(ValueError, match="cannot use balanced input"):
            Recipe(
                blocks=(
                    Input(name="input", K=3, balanced=True),
                    RegressionHead(name="output"),
                ),
                connections=(link("input", "output"),),
            )

    @pytest.mark.parametrize(
        ("blocks", "connections", "message"),
        [
            pytest.param(
                (_INPUT, Hidden(name="first", K=2), Hidden(name="second", K=2), _OUTPUT),
                (
                    link("input", "first", coupling=Coupling.M, theta_alpha=1.0),
                    link("first", "second", coupling=Coupling.M, theta_alpha=1.0),
                    link("second", "first", coupling=Coupling.M, theta_alpha=1.0),
                    link("second", "output"),
                ),
                "must be acyclic",
                id="cycle",
            ),
            pytest.param(
                (_INPUT, _OUTPUT, RegressionHead(name="regression")),
                (),
                "cannot mix classification and regression",
                id="mixed-heads",
            ),
        ],
    )
    def test_rejects_cycles_and_mixed_head_tasks(self, blocks, connections, message):
        with pytest.raises(ValueError, match=message):
            Recipe(blocks=blocks, connections=connections)


class TestRecipeScalarValidation:
    @pytest.mark.parametrize(
        ("description", "fields", "message"),
        [
            (Input, {"K": True}, "K must be a positive"),
            (ManifoldInput, {"subspace_dimension": 0}, "subspace_dimension must be a positive"),
            (Hidden, {"K": 0}, "K must be a positive"),
            (Input, {"epsilon": float("inf")}, "epsilon must be finite"),
            (Input, {"epsilon_D": float("-inf")}, "epsilon_D must be non-negative"),
            (Input, {"epsilon_T": float("nan")}, "epsilon_T must be non-negative"),
            (Input, {"W_std": float("inf")}, "W_std must be finite"),
            (ManifoldInput, {"alpha": -1.0}, "alpha must be finite"),
            (ManifoldInput, {"epsilon": float("nan")}, "epsilon must be finite"),
            (ManifoldInput, {"epsilon_T": -1.0}, "epsilon_T must be non-negative"),
            (RegressionHead, {"epsilon_M": float("nan")}, "epsilon_M must be non-negative"),
            (RegressionHead, {"W_M": []}, "W_M must be a non-empty"),
            (RegressionHead, {"W_M": [1.0, 0.0]}, "W_M must be a non-empty"),
            (RegressionHead, {"W_M": [float("nan")]}, "W_M must be a non-empty"),
            (RegressionHead, {"epsilon_M": 0.0, "W_M": [1.0]}, "requires epsilon_M to be infinite"),
        ],
    )
    def test_rejects_invalid_block_scalars(self, description, fields, message):
        with pytest.raises(ValueError, match=message):
            description(**fields)

    def test_canonicalises_fixed_output_weights(self):
        head = RegressionHead(W_M=[2.0, 3.0])

        assert head.W_M == (2.0, 3.0)

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"delta": 0.0}, "delta must be finite"),
            ({"delta": float("nan")}, "delta must be finite"),
            ({"theta_alpha": 0.5}, "theta_alpha must be finite"),
            ({"theta_alpha": float("inf")}, "theta_alpha must be finite"),
            ({"name": ""}, "name, source and target must be non-empty strings"),
            ({"source": ["a"]}, "name, source and target must be non-empty strings"),
            ({"target": 5}, "name, source and target must be non-empty strings"),
        ],
    )
    def test_rejects_invalid_connection_fields(self, fields, message):
        with pytest.raises(ValueError, match=message):
            Connection(**{"name": "bad", "source": "a", "target": "b", **fields})

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            pytest.param(lambda: Input(epsilon=_OVERSIZED), "epsilon must be finite", id="epsilon"),
            pytest.param(
                lambda: Input(epsilon_D=_OVERSIZED),
                "epsilon_D must be non-negative",
                id="epsilon_D",
            ),
            pytest.param(
                lambda: Connection(name="bad", source="a", target="b", delta=_OVERSIZED),
                "delta must be finite",
                id="delta",
            ),
            pytest.param(
                lambda: RegressionHead(W_M=[_OVERSIZED]), "W_M must be a non-empty", id="W_M"
            ),
        ],
    )
    def test_rejects_oversized_integers_without_overflow(self, build, message):
        with pytest.raises(ValueError, match=message):
            build()
