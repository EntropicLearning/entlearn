"""Profile capacity required only when seeding hidden connections."""

from entlearn.network.build import _BuiltRecipe
from entlearn.recipe import Connection, Coupling, Hidden


def validate_connection_capacity(
    connection: Connection, source_width: int, target_width: int, rows: int
) -> None:
    """Require enough profiles when this connection needs fresh initialisation."""
    uses_profiles = connection.coupling is Coupling.S or source_width < target_width
    if uses_profiles and rows < target_width:
        raise ValueError(
            f"connection {connection.name!r} needs {target_width} profiles "
            f"but only {rows} rows are available"
        )


def validate_profile_capacity(recipe: _BuiltRecipe, rows: int) -> None:
    """Reject a fresh fit without enough rows to seed its hidden profiles."""
    K_source = recipe.input.K
    for block in recipe.topological_blocks[1:-1]:
        assert isinstance(block, Hidden)
        (connection,) = recipe.incoming[block.name]
        validate_connection_capacity(connection, K_source, block.K, rows)
        K_source = block.K
