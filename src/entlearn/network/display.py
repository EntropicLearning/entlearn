"""Optional Rich presentation of fitted Networks, imported only when rendering.

Read configuration, tensor placement metadata and stored diagnostics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from entlearn.recipe import (
    Block,
    ClassificationHead,
    Connection,
    Hidden,
    Input,
    ManifoldInput,
    RegressionHead,
)

if TYPE_CHECKING:
    from entlearn.network.model import Network

_MAX_DISPLAY_WIDTH = 92
_WIDTH_CRITERIA = 80


def _count(n: int, noun: str) -> str:
    """Return ``n`` followed by ``noun``, pluralised with a trailing 's' unless ``n == 1``."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _number(value: float) -> str:
    if math.isinf(value):
        return "-∞" if value < 0 else "∞"
    return f"{value:.6g}"


def _parameters(block: Block, *, categorical: bool) -> str:
    """Select the applicable training controls, including temperature defaults."""
    parameters = []
    if isinstance(block, (Input, ManifoldInput, Hidden)):
        parameters.append(f"ε={_number(block.epsilon)}")
    if isinstance(block, Input):
        parameters.append(f"ε_D={_number(block.epsilon_D)} · ε_T={_number(block.epsilon_T)}")
        if categorical:
            parameters.append(f"δ_cat={_number(block.delta_cat)}")
    elif isinstance(block, ManifoldInput):
        parameters.extend(
            [
                f"ε_T={_number(block.epsilon_T)}",
                f"subspace_dimension={block.subspace_dimension}",
                f"alpha={_number(block.alpha)}",
            ]
        )
    elif isinstance(block, ClassificationHead):
        parameters.append(f"coupling={block.coupling.value}")
    elif isinstance(block, RegressionHead):
        parameters.append(f"ε_M={_number(block.epsilon_M)}")
        if block.W_M is not None:
            parameters.append(f"W_M=fixed ({len(block.W_M)} outputs)")
    return "\n".join(parameters)


def _incoming(connection: Connection) -> str:
    settings = [f"δ={_number(connection.delta)}"]
    if connection.coupling is not None:
        settings.insert(0, connection.coupling.value)
    if connection.theta_alpha is not None:
        settings.append(f"theta_alpha={_number(connection.theta_alpha)}")
    return f"{connection.source}\n" + " · ".join(settings)


def _facts(network: Network) -> list[tuple[str, str | Text]]:
    schema, diagnostics, config = network.schema, network.diagnostics, network.predict_config
    status = Text(
        "Converged" if diagnostics.converged else "Not converged",
        style="green" if diagnostics.converged else "yellow",
    )
    status.append(f" · {diagnostics.n_iter} iterations", style="default")
    cardinalities = ", ".join(str(count) for count in schema.M_cat[:6])
    if len(schema.M_cat) > 6:
        cardinalities += ", …"
    data = f"{schema.D_cont} continuous · {len(schema.M_cat)} categorical"
    if schema.M_cat:
        data += f" ({cardinalities})"
    output_label = "classes" if schema.task == "classification" else "outputs"
    data += f" · {schema.M} {output_label}"
    prediction = [config.predict_mode]
    if config.output_mode is not None:
        prediction.append(config.output_mode)
    if config.epsilon_P is not None:
        prediction.append(f"ε_P={_number(config.epsilon_P)}")
    if network._fitted.epsilon_P_source == "derived":
        calibration = "in-sample"
    elif network._fitted.epsilon_P_source == "supplied":
        calibration = "not performed (supplied ε_P)"
    else:
        calibration = "not applicable"
    retention = (
        "members not retained"
        if network.members is None
        else f"{_count(len(network.members), 'member')} retained"
    )
    n_candidates = len(diagnostics.initialisation_outcomes)
    multi_init = f"{_count(n_candidates, 'candidate')} · {retention}"
    if network.initial_states is not None:
        multi_init += f" · {_count(len(network.initial_states), 'initial state')} retained"
    history = diagnostics.loss_history
    loss = f"{_number(history[0])} → {_number(history[-1])}" if history else "unavailable"
    facts: list[tuple[str, str | Text]] = [
        ("Fit", status),
        ("Loss", loss),
        ("Data", data),
        ("Compute", f"{str(schema.computation_dtype).removeprefix('torch.')} · {network.device}"),
        ("Prediction", " · ".join(prediction)),
        ("Calibration", calibration),
        ("Multi-init", multi_init),
    ]
    if diagnostics.warnings:
        # Warning collections can describe many initialisations. Preview one, not all.
        preview = diagnostics.warnings[0][:160]
        if len(diagnostics.warnings[0]) > 160:
            preview += "…"
        facts.append(
            ("Warnings", Text(f"{len(diagnostics.warnings)} recorded · {preview}", style="yellow"))
        )
    return facts


def render_network(network: Network, *, width: int) -> Panel:
    """Build the table from the original Recipe and the fitted active schema."""
    width = min(width, _MAX_DISPLAY_WIDTH)
    wide = width >= _WIDTH_CRITERIA
    blocks = Table(
        box=box.SIMPLE,
        expand=True,
        padding=(0, 1),
        show_edge=False,
        leading=1,
    )
    blocks.add_column("Block", ratio=1, overflow="fold")
    if wide:
        blocks.add_column("Size", style="cyan", no_wrap=True)
        blocks.add_column("Hyperparameters", ratio=1, overflow="fold")
        blocks.add_column("Incoming", ratio=1, overflow="fold")
    else:
        blocks.add_column("Configuration", ratio=2, overflow="fold")
    descriptions = {block.name: block for block in network.recipe.blocks}
    incoming: dict[str, list[str]] = {}
    for connection in network.recipe.connections:
        incoming.setdefault(connection.target, []).append(_incoming(connection))
    for name, active in network.schema.K_active:
        block = descriptions[name]
        if isinstance(block, ClassificationHead):
            size = f"{active} classes"
        elif isinstance(block, RegressionHead):
            size = f"{active} outputs"
        else:
            size = f"{block.K} → {active}"
        parameters = _parameters(block, categorical=bool(network.schema.M_cat))
        connection_text = "\n".join(incoming.get(name, [])) or "—"
        label = Text(f"{name}\n{type(block).__name__}", overflow="fold")
        if wide:
            blocks.add_row(label, Text(size), Text(parameters), Text(connection_text))
        else:
            configuration = Text(size + "\n", style="cyan")
            configuration.append(parameters, style="default")
            configuration.append("\nIncoming: " + connection_text, style="default")
            blocks.add_row(label, configuration)
    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="bold", no_wrap=True)
    facts.add_column(overflow="fold")
    for label, value in _facts(network):
        facts.add_row(label, value if isinstance(value, Text) else Text(value))
    return Panel(
        Group(
            blocks,
            Text("Size: requested → active K, or class/output count.", style="dim"),
            Text("Incoming: source and connection settings. ∞ = frozen weights.", style="dim"),
            Text(""),
            facts,
        ),
        title=Text(f"Network · {network.schema.task}"),
        title_align="left",
        border_style="dim",
        width=width,
    )
