"""Private hidden-block affiliations and tensor calculations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from entlearn.network.blocks.affiliations import (
    affiliation_entropy_,
    assign_affiliations_,
)
from entlearn.network.connections import _ConnectionState
from entlearn.network.state import _HiddenParameters
from entlearn.network.transfer import copy_log, copy_tensor
from entlearn.primitives.normalise import _is_soft
from entlearn.recipe import Hidden

if TYPE_CHECKING:
    from entlearn.network.data import _StagedData
    from entlearn.network.session import _FitSession
    from entlearn.recipe import Connection


@dataclass
class _HiddenBlock:
    """Fitted affiliations, assignment regime and incoming connections."""

    description: Hidden
    gamma: torch.Tensor
    incoming: dict[str, _ConnectionState]
    soft_assignments: bool

    @classmethod
    def make_block(
        cls,
        description: Hidden,
        data: _StagedData,
        connection: Connection,
        K_source: int,
    ) -> _HiddenBlock:
        """Allocate one hidden block and its incoming transition."""
        state = _ConnectionState.allocate(
            connection,
            description.K,
            K_source,
            dtype=data.X_cont.dtype,
            device=data.X_cont.device,
        )
        return cls(
            description=description,
            soft_assignments=_is_soft(description.epsilon, data.X_cont.dtype),
            gamma=torch.empty(
                data.X_cont.shape[0],
                description.K,
                dtype=data.X_cont.dtype,
                device=data.X_cont.device,
            ),
            incoming={connection.name: state},
        )

    @property
    def K(self) -> int:
        """Return the active cluster count."""
        return self.gamma.shape[1]

    @property
    def effective_epsilon(self) -> float:
        """Return the affiliation temperature of the fitted regime: zero when hard."""
        return self.description.epsilon if self.soft_assignments else 0.0

    def prune(self, keep: torch.Tensor) -> None:
        """Compact the affiliation columns to the surviving clusters."""
        index = keep.nonzero(as_tuple=True)[0]
        self.gamma = torch.index_select(self.gamma, 1, index)

    def initialise_(self, session: _FitSession) -> None:
        """Propagate source affiliations through this block's existing incoming transition."""
        connection = session.graph.connection_into(self)
        gamma_source = session.graph.affiliations(connection.source)
        self.incoming[connection.name].propagate_(
            self.gamma,
            gamma_source,
            self.effective_epsilon,
            session.workspace,
        )

    def copy_active(
        self, dtype: torch.dtype, device: torch.device, row_weights: torch.Tensor | None
    ) -> _HiddenBlock:
        """Copy the incoming transitions, keeping the fitted affiliations or allocating new ones.

        ``row_weights=None`` keeps the fitted rows; otherwise the affiliations are
        allocated, unfilled, for the ``len(row_weights)`` new rows.
        """
        incoming = {}
        for name, state in self.incoming.items():
            theta = copy_tensor(state.theta, dtype, device)
            incoming[name] = replace(state, theta=theta, log_theta=copy_log(state.log_theta, theta))
        width = next(iter(incoming.values())).theta.shape[0]
        gamma = (
            copy_tensor(self.gamma, dtype, device)
            if row_weights is None
            else torch.empty(len(row_weights), width, dtype=dtype, device=device)
        )
        return replace(self, gamma=gamma, incoming=incoming)

    def capture_parameters(self) -> _HiddenParameters:
        """Copy this block's complete active transition, without row affiliations."""
        (connection,) = self.incoming.values()
        return _HiddenParameters(
            replace(self.description, K=connection.theta.shape[0]),
            connection.coupling,
            connection.theta.detach().clone(),
        )

    def update_affiliations_(self, session: _FitSession) -> None:
        """Update affiliations from every incoming and outgoing coupling."""
        graph = session.graph
        workspace = session.workspace
        T = session.data.X_cont.shape[0]
        K = self.K
        cost = workspace.matrix[:T, :K]
        cost.zero_()
        for connection in graph.incoming[self.description.name]:
            self.incoming[connection.name].accumulate_into_target_cost_(
                cost,
                graph.affiliations(connection.source),
                session,
            )
        graph.accumulate_outgoing_cost_(self, cost, session)
        assign_affiliations_(self.gamma, cost, self, session)

    def update_incoming_connections_(self, session: _FitSession) -> None:
        """Update incoming transitions."""
        graph = session.graph
        for connection in graph.incoming[self.description.name]:
            self.incoming[connection.name].update_parameters_(
                self.gamma,
                graph.affiliations(connection.source),
                session,
            )

    def prune_(self, session: _FitSession) -> None:
        """Ask the graph to compact every owner affected by this block's emptiness."""
        session.graph.prune_(self, session.workspace)

    def update_parameters_(self, session: _FitSession) -> None:
        """Run the complete hidden coordinate step in dependency order."""
        self.update_affiliations_(session)
        self.prune_(session)
        self.update_incoming_connections_(session)

    def partial_loss_(self, session: _FitSession) -> None:
        """Add this hidden block's affiliation entropy."""
        affiliation_entropy_(self.gamma, self, session)
