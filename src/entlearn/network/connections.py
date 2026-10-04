"""Private target-owned connection state and coordinate operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from entlearn.primitives.coupling import accumulate_coupling_, transition_partial_loss_
from entlearn.primitives.normalise import floored_log_, normalise_
from entlearn.primitives.softmax import assign_simplex_
from entlearn.primitives.transitions import update_theta_
from entlearn.recipe import Connection, Coupling

if TYPE_CHECKING:
    from entlearn.network.session import _FitSession, _Workspace


@dataclass
class _ConnectionState:
    """Transition tensors owned by the target block."""

    description: Connection
    theta: torch.Tensor
    log_theta: torch.Tensor
    initial_widths: tuple[int, int]

    @classmethod
    def allocate(
        cls,
        description: Connection,
        K_target: int,
        K_source: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _ConnectionState:
        """Allocate an uninitialised transition and its safe-log cache."""
        theta = torch.empty(K_target, K_source, dtype=dtype, device=device)
        return cls(
            description=description,
            theta=theta,
            log_theta=torch.empty_like(theta),
            initial_widths=(K_target, K_source),
        )

    @property
    def coupling(self) -> Coupling:
        """Return the coupling the Recipe stores for this connection."""
        assert self.description.coupling is not None
        return self.description.coupling

    def prune_source_(self, keep: torch.Tensor, scratch: torch.Tensor) -> None:
        """Compact the source columns, renormalising only under S coupling."""
        index = keep.nonzero(as_tuple=True)[0]
        self.theta = torch.index_select(self.theta, 1, index)
        self.log_theta = torch.index_select(self.log_theta, 1, index)
        if self.coupling is Coupling.S:
            normalise_(self.theta, 1, scratch[: self.theta.shape[0]])
            floored_log_(self.log_theta, self.theta)

    def prune_target_(self, keep: torch.Tensor, scratch: torch.Tensor) -> None:
        """Compact the target rows, renormalising only under M coupling."""
        index = keep.nonzero(as_tuple=True)[0]
        self.theta = torch.index_select(self.theta, 0, index)
        self.log_theta = torch.index_select(self.log_theta, 0, index)
        if self.coupling is Coupling.M:
            normalise_(self.theta, 0, scratch[: self.theta.shape[1]])
            floored_log_(self.log_theta, self.theta)

    @property
    def _norm_dim(self) -> int:
        """Return the stochastic axis selected by the connection coupling."""
        return 0 if self.coupling is Coupling.M else 1

    @property
    def pseudocount(self) -> float:
        """Return the per-entry Dirichlet prior at unit-total count scale.

        Counts always use normalised sample weights, including uniform weights for
        an unweighted fit. Dividing native counts and their prior by the same total
        preserves the transition and the objective.
        """
        assert self.description.theta_alpha is not None
        K_target, K_source = self.initial_widths
        return (self.description.theta_alpha - 1.0) / (K_target * K_source)

    def update_parameters_(
        self,
        gamma_target: torch.Tensor,
        gamma_source: torch.Tensor,
        session: _FitSession,
    ) -> None:
        """Update this transition from the source and target affiliations."""
        T, K_source = gamma_source.shape
        norm_dim = self._norm_dim
        scratch = (
            session.workspace.scratch_K[:K_source]
            if norm_dim == 0
            else session.workspace.scratch_K[: gamma_target.shape[1]]
        )
        update_theta_(
            self.theta,
            self.log_theta,
            gamma_target,
            gamma_source,
            scratch,
            sample_weights=session.data.sample_weights,
            scratch_TK_source=session.workspace.matrix[:T, :K_source],
            pseudocount=self.pseudocount,
            norm_dim=norm_dim,
        )

    def accumulate_into_target_cost_(
        self,
        cost: torch.Tensor,
        gamma_source: torch.Tensor,
        session: _FitSession,
    ) -> None:
        """Add this connection's target-side coupling cost."""
        K_source = gamma_source.shape[1]
        accumulate_coupling_(
            cost,
            self.log_theta.transpose(0, 1),
            gamma_source,
            self.description.delta,
            session.data.sample_weights,
            session.workspace.log_matrix[:, :K_source],
        )

    def accumulate_into_source_cost_(
        self,
        cost: torch.Tensor,
        gamma_target: torch.Tensor,
        session: _FitSession,
    ) -> None:
        """Add this connection's source-side coupling cost."""
        K_target = gamma_target.shape[1]
        accumulate_coupling_(
            cost,
            self.log_theta,
            gamma_target,
            self.description.delta,
            session.data.sample_weights,
            session.workspace.log_matrix[:, :K_target],
        )

    def partial_loss_(
        self,
        gamma_target: torch.Tensor,
        gamma_source: torch.Tensor,
        session: _FitSession,
    ) -> None:
        """Add this transition's loss to the session loss.

        Both the counts and prior use unit-total scale. Thus the multiplier is delta,
        not delta / total: -(delta / total) Σ(counts + prior) log(theta) equals
        -delta Σ(counts / total + prior / total) log(theta).
        """
        T, K_source = gamma_source.shape
        transition_partial_loss_(
            session.workspace.loss,
            self.log_theta,
            gamma_target,
            gamma_source,
            self.description.delta,
            session.workspace.matrix[:T, :K_source],
            session.workspace.scalar,
            session.data.sample_weights,
        )
        pseudocount = self.pseudocount
        if pseudocount != 0.0:
            torch.sum(self.log_theta, dim=(0, 1), out=session.workspace.scalar)
            session.workspace.loss.add_(
                session.workspace.scalar,
                alpha=-self.description.delta * pseudocount,
            )

    def propagate_(
        self,
        gamma_target: torch.Tensor,
        gamma_source: torch.Tensor,
        epsilon: float,
        workspace: _Workspace,
    ) -> None:
        """Propagate affiliations with the target's resolved hard or soft temperature."""
        gamma_target.zero_()
        accumulate_coupling_(
            gamma_target,
            self.log_theta.transpose(0, 1),
            gamma_source,
            self.description.delta,
        )
        assign_simplex_(
            gamma_target,
            gamma_target,
            epsilon,
            1,
            workspace.row_keepdim,
            workspace.row_indices,
        )
