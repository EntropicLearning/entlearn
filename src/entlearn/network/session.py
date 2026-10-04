"""Private fitted graph and operation-local fit state.

Blocks own their rebuildable, row-bound caches. The fit session owns only data shared by
the graph and scratch reused across block calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.blocks.types import (
    _ClusteringBlock,
    _InputBlock,
    _OutputBlock,
    _RuntimeBlock,
)
from entlearn.network.data import _StagedData
from entlearn.primitives.geometry import mark_non_empty_clusters_
from entlearn.primitives.normalise import _is_soft
from entlearn.recipe import Connection, Recipe


@dataclass
class _Workspace:
    """Scratch shared by more than one block calculation.

    T is the row count of this fit or query operation. Row-indexed buffers
    have exactly T rows, only ``matrix`` may have more, to support output coordinates
    when M > T. Cluster/output dimensions retain their allocated capacity after
    pruning, so callers take views limited to the active clusters. A new query batch allocates
    a workspace for its own T rather than reusing the training workspace.
    """

    matrix: torch.Tensor  # (max(T, M), max(K, M)); the widest general-purpose scratch
    log_matrix: torch.Tensor  # (T, max(K, M)); the log buffer of an entropy term
    scratch_K: torch.Tensor  # (max(K, M),) cluster-indexed scratch, holds the masses
    scratch_T: torch.Tensor  # (T,) row-indexed scratch
    row_keepdim: torch.Tensor  # (T, 1) the softmax slice maximum and normaliser
    row_indices: torch.Tensor  # (T, 1) int64, the argmin index
    non_empty: torch.Tensor  # (K,) bool, the prune's survivors by unweighted mass
    empty_clusters: torch.Tensor  # (K,) bool, a centroid update's empty clusters by Wt mass
    loss: torch.Tensor  # () the accumulated loss
    scalar: torch.Tensor  # () scratch for a full reduction

    @classmethod
    def allocate(
        cls,
        rows: int,
        clusters: int,
        outputs: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _Workspace:
        """Allocate shared scratch for one operation."""
        columns = max(clusters, outputs)
        matrix_rows = max(rows, outputs)
        return cls(
            matrix=torch.empty(matrix_rows, columns, dtype=dtype, device=device),
            log_matrix=torch.empty(rows, columns, dtype=dtype, device=device),
            scratch_K=torch.empty(columns, dtype=dtype, device=device),
            scratch_T=torch.empty(rows, dtype=dtype, device=device),
            row_keepdim=torch.empty(rows, 1, dtype=dtype, device=device),
            row_indices=torch.empty(rows, 1, dtype=torch.int64, device=device),
            non_empty=torch.empty(clusters, dtype=torch.bool, device=device),
            empty_clusters=torch.empty(clusters, dtype=torch.bool, device=device),
            loss=torch.zeros((), dtype=dtype, device=device),
            scalar=torch.empty((), dtype=dtype, device=device),
        )


@dataclass
class _CompiledGraph:
    """Private parameters and stable graph structure of a fitted Network."""

    recipe: Recipe
    order: tuple[str, ...]
    connections: tuple[Connection, ...]
    incoming: dict[str, tuple[Connection, ...]]
    outgoing: dict[str, tuple[Connection, ...]]
    blocks: dict[str, _RuntimeBlock]
    connection_sub_seeds: tuple[tuple[str, int], ...]
    training_rows: int

    @property
    def input(self) -> _InputBlock:
        """Return the input block."""
        block = self.blocks[self.order[0]]
        assert isinstance(block, _InputBlock)
        return block

    @property
    def head(self) -> _OutputBlock:
        """Return the output (head)."""
        block = self.blocks[self.order[-1]]
        assert isinstance(block, _OutputBlock)
        return block

    @property
    def terminal(self) -> Connection:
        """Return the connection into the head."""
        (connection,) = self.incoming[self.order[-1]]
        return connection

    def affiliations(self, name: str) -> torch.Tensor:
        """Return a clustering block's fitted affiliations."""
        block = self.blocks[name]
        assert isinstance(block, _ClusteringBlock)
        return block.gamma

    def connection_into(self, block: _RuntimeBlock) -> Connection:
        """Return the connection whose target is ``block``."""
        (connection,) = self.incoming[block.description.name]
        return connection

    def source_affiliations(self, block: _OutputBlock) -> torch.Tensor:
        """Return the affiliations of the clustering block feeding an output head."""
        return self.affiliations(self.connection_into(block).source)

    def resolve_affiliation_regimes(self) -> None:
        """Resolve every block's regime from the current descriptions and dtype."""
        dtype = self.input.continuous_centroids.dtype
        for block in self.blocks.values():
            if isinstance(block, _ClusteringBlock):
                block.soft_assignments = _is_soft(block.description.epsilon, dtype)

    def accumulate_outgoing_cost_(
        self,
        block: _ClusteringBlock,
        cost: torch.Tensor,
        session: _FitSession,
    ) -> None:
        """Add every downstream coupling to ``block``'s assignment-cost accumulator."""
        for connection in self.outgoing[block.description.name]:
            target = self.blocks[connection.target]
            if isinstance(target, _HiddenBlock):
                target.incoming[connection.name].accumulate_into_source_cost_(
                    cost,
                    target.gamma,
                    session,
                )
            else:
                assert isinstance(target, _OutputBlock)
                target.accumulate_into_source_cost_(session, cost=cost)

    def prune_(
        self,
        block: _ClusteringBlock,
        workspace: _Workspace,
    ) -> None:
        """Remove empty clusters from every affected owner in one graph operation.

        Emptiness, according to the Buddha, depends only on unweighted affiliation mass and the computation dtype.
        The block prunes its own tensors and caches, its incoming state loses target
        rows, and each downstream owner loses source columns. Shared workspace retains
        its capacity and subsequent coordinates take views limited to the remaining clusters.

        Args:
            block: Clustering block whose affiliations have just been updated.
            workspace: Scratch for masses, the survivor mask and renormalisation.
        """
        K = block.K
        keep = workspace.non_empty[:K]
        scratch = workspace.scratch_K
        if mark_non_empty_clusters_(keep, scratch[:K], block.gamma) == K:
            return
        block.prune(keep)
        name = block.description.name
        if isinstance(block, _HiddenBlock):
            for connection in self.incoming[name]:
                block.incoming[connection.name].prune_target_(
                    keep,
                    scratch,
                )
        for connection in self.outgoing[name]:
            downstream = self.blocks[connection.target]
            if isinstance(downstream, _HiddenBlock):
                downstream.incoming[connection.name].prune_source_(
                    keep,
                    scratch,
                )
            else:
                assert isinstance(downstream, _OutputBlock)
                downstream.prune_source(keep, scratch)

    def release_caches(self) -> None:
        """Drop every block's derived cache, keeping the fitted parameters.

        Publication calls this, since the caches are row-bound and rebuildable, so a retained
        Network can hold geometry alone.
        """
        self.input.release_cache()
        head = self.head
        if isinstance(head, _RegressionBlock):
            head.release_cache()


@dataclass
class _FitSession:
    """Validated data, the graph it fits, diagnostics, and shared scratch for one fit."""

    data: _StagedData
    graph: _CompiledGraph
    workspace: _Workspace
    uniform_sample_weights: bool
    warning_messages: list[str]

    @classmethod
    def open(
        cls,
        graph: _CompiledGraph,
        data: _StagedData,
        *,
        capacities: dict[str, int] | None = None,
    ) -> _FitSession:
        """Stage every block cache and the shared workspace, and fill the input cache.

        ``capacities`` maps a clustering block's name to the cluster count its scratch is
        sized for. A block without one takes its active count. A continued fit passes the
        original counts, so its views after a prune keep the uninterrupted fit's strides.
        Everything a coordinate step reuses is allocated here, outside any step.
        """
        capacity = {} if capacities is None else capacities
        graph.input.stage_cache_(data, capacity=capacity.get(graph.order[0]))
        head = graph.head
        if isinstance(head, _RegressionBlock):
            head.stage_cache_(data, capacity=capacity.get(graph.terminal.source))
        weights = data.sample_weights
        workspace = _Workspace.allocate(
            data.X_cont.shape[0],
            max(
                capacity.get(name, block.K)
                for name, block in graph.blocks.items()
                if isinstance(block, _ClusteringBlock)
            ),
            data.schema.M,
            dtype=data.X_cont.dtype,
            device=data.X_cont.device,
        )
        session = cls(data, graph, workspace, bool(weights.max() == weights.min()), [])
        graph.input.prepare_cache_(session)
        return session
