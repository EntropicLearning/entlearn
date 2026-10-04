"""Private manifold parameters and ordered coordinate calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from entlearn.network.blocks.input_common import (
    copy_input_rows,
    discretisation_loss_,
    instance_weight_loss_,
    prune_,
    update_affiliations_,
    update_instance_weights_,
)
from entlearn.network.blocks.manifold_geometry import _ManifoldDistance, _SubspaceWorkspace
from entlearn.network.data import _StagedData
from entlearn.network.state import InitialState, _input_width
from entlearn.network.transfer import copy_tensor
from entlearn.network.validation import (
    _FLOAT_DTYPES,
    _validate_prediction_features,
    _validate_storage,
)
from entlearn.primitives.centroids import compute_gamma_wt_, update_euclidean_centroids_
from entlearn.primitives.normalise import _eps, _is_soft
from entlearn.recipe import ManifoldInput

if TYPE_CHECKING:
    from entlearn.network.session import _FitSession, _Workspace

# Each entry of PᵀP sums D products, so allow a D-scaled round-off guard.
_ORTHONORMALITY_GUARD = 32


@dataclass
class _ManifoldCache:
    """Row-bound distances, weighted statistics, and reusable coordinate scratch.

    Pruning compacts ``sqdist`` and ``disc_cost`` to active K. Other cluster-indexed
    scratch retains its initial capacity K_max and is sliced to the active cluster count.
    """

    sqdist: torch.Tensor  # (T, K) unweighted metrised discretisation error
    disc_cost: torch.Tensor  # (T, K) Wt-weighted error; adds coupling before Γ updates
    weighted_gamma: torch.Tensor  # (T, K_max) Γ ∘ Wt for centroid and projector updates
    centroid_numerator: torch.Tensor  # (K_max, D) (Γ∘Wt)ᵀX, the centroid numerator
    instance_weight_cost: torch.Tensor  # (T,) per-instance error, the Wt assignment cost
    scratch_one: torch.Tensor  # (1,) Wt assignment reduction scratch
    scratch_index: torch.Tensor  # (1,) int64 argmin scratch for hard Wt assignment
    distance: _ManifoldDistance  # Reusable buffers for metrised distance assembly
    subspace: _SubspaceWorkspace  # One cluster's covariance/SVD scratch for projectors
    categorical_cost: None = None  # Absent categorical term in the shared Wt calculation

    @classmethod
    def allocate(
        cls, X: torch.Tensor, K: int, d: int, *, capacity: int | None = None
    ) -> _ManifoldCache:
        """Allocate the buffers once for this fit's rows and initial cluster count."""
        T, D = X.shape
        K_max = K if capacity is None else capacity
        dtype, device = X.dtype, X.device
        return cls(
            sqdist=torch.empty(T, K, dtype=dtype, device=device),
            disc_cost=torch.empty(T, K, dtype=dtype, device=device),
            weighted_gamma=torch.empty(T, K_max, dtype=dtype, device=device),
            centroid_numerator=torch.empty(K_max, D, dtype=dtype, device=device),
            instance_weight_cost=torch.empty(T, dtype=dtype, device=device),
            scratch_one=torch.empty(1, dtype=dtype, device=device),
            scratch_index=torch.empty(1, dtype=torch.int64, device=X.device),
            distance=_ManifoldDistance.allocate(X, K_max, d),
            subspace=_SubspaceWorkspace.allocate(X, d),
        )


def stage_projectors(
    value: object,
    centroids: torch.Tensor,
    d: int,
    *,
    allow_conversion: bool,
) -> torch.Tensor:
    """Validate and copy the complete local-subspace projector tensor.

    Each stored (D, d) projector P is an orthonormal basis. PPᵀ is the corresponding
    (D, D) orthogonal projection operator.

    The orthonormality check tolerates float32 rounding in every dtype, as the retained
    simplex checks do. A basis computed in float32 keeps its float32 error after
    promotion to float64. No re-orthogonalisation is performed, so valid transferred
    values remain unchanged.

    Raises:
        ValueError: If shape, storage, placement, finiteness or orthonormality is invalid.
    """
    if not isinstance(value, torch.Tensor):
        raise ValueError("InitialState requires manifold_projectors")
    _validate_storage(value, name="manifold_projectors")
    if value.shape != (*centroids.shape, d):
        raise ValueError("manifold_projectors have the wrong shape")
    if value.dtype not in _FLOAT_DTYPES or value.device != centroids.device:
        raise ValueError("manifold_projectors must be floating tensors on the input device")
    if value.dtype != centroids.dtype and not allow_conversion:
        raise ValueError("manifold_projectors must use the computation dtype")
    projectors = value.to(dtype=centroids.dtype).detach().clone()
    if not bool(torch.isfinite(projectors).all()):
        raise ValueError("manifold_projectors must be finite")
    gram = projectors.transpose(1, 2) @ projectors
    identity = torch.eye(d, dtype=projectors.dtype, device=projectors.device).expand_as(gram)
    tolerance = _ORTHONORMALITY_GUARD * _eps(torch.float32) * centroids.shape[1]
    if not torch.allclose(gram, identity, rtol=0.0, atol=tolerance):
        raise ValueError("manifold_projectors must be orthonormal")
    return projectors


@dataclass
class _ManifoldInputBlock:
    """Fitted centroids and local-subspace projectors, affiliation regime, and instance weights.

    ``cache`` operates as the standard input block's does: ``stage_cache_`` allocates it
    when a fit session opens and ``release_cache`` drops it at publication.
    """

    description: ManifoldInput
    continuous_centroids: torch.Tensor
    manifold_projectors: torch.Tensor
    gamma: torch.Tensor
    instance_weights: torch.Tensor
    log_partition: torch.Tensor
    soft_assignments: bool
    cache: _ManifoldCache | None = field(default=None, repr=False)

    @property
    def K(self) -> int:
        """Return the active cluster count."""
        return self.continuous_centroids.shape[0]

    @property
    def effective_epsilon(self) -> float:
        """Return the affiliation temperature of the fitted regime: zero when hard."""
        return self.description.epsilon if self.soft_assignments else 0.0

    @property
    def live_cache(self) -> _ManifoldCache:
        """Return operation-local caches, which are absent after publication."""
        if self.cache is None:
            raise RuntimeError("the manifold block holds no cache")
        return self.cache

    @classmethod
    def make_block(
        cls, description: ManifoldInput, state: InitialState, data: _StagedData
    ) -> _ManifoldInputBlock:
        """Materialise owned manifold parameters from a resolved InitialState.

        ``state`` must hold manifold geometry that ``resolve_state`` has matched to
        ``description``, in the dtype and on the device of ``data``, with its feature
        dimension.
        """
        geometry = state.input_geometry
        assert geometry is not None and geometry.manifold_projectors is not None
        K = _input_width(geometry, description)
        # Keep only this fit's first K centroids and their corresponding stored projectors.
        centroids = geometry.continuous_centroids[:K].detach().clone()
        projectors = geometry.manifold_projectors[:K].detach().clone()
        T = data.X_cont.shape[0]
        return cls(
            description=description,
            continuous_centroids=centroids,
            manifold_projectors=projectors,
            gamma=torch.empty(T, K, dtype=centroids.dtype, device=centroids.device),
            instance_weights=data.sample_weights.detach().clone(),
            log_partition=torch.full((), torch.inf, dtype=centroids.dtype, device=centroids.device),
            soft_assignments=_is_soft(description.epsilon, centroids.dtype),
        )

    def stage_cache_(self, data: _StagedData, *, capacity: int | None = None) -> None:
        """Allocate this fit's cache, its cluster scratch sized for ``capacity`` clusters.

        ``capacity=None`` sizes the scratch for the active clusters.
        """
        self.cache = _ManifoldCache.allocate(
            data.X_cont, self.K, self.description.subspace_dimension, capacity=capacity
        )

    def release_cache(self) -> None:
        """Drop the cache, keeping the fitted parameters and row-bound state."""
        self.cache = None

    def copy_active(
        self, dtype: torch.dtype, device: torch.device, row_weights: torch.Tensor | None
    ) -> _ManifoldInputBlock:
        """Copy the active parameters, keeping the fitted rows or starting new ones.

        See ``copy_input_rows`` for the rows. The copy holds no cache.
        """
        gamma, instance_weights, log_partition = copy_input_rows(self, dtype, device, row_weights)
        return replace(
            self,
            continuous_centroids=copy_tensor(self.continuous_centroids, dtype, device),
            manifold_projectors=copy_tensor(self.manifold_projectors, dtype, device),
            gamma=gamma,
            instance_weights=instance_weights,
            log_partition=log_partition,
            cache=None,
        )

    def validate_prediction_input(
        self, X_cont_new: object, X_cat_new: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Check query data against the fitted continuous-only geometry."""
        return _validate_prediction_features(X_cont_new, X_cat_new, self.continuous_centroids, ())

    def prepare_cache_(self, session: _FitSession) -> None:
        """Fill the staged cache from the current parameters."""
        refresh_cache_(self, session)

    def price_query_rows_(
        self,
        X: torch.Tensor,
        _X_cat: tuple[torch.Tensor, ...],
        _workspace: _Workspace,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Price validated query rows against the fitted geometry, without instance weights.

        Returns a new ``(T, K)`` metrised distance as both the total cost and its
        continuous part, with no categorical part, in the standard input's shape.
        """
        cost = torch.zeros(X.shape[0], self.K, dtype=X.dtype, device=X.device)
        distance = _ManifoldDistance.allocate(X, self.K, self.description.subspace_dimension)
        distance.assemble_(
            cost, X, self.continuous_centroids, self.manifold_projectors, self.description.alpha
        )
        return cost, cost, None

    def update_parameters_(self, session: _FitSession) -> None:
        """Update affiliations, prune, then update weights, centroids and projectors."""
        session.graph.accumulate_outgoing_cost_(self, self.live_cache.disc_cost, session)
        update_affiliations_(self, session)
        prune_(self, session)
        update_instance_weights_(self, session)
        update_centroids_(self, session)
        update_projectors_(self, session)
        refresh_cache_(self, session)

    def prune(self, keep: torch.Tensor) -> None:
        """Prune fitted clusters, affiliations and dependent distance caches together."""
        index = keep.nonzero(as_tuple=True)[0]
        self.gamma = self.gamma.index_select(1, index)
        self.continuous_centroids = self.continuous_centroids.index_select(0, index)
        self.manifold_projectors = self.manifold_projectors.index_select(0, index)
        cache = self.live_cache
        cache.sqdist = cache.sqdist.index_select(1, index)
        cache.disc_cost = cache.disc_cost.index_select(1, index)

    def partial_loss_(self, session: _FitSession) -> None:
        """Add metrised discretisation error and affiliation and instance-weight entropies to the session loss."""
        discretisation_loss_(self, session)
        instance_weight_loss_(self, session)


def refresh_cache_(block: _ManifoldInputBlock, session: _FitSession) -> None:
    """Refresh the distances and their instance-weighted contribution."""
    cache = block.live_cache
    cache.distance.assemble_(
        cache.sqdist,
        session.data.X_cont,
        block.continuous_centroids,
        block.manifold_projectors,
        block.description.alpha,
    )
    torch.mul(cache.sqdist, block.instance_weights[:, None], out=cache.disc_cost)


def update_centroids_(block: _ManifoldInputBlock, session: _FitSession) -> None:
    """Stage weighted affiliations and update the weighted centroid means."""
    cache = block.live_cache
    K = block.K
    compute_gamma_wt_(
        cache.weighted_gamma[:, :K],
        session.workspace.scratch_K[:K],
        block.gamma,
        block.instance_weights,
    )
    torch.mm(cache.weighted_gamma[:, :K].t(), session.data.X_cont, out=cache.centroid_numerator[:K])
    update_euclidean_centroids_(
        block.continuous_centroids,
        cache.centroid_numerator[:K],
        session.workspace.scratch_K[:K],
        session.workspace.empty_clusters[:K],
    )


def update_projectors_(block: _ManifoldInputBlock, session: _FitSession) -> None:
    """Update local subspaces about the fresh centroids, skipping the Wt-weighted empty ones.

    Reads the empty mask and the weighted affiliations that ``update_centroids_`` leaves.
    """
    cache = block.live_cache
    empty = session.workspace.empty_clusters[: block.K].tolist()
    for index in range(block.K):
        if not empty[index]:
            cache.subspace.write_(
                session.data.X_cont,
                block.continuous_centroids[index],
                cache.weighted_gamma[:, index],
                block.manifold_projectors[index],
            )
