"""Private input-block parameters and tensor calculations."""

from __future__ import annotations

import math
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
from entlearn.network.data import _StagedData
from entlearn.network.state import InitialState, _input_width
from entlearn.network.transfer import copy_log, copy_tensor
from entlearn.network.validation import _validate_prediction_features
from entlearn.primitives.centroids import (
    _ONE_HOT_BATCHES,
    compute_gamma_wt_,
    compute_wd_cost_categorical_,
    compute_wd_cost_euclidean_,
    update_categorical_centroids_,
    update_euclidean_centroids_,
)
from entlearn.primitives.distance import (
    assemble_categorical_cost_,
    refresh_weighted_norm_,
    weighted_sq_distance_,
    weighted_sq_norm_,
)
from entlearn.primitives.normalise import _is_soft, _scratch_view, floored_log_
from entlearn.primitives.softmax import assign_simplex_
from entlearn.primitives.statistics import entropy_penalty_
from entlearn.recipe import Input, Recipe

if TYPE_CHECKING:
    from entlearn.network.session import _FitSession, _Workspace


def _categorical_channel_scale(delta_cat: float | None, cardinality: int) -> float:
    """Return a categorical channel's cardinality-adjusted scale."""
    assert delta_cat is not None  # fitting fills an omitted value when there are categories
    return delta_cat * (1.0 - 1.0 / cardinality) / math.log(cardinality)


def _guess_delta_cat(
    X: torch.Tensor, X_cat: Sequence[torch.Tensor], weights: torch.Tensor | None
) -> float | None:
    """Return the median continuous variance divided by the median scaled categorical entropy.

    ``None`` when no continuous feature varies or no categorical feature has at least two levels.
    """
    w = torch.ones(X.shape[0], dtype=X.dtype, device=X.device) if weights is None else weights
    w = w / w.max()  # keeps the sums below finite
    total = w.sum()
    mean = w @ X / total
    variances = (w @ (X - mean).square_() / total)[X.amax(0) > X.amin(0)]
    w_cpu = w.cpu()  # code masses are summed on the CPU, where index_add_ repeats
    costs = []
    for feature in X_cat:
        if feature.ndim == 1:
            masses = torch.zeros(int(feature.max()) + 1, dtype=X.dtype)
            masses = masses.index_add_(0, feature.cpu(), w_cpu).to(X.device)
        else:  # rows of level distributions
            masses = w @ feature
        frequencies = masses[masses > 0] / total  # the levels present
        if frequencies.numel() > 1:
            entropy = -float((frequencies * frequencies.log()).sum())
            costs.append(_categorical_channel_scale(1.0, frequencies.numel()) * entropy)
    if variances.numel() == 0 or not costs:
        return None
    # quantile, unlike median, averages the two middle values of an even count
    cost = torch.tensor(costs, dtype=X.dtype, device=X.device).quantile(0.5)
    return float(variances.quantile(0.5) / cost)


def _resolve_delta_cat(recipe: Recipe, data: _StagedData) -> Recipe:
    """Fill an omitted ``delta_cat`` from categorical training rows, or with ``1`` when undefined."""
    block = next((b for b in recipe.blocks if isinstance(b, Input) and b.delta_cat is None), None)
    if block is None or not data.X_cat:
        return recipe
    guess = _guess_delta_cat(data.X_cont, data.X_cat, data.raw_sample_weights)
    return recipe.replace_block(block.name, delta_cat=1.0 if guess is None else guess)


@dataclass
class _InputCache:
    """Data-dependent caches and local scratch for one input-block fit.

    Pruning compacts ``disc_cost`` to active K, because it is the one cache that is read after pruning and before being refreshed. Other cluster-indexed scratch keeps the
    size K_max staged for the fit, and steps read contiguous views of the active cluster
    count.
    """

    X_sq: torch.Tensor  # (T, D_cont) elementwise X², the weighted norm's input
    disc_cost: torch.Tensor  # (T, K) discretisation error per row and cluster, not multiplied by Wt
    X_sq_weighted: torch.Tensor | None  # (T,) Σ_d Wd[d]·X[t,d]²
    weighted_gamma: torch.Tensor | None  # (T, K_max) Γ ∘ Wt; None when Wt is frozen and uniform
    masses: torch.Tensor  # (K_max,) Σ_t Γ[t,k]·Wt[t], staged for the centroid step
    centroid_numerator: torch.Tensor | None  # (K_max, D_cont) (Γ∘Wt)ᵀ @ X
    categorical_numerators: tuple[torch.Tensor, ...]  # per feature (M_i, K_max) X̃_iᵀ @ (Γ∘Wt)
    feature_weight_cost: torch.Tensor  # (D,) per-feature discretisation error, the Wd cost
    instance_weight_cost: torch.Tensor  # (T,) Wt cost: row error with epsilon_T·log(T·w) subtracted
    scratch_KD: torch.Tensor | None  # (K_max, D_cont)
    scratch_D: torch.Tensor  # (D,)
    scratch_KM: torch.Tensor | None  # (K_max, max_i M_i)
    code_scratch: tuple[torch.Tensor, torch.Tensor] | None  # off the CPU, with code features
    scratch_MK: torch.Tensor  # (max_i M_i, K_max), negated scaled log centroids, transposed
    scratch_one: torch.Tensor  # (1,)
    scratch_index: torch.Tensor  # (1,) int64
    categorical_scales: torch.Tensor  # (D_cat,) delta_cat·s_i
    weighted_categorical_scales: torch.Tensor  # (D_cat,) delta_cat·s_i·Wd[D_cont+i]
    uniform_instance_weight: float | None  # the frozen Wt when every row shares it, else None

    @classmethod
    def allocate(cls, block: _StandardInputBlock, data: _StagedData) -> _InputCache:
        """Allocate the caches from the fitted dimensions and the operation data."""
        T, D_cont = data.X_cont.shape
        K = block.K
        D_cat = len(data.X_cat)
        D = D_cont + D_cat
        dtype = data.X_cont.dtype
        device = data.X_cont.device
        cardinalities = data.schema.M_cat
        Wt = block.instance_weights
        uniform = not math.isfinite(block.description.epsilon_T) and bool(Wt.max() == Wt.min())
        scales = torch.tensor(
            [_categorical_channel_scale(block.description.delta_cat, M_i) for M_i in cardinalities],
            dtype=dtype,
            device=device,
        )
        return cls(
            X_sq=data.X_cont * data.X_cont,
            disc_cost=torch.empty(T, K, dtype=dtype, device=device),
            X_sq_weighted=(torch.empty(T, dtype=dtype, device=device) if D_cont else None),
            weighted_gamma=None if uniform else torch.empty(T, K, dtype=dtype, device=device),
            masses=torch.empty(K, dtype=dtype, device=device),
            centroid_numerator=(
                torch.empty(K, D_cont, dtype=dtype, device=device) if D_cont else None
            ),
            categorical_numerators=tuple(
                torch.empty(M_i, K, dtype=dtype, device=device) for M_i in cardinalities
            ),
            feature_weight_cost=torch.empty(D, dtype=dtype, device=device),
            instance_weight_cost=torch.empty(T, dtype=dtype, device=device),
            scratch_KD=(torch.empty(K, D_cont, dtype=dtype, device=device) if D_cont else None),
            scratch_D=torch.empty(D, dtype=dtype, device=device),
            scratch_KM=(
                torch.empty(K, max(cardinalities), dtype=dtype, device=device) if D_cat else None
            ),
            # CUDA's atomic index_add_ varies from run to run
            code_scratch=(
                (
                    torch.empty(T, max(cardinalities), dtype=dtype, device=device),
                    torch.empty(
                        _ONE_HOT_BATCHES * max(cardinalities) * K, dtype=dtype, device=device
                    ),
                )
                if device.type != "cpu" and any(X_i.dim() == 1 for X_i in data.X_cat)
                else None
            ),
            scratch_MK=torch.empty(max(cardinalities, default=0), K, dtype=dtype, device=device),
            scratch_one=torch.empty(1, dtype=dtype, device=device),
            scratch_index=torch.empty(1, dtype=torch.int64, device=device),
            categorical_scales=scales,
            weighted_categorical_scales=torch.empty(D_cat, dtype=dtype, device=device),
            uniform_instance_weight=float(Wt[0]) if uniform else None,
        )


@dataclass
class _StandardInputBlock:
    """Fitted parameters, row-bound state, and derived caches of an input block.

    ``cache`` holds the values derived from those parameters while the fit runs:
    ``stage_cache_`` allocates it when a fit session opens and ``release_cache`` drops it
    at publication. Every step that changes a parameter must be followed by
    ``refresh_cache_`` before the cache is read again.
    """

    description: Input
    gamma: torch.Tensor
    feature_weights: torch.Tensor
    instance_weights: torch.Tensor
    continuous_centroids: torch.Tensor
    categorical_centroids: tuple[torch.Tensor, ...]
    log_C_cat: tuple[torch.Tensor, ...]
    log_partition: torch.Tensor
    partition_shift: torch.Tensor
    cache: _InputCache | None = field(default=None, repr=False)

    @property
    def K(self) -> int:
        """Return the active cluster count."""
        return self.continuous_centroids.shape[0]

    @property
    def effective_epsilon(self) -> float:
        """Return the affiliation temperature ``epsilon``."""
        return self.description.epsilon

    @property
    def soft_assignments(self) -> bool:
        """Whether the block uses softmax (positive ``epsilon``) or argmin (``epsilon=0``)."""
        return _is_soft(self.description.epsilon)

    @property
    def live_cache(self) -> _InputCache:
        """Return the live cache.

        Raises:
            RuntimeError: If publication has already released the cache.
        """
        if self.cache is None:
            raise RuntimeError(
                "the input block holds no cache. A published Network must be re-staged "
                "before its tensor calculations can run again"
            )
        return self.cache

    @classmethod
    def make_block(
        cls, description: Input, state: InitialState, data: _StagedData
    ) -> _StandardInputBlock:
        """Materialise fitted input state from a resolved InitialState.

        ``state`` must hold input geometry that ``resolve_state`` has matched to
        ``description``, in the dtype and on the device of ``data``, with its feature
        dimensions.
        """
        geometry = state.input_geometry
        assert geometry is not None and geometry.feature_weights is not None
        K = _input_width(geometry, description)
        # The stored shared state can be wider than this fit: take its first K centroids.
        C = geometry.continuous_centroids[:K].detach().clone()
        C_cat = tuple(value[:K].detach().clone() for value in geometry.categorical_centroids)
        Wd = geometry.feature_weights.detach().clone()
        log_C_cat = tuple(torch.empty_like(value) for value in C_cat)
        for out, value in zip(log_C_cat, C_cat, strict=True):
            floored_log_(out, value)
        return cls(
            description=description,
            gamma=torch.empty(
                data.X_cont.shape[0],
                K,
                dtype=data.X_cont.dtype,
                device=data.X_cont.device,
            ),
            feature_weights=Wd,
            instance_weights=data.sample_weights.detach().clone(),
            continuous_centroids=C,
            categorical_centroids=C_cat,
            log_C_cat=log_C_cat,
            log_partition=torch.full(
                (), torch.inf, dtype=data.X_cont.dtype, device=data.X_cont.device
            ),
            partition_shift=torch.full(
                (), torch.inf, dtype=data.X_cont.dtype, device=data.X_cont.device
            ),
        )

    def stage_cache_(self, data: _StagedData) -> None:
        """Allocate this fit's cache, its cluster scratch sized for the active clusters."""
        self.cache = _InputCache.allocate(self, data)

    def release_cache(self) -> None:
        """Drop the cache, keeping the fitted parameters and row-bound state."""
        self.cache = None

    def copy_active(
        self, dtype: torch.dtype, device: torch.device, row_weights: torch.Tensor | None
    ) -> _StandardInputBlock:
        """Copy the active parameters, keeping the fitted rows or starting new ones.

        See ``copy_input_rows`` for the rows. The copy holds no cache.
        """
        categories = tuple(
            copy_tensor(value, dtype, device) for value in self.categorical_centroids
        )
        gamma, instance_weights, log_partition, shift = copy_input_rows(
            self, dtype, device, row_weights
        )
        return replace(
            self,
            continuous_centroids=copy_tensor(self.continuous_centroids, dtype, device),
            categorical_centroids=categories,
            feature_weights=copy_tensor(self.feature_weights, dtype, device),
            log_C_cat=tuple(
                copy_log(log_C_i, C_i)
                for log_C_i, C_i in zip(self.log_C_cat, categories, strict=True)
            ),
            gamma=gamma,
            instance_weights=instance_weights,
            log_partition=log_partition,
            partition_shift=shift,
            cache=None,
        )

    def refresh_categorical_weight_scales(self, D_cont: int) -> None:
        """Refresh the categorical cost scales from the current feature weights."""
        cache = self.live_cache
        if cache.categorical_scales.numel() == 0:
            return
        torch.mul(
            cache.categorical_scales,
            self.feature_weights[D_cont:],
            out=cache.weighted_categorical_scales,
        )

    def validate_prediction_input(
        self,
        X_cont_new: object,
        X_cat_new: Sequence[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return prediction inputs checked against this block's fitted geometry.

        Fit-time staging infers the number of continuous features and each categorical
        feature's number of categories. This checks new data against those fitted dimensions.

        Raises:
            ValueError: If the dtype, device, feature count or category counts disagree with the fit.
        """
        return _validate_prediction_features(
            X_cont_new,
            X_cat_new,
            self.continuous_centroids,
            tuple(value.shape[1] for value in self.categorical_centroids),
        )

    def prune(self, keep: torch.Tensor) -> None:
        """Prune the parameters and the K-dependent caches to the kept clusters."""
        cache = self.live_cache
        index = keep.nonzero(as_tuple=True)[0]
        self.gamma = torch.index_select(self.gamma, 1, index)
        self.continuous_centroids = torch.index_select(self.continuous_centroids, 0, index)
        self.categorical_centroids = tuple(
            torch.index_select(value, 0, index) for value in self.categorical_centroids
        )
        self.log_C_cat = tuple(torch.index_select(value, 0, index) for value in self.log_C_cat)
        cache.disc_cost = torch.index_select(cache.disc_cost, 1, index)

    def prepare_cache_(self, session: _FitSession) -> None:
        """Fill the staged cache from the current parameters."""
        D_cont = session.data.schema.D_cont
        cache = self.live_cache
        if D_cont:
            assert cache.X_sq_weighted is not None
            refresh_weighted_norm_(
                cache.X_sq_weighted,
                cache.X_sq,
                self.feature_weights[:D_cont],
            )
        self.refresh_categorical_weight_scales(D_cont)
        refresh_cache_(self, session)

    def update_parameters_(self, session: _FitSession) -> None:
        """Run this block's complete coordinate step, leaving its caches fresh.

        Affiliations are updated first. The instance weights precede the staged statistics,
        because the centroid step is a ``Wt``-weighted mean.
        """
        update_affiliations_(self, session)
        prune_(self, session)
        update_instance_weights_(self, session)
        stage_statistics_(self, session)
        update_feature_weights_(self, session)
        update_centroids_(self, session)
        refresh_cache_(self, session)

    def start_categorical_centroids_(self, session: _FitSession) -> None:
        """Set the categorical centroids to their exact update from the current affiliations.

        Run once by a fresh fit. A seed row's categorical centroid is one hot, and every
        other level is set to cost ``-log(tiny)``, so that the first instance-weight update
        would keep only the rows exactly matching a seed. The continuous centroids are left
        as they are.
        """
        if not session.data.X_cat:
            return
        stage_statistics_(self, session)
        update_all_categorical_centroids_(self, session)
        refresh_cache_(self, session)

    @property
    def learns_feature_weights(self) -> bool:
        """Whether the loss holds the feature-weight entropy penalty."""
        epsilon_D = self.description.epsilon_D
        return math.isfinite(epsilon_D) and _is_soft(epsilon_D)

    def partial_loss_(self, session: _FitSession) -> float:
        """Add the input block's contribution to the session loss.

        Returns the largest magnitude of its entropy penalties, ``Σ c log n``.
        """
        cache = self.live_cache
        workspace = session.workspace
        affiliations = discretisation_loss_(self, session)
        features = 0.0
        if self.learns_feature_weights:
            entropy_penalty_(
                workspace.loss,
                self.feature_weights,
                self.description.epsilon_D,
                cache.scratch_D,
                workspace.scalar,
            )
            features = self.description.epsilon_D * math.log(self.feature_weights.numel())
        return affiliations + instance_weight_loss_(self, session) + features

    def price_query_rows_(
        self,
        X: torch.Tensor,
        X_cat: tuple[torch.Tensor, ...],
        workspace: _Workspace,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Price validated query rows against the fitted geometry, without instance weights.

        Returns new ``(T, K)`` tensors: the total cost and its continuous and categorical
        parts, either being ``None`` when its modality is absent. ``workspace``'s cluster
        and matrix scratch is overwritten. A query runs after publication has released
        the fit cache, so nothing is read from it.

        The total equals the fit's cached discretisation error.
        """
        T, D_cont = X.shape
        K = self.K
        cost = torch.zeros(T, K, dtype=X.dtype, device=X.device)
        if D_cont:
            x_norm = torch.empty(T, dtype=X.dtype, device=X.device)
            scratch_KD = torch.empty(K, D_cont, dtype=X.dtype, device=X.device)
            weighted_sq_norm_(x_norm, X, self.feature_weights[:D_cont])
            weighted_sq_distance_(
                cost,
                X,
                self.continuous_centroids,
                x_norm,
                workspace.scratch_K[:K],
                scratch_KD,
                feature_weights=self.feature_weights[:D_cont],
            )
        sqdist = (cost.clone() if X_cat else cost) if D_cont else None
        categorical_cost = torch.zeros_like(cost) if X_cat else None
        for index, (X_i, log_C_i) in enumerate(zip(X_cat, self.log_C_cat, strict=True)):
            assert categorical_cost is not None
            # Scale the weight on the device, as done in fit
            scale = self.feature_weights[D_cont + index] * _categorical_channel_scale(
                self.description.delta_cat, log_C_i.shape[1]
            )
            xent = _scratch_view(workspace.matrix, K, T)
            scratch_MK = log_C_i.new_empty(log_C_i.shape[1], K) if X_i.dim() == 1 else None
            assemble_categorical_cost_(cost, xent, X_i, log_C_i, scale, scratch_MK)
            categorical_cost.add_(xent)
        return cost, sqdist, categorical_cost


def refresh_cache_(block: _StandardInputBlock, session: _FitSession) -> None:
    """Refresh the discretisation error from the block's current parameters."""
    cache = block.live_cache
    data = session.data
    workspace = session.workspace
    T, D_cont = data.X_cont.shape
    K = block.K
    cost = cache.disc_cost
    if D_cont:
        assert cache.X_sq_weighted is not None
        assert cache.scratch_KD is not None
        weighted_sq_distance_(
            cost,
            data.X_cont,
            block.continuous_centroids,
            cache.X_sq_weighted,
            workspace.scratch_K[:K],
            cache.scratch_KD[:K],
            feature_weights=block.feature_weights[:D_cont],
        )
    else:
        cost.zero_()
    for X_i, log_C_i, scale in zip(
        data.X_cat, block.log_C_cat, cache.weighted_categorical_scales, strict=True
    ):
        assemble_categorical_cost_(
            cost,
            _scratch_view(workspace.matrix, K, T),
            X_i,
            log_C_i,
            scale,
            cache.scratch_MK[: log_C_i.shape[1], :K],
        )


def stage_statistics_(block: _StandardInputBlock, session: _FitSession) -> None:
    """Stage the cluster masses, the feature costs, and the centroid numerators.

    Runs after the instance-weight update, because the centroid step is a ``Wt``-weighted
    mean. Staging first would weight it against the previous ``Wt``.
    """
    data = session.data
    cache = block.live_cache
    workspace = session.workspace
    D_cont = data.X_cont.shape[1]
    K = block.K
    masses = cache.masses[:K]
    weight = cache.uniform_instance_weight
    if weight is None:
        assert cache.weighted_gamma is not None
        gamma_wt = _scratch_view(cache.weighted_gamma, K)
        compute_gamma_wt_(gamma_wt, masses, block.gamma, block.instance_weights)
    else:  # a uniform Wt scales the statistics of Γ, so they are formed from Γ and scaled below
        gamma_wt = block.gamma
        torch.sum(gamma_wt, dim=0, out=masses)
    learn_features = math.isfinite(block.description.epsilon_D)
    if D_cont:
        assert cache.centroid_numerator is not None
        assert cache.scratch_KD is not None
        compute_wd_cost_euclidean_(
            cache.feature_weight_cost[:D_cont],
            data.X_cont,
            block.continuous_centroids,
            gamma_wt,
            masses,
            cache.X_sq,
            cache.centroid_numerator[:K],
            workspace.scratch_T,
            cache.scratch_KD[:K],
            cache.scratch_D[:D_cont],
            need_cost=learn_features,
        )
    if data.X_cat:
        assert cache.scratch_KM is not None
        compute_wd_cost_categorical_(
            cache.feature_weight_cost[D_cont:],
            data.X_cat,
            gamma_wt,
            block.log_C_cat,
            cache.categorical_scales,
            cache.categorical_numerators,
            cache.scratch_KM,
            need_cost=learn_features,
            code_scratch=cache.code_scratch,
        )
    if weight is not None:
        masses.mul_(weight)
        if D_cont:
            assert cache.centroid_numerator is not None
            cache.centroid_numerator[:K].mul_(weight)
        for numerator in cache.categorical_numerators:
            _scratch_view(numerator, K).mul_(weight)
        if learn_features:
            cache.feature_weight_cost.mul_(weight)


def update_feature_weights_(block: _StandardInputBlock, session: _FitSession) -> None:
    """Update the input feature weights."""
    if not math.isfinite(block.description.epsilon_D):
        return
    cache = block.live_cache
    D_cont = session.data.schema.D_cont
    assign_simplex_(
        block.feature_weights,
        cache.feature_weight_cost,
        block.description.epsilon_D,
        0,
        cache.scratch_one,
        cache.scratch_index,
    )
    if D_cont:
        assert cache.X_sq_weighted is not None
        refresh_weighted_norm_(
            cache.X_sq_weighted,
            cache.X_sq,
            block.feature_weights[:D_cont],
        )
    block.refresh_categorical_weight_scales(D_cont)


def update_centroids_(block: _StandardInputBlock, session: _FitSession) -> None:
    """Update all input centroids from the staged numerators."""
    cache = block.live_cache
    workspace = session.workspace
    D_cont = session.data.schema.D_cont
    K = block.K
    if D_cont:
        assert cache.centroid_numerator is not None
        update_euclidean_centroids_(
            block.continuous_centroids,
            cache.centroid_numerator[:K],
            cache.masses[:K],
            workspace.empty_clusters[:K],
        )
    if session.data.X_cat:
        update_all_categorical_centroids_(block, session)


def update_all_categorical_centroids_(block: _StandardInputBlock, session: _FitSession) -> None:
    """Update every categorical centroid from the staged numerators and cluster masses."""
    K = block.K
    update_categorical_centroids_(
        block.categorical_centroids,
        block.log_C_cat,
        [_scratch_view(numerator, K) for numerator in block.live_cache.categorical_numerators],
        block.live_cache.masses[:K],
        session.workspace.empty_clusters[:K],
    )
