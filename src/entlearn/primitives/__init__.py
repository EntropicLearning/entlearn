"""State-free tensor primitives, re-exported flat as the curated public surface."""

from entlearn.primitives.centroids import (
    compute_gamma_wt_,
    compute_wd_cost_categorical_,
    compute_wd_cost_euclidean_,
    update_categorical_centroids_,
    update_euclidean_centroids_,
)
from entlearn.primitives.coupling import (
    accumulate_coupling_,
    transition_partial_loss_,
)
from entlearn.primitives.dissimilarity import seed_dissimilarity
from entlearn.primitives.distance import (
    assemble_categorical_cost_,
    assemble_euclidean_cost_,
    refresh_weighted_norm_,
    weighted_sq_distance_,
    weighted_sq_norm_,
)
from entlearn.primitives.encoding import (
    densify_categorical,
    split_wd,
)
from entlearn.primitives.geometry import (
    grouped_means,
    mark_non_empty_clusters_,
)
from entlearn.primitives.manifold import (
    assemble_metrised_cost_,
    project,
    weighted_subspace_,
)
from entlearn.primitives.normalise import (
    floored_log_,
    normalise_,
)
from entlearn.primitives.output import (
    apply_output_gate_,
    classification_output_accumulate_into_source_cost_,
    classification_output_loss_,
    propagate_classification_output_,
    propagate_regression_output_,
    regression_output_accumulate_into_source_cost_,
    regression_output_loss_,
    update_classification_output_,
    update_output_weights_,
    update_regression_output_,
)
from entlearn.primitives.reductions import (
    compute_wt_cost_,
    reduce_input_cost_,
)
from entlearn.primitives.softmax import (
    argmin_assign_,
    assign_simplex_,
    compute_log_partition_,
    softmax_with_temp_,
    weighted_assign_simplex_,
)
from entlearn.primitives.statistics import (
    effective_dimension,
    entropy,
    entropy_penalty_,
    inlier_scores_,
)
from entlearn.primitives.transitions import (
    update_theta_,
)

__all__ = [
    "accumulate_coupling_",
    "apply_output_gate_",
    "argmin_assign_",
    "assemble_categorical_cost_",
    "assemble_euclidean_cost_",
    "assemble_metrised_cost_",
    "assign_simplex_",
    "classification_output_accumulate_into_source_cost_",
    "classification_output_loss_",
    "compute_gamma_wt_",
    "compute_log_partition_",
    "compute_wd_cost_categorical_",
    "compute_wd_cost_euclidean_",
    "compute_wt_cost_",
    "densify_categorical",
    "effective_dimension",
    "entropy",
    "entropy_penalty_",
    "floored_log_",
    "grouped_means",
    "inlier_scores_",
    "mark_non_empty_clusters_",
    "normalise_",
    "project",
    "propagate_classification_output_",
    "propagate_regression_output_",
    "reduce_input_cost_",
    "refresh_weighted_norm_",
    "regression_output_accumulate_into_source_cost_",
    "regression_output_loss_",
    "seed_dissimilarity",
    "softmax_with_temp_",
    "split_wd",
    "transition_partial_loss_",
    "update_categorical_centroids_",
    "update_classification_output_",
    "update_euclidean_centroids_",
    "update_output_weights_",
    "update_regression_output_",
    "update_theta_",
    "weighted_assign_simplex_",
    "weighted_sq_distance_",
    "weighted_sq_norm_",
    "weighted_subspace_",
]
