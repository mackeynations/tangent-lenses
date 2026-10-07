"""Tangent- and normal-projected Jacobian lens experiments."""

from tangent_lenses.geometry import (
    DirectPCASpace,
    EmbeddingGeometry,
    TangentSpace,
    direct_pca_space,
    space_from_factor,
)
from tangent_lenses.hf import HFLensModel, Layout, from_hf, load_hf_model
from tangent_lenses.lens import LensView, ProjectedLensFamily
from tangent_lenses.model import LensModel
from tangent_lenses.study import (
    LocalProjectionResult,
    MeanJacobianSplitResult,
    StateDependentProjectors,
    collect_activation_bank,
    compute_local_projections,
    compute_mean_jacobian_splits,
    estimate_state_dependent_projectors,
    fit_projected_lenses,
)

__all__ = [
    "DirectPCASpace",
    "EmbeddingGeometry",
    "HFLensModel",
    "Layout",
    "LensModel",
    "LensView",
    "LocalProjectionResult",
    "MeanJacobianSplitResult",
    "ProjectedLensFamily",
    "StateDependentProjectors",
    "TangentSpace",
    "collect_activation_bank",
    "compute_local_projections",
    "compute_mean_jacobian_splits",
    "direct_pca_space",
    "estimate_state_dependent_projectors",
    "fit_projected_lenses",
    "from_hf",
    "load_hf_model",
    "space_from_factor",
]
