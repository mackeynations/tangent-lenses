"""Local oracles, corpus averages, and state-dependent mean-Jacobian splits."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from tangent_lenses.geometry import (
    DirectPCASpace,
    EmbeddingGeometry,
    TangentSpace,
    direct_pca_space,
    space_from_factor,
)
from tangent_lenses.hooks import ActivationRecorder
from tangent_lenses.jacobians import (
    PromptState,
    capture_prompt_state,
    compute_prompt_derivatives,
    compute_pushforward_factors,
    sample_positions,
    valid_positions,
)
from tangent_lenses.lens import ProjectedLensFamily
from tangent_lenses.model import LensModel

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("pushforward", "direct_pca")


@dataclass
class ProjectionResult:
    """A matched full/tangent/normal matrix decomposition at one state."""

    position: int
    full: dict[int, torch.Tensor]
    tangent: dict[str, dict[int, torch.Tensor]]
    normal: dict[str, dict[int, torch.Tensor]]
    projectors: dict[str, dict[int, torch.Tensor]]
    ranks: dict[str, dict[int, int]]
    source_activations: dict[int, torch.Tensor]
    target_activation: torch.Tensor

    def decomposition_error(self, method: str, layer: int) -> float:
        remainder = self.full[layer] - self.tangent[method][layer] - self.normal[method][layer]
        denominator = max(float(torch.linalg.norm(self.full[layer]).item()), 1e-12)
        return float(torch.linalg.norm(remainder).item()) / denominator


@dataclass
class LocalProjectionResult(ProjectionResult):
    """Prompt-local ``J_h``, ``J_h P_T(h)``, and ``J_h P_N(h)``."""


@dataclass
class MeanJacobianSplitResult(ProjectionResult):
    """State-dependent ``E[J]``, ``E[J] P_T(h)``, and ``E[J] P_N(h)``."""


@dataclass
class StateDependentProjectors:
    """Tangent/normal projectors estimated at one realized hidden state."""

    position: int
    projectors: dict[str, dict[int, torch.Tensor]]
    normal_projectors: dict[str, dict[int, torch.Tensor]]
    ranks: dict[str, dict[int, int]]
    source_activations: dict[int, torch.Tensor]
    target_activation: torch.Tensor


def collect_activation_bank(
    model: LensModel,
    prompts: Sequence[str],
    source_layers: Sequence[int],
    *,
    samples_per_prompt: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
) -> dict[int, torch.Tensor]:
    """Collect fixed realized-activation banks for direct local PCA."""

    layers = sorted(set(int(layer) for layer in source_layers))
    chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for prompt_index, prompt in enumerate(prompts):
        input_ids = model.encode(prompt, max_length=max_seq_len)
        try:
            selected = sample_positions(
                valid_positions(int(input_ids.shape[1]), skip_first=skip_first),
                samples_per_prompt,
            )
        except ValueError:
            logger.warning("Skipping short activation-bank prompt %d", prompt_index)
            continue
        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
        for layer in layers:
            chunks[layer].append(recorder.activations[layer][0, selected].detach().float().cpu())
    result = {
        layer: torch.cat(layer_chunks, dim=0) if layer_chunks else torch.zeros(0, model.d_model)
        for layer, layer_chunks in chunks.items()
    }
    if any(bank.shape[0] == 0 for bank in result.values()):
        raise ValueError("activation-bank corpus produced no usable samples")
    return result


def _validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(methods))
    unknown = set(result) - set(SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unknown tangent estimator(s): {sorted(unknown)}")
    if not result:
        raise ValueError("at least one tangent estimator is required")
    return result


def _prefix_factors(
    derivatives: PromptState,
    embedding_geometry: EmbeddingGeometry,
    *,
    prefix_lookback: int,
) -> dict[int, dict[int, torch.Tensor]]:
    if prefix_lookback < 0:
        raise ValueError("prefix_lookback must be non-negative")
    return {
        position: {
            prefix_position: embedding_geometry.factor(derivatives.token_ids[prefix_position]).cpu()
            for prefix_position in range(max(0, position - prefix_lookback), position + 1)
        }
        for position in derivatives.positions
    }


def _estimate_spaces(
    model: LensModel,
    derivatives: PromptState,
    source_layers: Sequence[int],
    *,
    methods: Sequence[str],
    embedding_geometry: EmbeddingGeometry | None,
    activation_bank: dict[int, torch.Tensor] | None,
    prefix_lookback: int,
    k_neighbors: int,
    direct_metric: str,
    energy_threshold: float,
    rank: int | None,
    jvp_batch_size: int,
    forward_ad_attention: str,
) -> dict[int, dict[str, dict[int, TangentSpace]]]:
    spaces: dict[int, dict[str, dict[int, TangentSpace]]] = {
        position: {method: {} for method in methods} for position in derivatives.positions
    }

    if "pushforward" in methods:
        if embedding_geometry is None:
            raise ValueError("pushforward estimation requires embedding_geometry")
        factors = compute_pushforward_factors(
            model,
            derivatives,
            source_layers,
            _prefix_factors(
                derivatives,
                embedding_geometry,
                prefix_lookback=prefix_lookback,
            ),
            jvp_batch_size=jvp_batch_size,
            forward_ad_attention=forward_ad_attention,
        )
        for position in derivatives.positions:
            for layer in source_layers:
                spaces[position]["pushforward"][layer] = space_from_factor(
                    factors[position][layer],
                    energy_threshold=energy_threshold,
                    rank=rank,
                )

    if "direct_pca" in methods:
        if activation_bank is None:
            raise ValueError("direct-PCA estimation requires an activation_bank")
        for position in derivatives.positions:
            for layer in source_layers:
                direct: DirectPCASpace = direct_pca_space(
                    derivatives.source_activations[layer][position],
                    activation_bank[layer],
                    k_neighbors=k_neighbors,
                    metric=direct_metric,
                    energy_threshold=energy_threshold,
                    rank=rank,
                )
                spaces[position]["direct_pca"][layer] = direct
    return spaces


def estimate_state_dependent_projectors(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    position: int = -1,
    target_layer: int | None = None,
    methods: Sequence[str] = SUPPORTED_METHODS,
    embedding_geometry: EmbeddingGeometry | None = None,
    activation_bank: dict[int, torch.Tensor] | None = None,
    prefix_lookback: int = 4,
    k_neighbors: int = 64,
    direct_metric: str = "l2",
    energy_threshold: float = 0.95,
    rank: int | None = None,
    jvp_batch_size: int = 8,
    max_seq_len: int = 128,
    forward_ad_attention: str = "math",
) -> StateDependentProjectors:
    """Estimate ``P_T(h)`` and ``P_N(h)`` without computing a local ``J_h``."""

    methods = _validate_methods(methods)
    layers = sorted(set(int(layer) for layer in source_layers))
    state = capture_prompt_state(
        model,
        prompt,
        layers,
        target_layer=target_layer,
        positions=[position],
        max_seq_len=max_seq_len,
    )
    resolved_position = state.positions[0]
    spaces = _estimate_spaces(
        model,
        state,
        layers,
        methods=methods,
        embedding_geometry=embedding_geometry,
        activation_bank=activation_bank,
        prefix_lookback=prefix_lookback,
        k_neighbors=k_neighbors,
        direct_metric=direct_metric,
        energy_threshold=energy_threshold,
        rank=rank,
        jvp_batch_size=jvp_batch_size,
        forward_ad_attention=forward_ad_attention,
    )[resolved_position]
    return StateDependentProjectors(
        position=resolved_position,
        projectors={
            method: {layer: spaces[method][layer].projector for layer in layers}
            for method in methods
        },
        normal_projectors={
            method: {layer: spaces[method][layer].normal_projector for layer in layers}
            for method in methods
        },
        ranks={
            method: {layer: spaces[method][layer].rank for layer in layers} for method in methods
        },
        source_activations={
            layer: state.source_activations[layer][resolved_position] for layer in layers
        },
        target_activation=state.target_activations[resolved_position],
    )


def compute_mean_jacobian_splits(
    model: LensModel,
    prompt: str,
    family: ProjectedLensFamily,
    *,
    position: int = -1,
    target_layer: int | None = None,
    methods: Sequence[str] = SUPPORTED_METHODS,
    embedding_geometry: EmbeddingGeometry | None = None,
    activation_bank: dict[int, torch.Tensor] | None = None,
    prefix_lookback: int = 4,
    k_neighbors: int = 64,
    direct_metric: str = "l2",
    energy_threshold: float = 0.95,
    rank: int | None = None,
    jvp_batch_size: int = 8,
    max_seq_len: int = 128,
    forward_ad_attention: str = "math",
) -> MeanJacobianSplitResult:
    r"""Compute ``E[J] P_T(h)`` and ``E[J] P_N(h)`` at one realized state.

    The mean matrix comes from ``family.full``. Only the state-dependent
    projectors are recomputed, so this avoids the expensive reverse-mode local
    Jacobian required by :func:`compute_local_projections`.
    """

    projectors = estimate_state_dependent_projectors(
        model,
        prompt,
        family.source_layers,
        position=position,
        target_layer=target_layer,
        methods=methods,
        embedding_geometry=embedding_geometry,
        activation_bank=activation_bank,
        prefix_lookback=prefix_lookback,
        k_neighbors=k_neighbors,
        direct_metric=direct_metric,
        energy_threshold=energy_threshold,
        rank=rank,
        jvp_batch_size=jvp_batch_size,
        max_seq_len=max_seq_len,
        forward_ad_attention=forward_ad_attention,
    )
    full = family.full
    tangent = {
        method: {
            layer: full[layer] @ projectors.projectors[method][layer]
            for layer in family.source_layers
        }
        for method in projectors.projectors
    }
    normal = {
        method: {
            layer: full[layer] @ projectors.normal_projectors[method][layer]
            for layer in family.source_layers
        }
        for method in projectors.projectors
    }
    return MeanJacobianSplitResult(
        position=projectors.position,
        full=full,
        tangent=tangent,
        normal=normal,
        projectors=projectors.projectors,
        ranks=projectors.ranks,
        source_activations=projectors.source_activations,
        target_activation=projectors.target_activation,
    )


def compute_local_projections(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    position: int = -1,
    target_layer: int | None = None,
    methods: Sequence[str] = SUPPORTED_METHODS,
    embedding_geometry: EmbeddingGeometry | None = None,
    activation_bank: dict[int, torch.Tensor] | None = None,
    prefix_lookback: int = 4,
    k_neighbors: int = 64,
    direct_metric: str = "l2",
    energy_threshold: float = 0.95,
    rank: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    broadcast: bool = True,
    forward_ad_attention: str = "math",
) -> LocalProjectionResult:
    """Compute ``J_x P_T(x)`` and ``J_x P_N(x)`` for one local oracle."""

    methods = _validate_methods(methods)
    layers = sorted(set(int(layer) for layer in source_layers))
    derivatives = compute_prompt_derivatives(
        model,
        prompt,
        layers,
        target_layer=target_layer,
        positions=[position],
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        broadcast=broadcast,
    )
    resolved_position = derivatives.positions[0]
    spaces = _estimate_spaces(
        model,
        derivatives,
        layers,
        methods=methods,
        embedding_geometry=embedding_geometry,
        activation_bank=activation_bank,
        prefix_lookback=prefix_lookback,
        k_neighbors=k_neighbors,
        direct_metric=direct_metric,
        energy_threshold=energy_threshold,
        rank=rank,
        jvp_batch_size=dim_batch,
        forward_ad_attention=forward_ad_attention,
    )[resolved_position]

    full = {layer: derivatives.downstream[layer][resolved_position] for layer in layers}
    tangent = {
        method: {layer: full[layer] @ spaces[method][layer].projector for layer in layers}
        for method in methods
    }
    normal = {
        method: {layer: full[layer] @ spaces[method][layer].normal_projector for layer in layers}
        for method in methods
    }
    return LocalProjectionResult(
        position=resolved_position,
        full=full,
        tangent=tangent,
        normal=normal,
        projectors={
            method: {layer: spaces[method][layer].projector for layer in layers}
            for method in methods
        },
        ranks={
            method: {layer: spaces[method][layer].rank for layer in layers} for method in methods
        },
        source_activations={
            layer: derivatives.source_activations[layer][resolved_position] for layer in layers
        },
        target_activation=derivatives.target_activations[resolved_position],
    )


def fit_projected_lenses(
    model: LensModel,
    prompts: Sequence[str],
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    methods: Sequence[str] = SUPPORTED_METHODS,
    embedding_geometry: EmbeddingGeometry | None = None,
    activation_bank: dict[int, torch.Tensor] | None = None,
    samples_per_prompt: int = 4,
    prefix_lookback: int = 4,
    k_neighbors: int = 64,
    direct_metric: str = "l2",
    energy_threshold: float = 0.95,
    rank: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    broadcast: bool = True,
    forward_ad_attention: str = "math",
    metadata: dict[str, Any] | None = None,
) -> ProjectedLensFamily:
    r"""Fit matched ``E[J]``, ``E[J P_T]``, and ``E[J P_N]`` estimates."""

    methods = _validate_methods(methods)
    layers = sorted(set(int(layer) for layer in source_layers))
    sums_full = {layer: torch.zeros(model.d_model, model.d_model) for layer in layers}
    sums_tangent = {
        method: {layer: torch.zeros(model.d_model, model.d_model) for layer in layers}
        for method in methods
    }
    sums_normal = {
        method: {layer: torch.zeros(model.d_model, model.d_model) for layer in layers}
        for method in methods
    }
    counts = {layer: 0 for layer in layers}

    for prompt_index, prompt in enumerate(prompts):
        try:
            derivatives = compute_prompt_derivatives(
                model,
                prompt,
                layers,
                target_layer=target_layer,
                samples_per_prompt=samples_per_prompt,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                broadcast=broadcast,
            )
        except ValueError as error:
            logger.warning("Skipping training prompt %d: %s", prompt_index, error)
            continue
        spaces_by_position = _estimate_spaces(
            model,
            derivatives,
            layers,
            methods=methods,
            embedding_geometry=embedding_geometry,
            activation_bank=activation_bank,
            prefix_lookback=prefix_lookback,
            k_neighbors=k_neighbors,
            direct_metric=direct_metric,
            energy_threshold=energy_threshold,
            rank=rank,
            jvp_batch_size=dim_batch,
            forward_ad_attention=forward_ad_attention,
        )
        for position in derivatives.positions:
            for layer in layers:
                full = derivatives.downstream[layer][position]
                sums_full[layer] += full
                counts[layer] += 1
                for method in methods:
                    space = spaces_by_position[position][method][layer]
                    sums_tangent[method][layer] += full @ space.projector
                    sums_normal[method][layer] += full @ space.normal_projector
        logger.info(
            "Processed fit prompt %d/%d (%d positions)",
            prompt_index + 1,
            len(prompts),
            len(derivatives.positions),
        )

    if any(count == 0 for count in counts.values()):
        raise ValueError("training corpus produced no samples")
    full_mean = {layer: sums_full[layer] / counts[layer] for layer in layers}
    tangent_mean = {
        method: {layer: sums_tangent[method][layer] / counts[layer] for layer in layers}
        for method in methods
    }
    normal_mean = {
        method: {layer: sums_normal[method][layer] / counts[layer] for layer in layers}
        for method in methods
    }
    fit_metadata = {
        "methods": list(methods),
        "samples_per_prompt": samples_per_prompt,
        "prefix_lookback": prefix_lookback,
        "k_neighbors": k_neighbors,
        "direct_metric": direct_metric,
        "energy_threshold": energy_threshold,
        "rank": rank,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "broadcast": broadcast,
        **(metadata or {}),
    }
    return ProjectedLensFamily(
        full_mean,
        tangent_mean,
        normal_mean,
        n_samples=counts,
        metadata=fit_metadata,
    )
