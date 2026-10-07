from __future__ import annotations

import torch
from conftest import TinyModel

from tangent_lenses.geometry import EmbeddingGeometry
from tangent_lenses.lens import ProjectedLensFamily
from tangent_lenses.study import (
    collect_activation_bank,
    compute_local_projections,
    compute_mean_jacobian_splits,
    fit_projected_lenses,
)


def test_local_oracle_builds_both_estimators_and_complements() -> None:
    model = TinyModel()
    prompts = ["abcdefgh", "ijklmnop", "qrstuvwx"]
    bank = collect_activation_bank(
        model,
        prompts,
        [1],
        samples_per_prompt=3,
        skip_first=0,
    )
    embedding = EmbeddingGeometry(model.embedding_weight, k_neighbors=4, energy_threshold=0.9)
    result = compute_local_projections(
        model,
        "different prompt",
        [1],
        position=3,
        target_layer=2,
        embedding_geometry=embedding,
        activation_bank=bank,
        prefix_lookback=1,
        k_neighbors=4,
        energy_threshold=0.9,
        dim_batch=2,
        broadcast=False,
    )
    assert set(result.tangent) == {"pushforward", "direct_pca"}
    for method in result.tangent:
        assert result.decomposition_error(method, 1) < 1e-5


def test_fitted_family_preserves_matched_decomposition(tmp_path) -> None:
    model = TinyModel()
    prompts = ["abcdefgh", "ijklmnop", "qrstuvwx"]
    bank = collect_activation_bank(
        model,
        prompts,
        [1],
        samples_per_prompt=3,
        skip_first=0,
    )
    embedding = EmbeddingGeometry(model.embedding_weight, k_neighbors=4, energy_threshold=0.9)
    family = fit_projected_lenses(
        model,
        prompts[:2],
        [1],
        target_layer=2,
        embedding_geometry=embedding,
        activation_bank=bank,
        samples_per_prompt=1,
        prefix_lookback=1,
        k_neighbors=4,
        energy_threshold=0.9,
        dim_batch=2,
        skip_first=0,
        broadcast=False,
    )
    for method in family.methods:
        assert family.decomposition_error(method, 1) < 1e-5

    checkpoint = tmp_path / "family.pt"
    family.save(checkpoint, dtype=torch.float32)
    restored = ProjectedLensFamily.load(checkpoint)
    assert restored.methods == family.methods
    torch.testing.assert_close(restored.full[1], family.full[1])


def test_mean_jacobian_can_be_split_by_state_dependent_projectors() -> None:
    model = TinyModel()
    prompts = ["abcdefgh", "ijklmnop", "qrstuvwx"]
    bank = collect_activation_bank(
        model,
        prompts,
        [1],
        samples_per_prompt=3,
        skip_first=0,
    )
    embedding = EmbeddingGeometry(model.embedding_weight, k_neighbors=4, energy_threshold=0.9)
    family = fit_projected_lenses(
        model,
        prompts[:2],
        [1],
        target_layer=2,
        embedding_geometry=embedding,
        activation_bank=bank,
        samples_per_prompt=1,
        prefix_lookback=1,
        k_neighbors=4,
        energy_threshold=0.9,
        dim_batch=2,
        skip_first=0,
        broadcast=False,
    )
    result = compute_mean_jacobian_splits(
        model,
        "different prompt",
        family,
        position=3,
        target_layer=2,
        embedding_geometry=embedding,
        activation_bank=bank,
        prefix_lookback=1,
        k_neighbors=4,
        energy_threshold=0.9,
        jvp_batch_size=2,
    )

    assert set(result.tangent) == {"pushforward", "direct_pca"}
    torch.testing.assert_close(result.full[1], family.full[1])
    for method in result.tangent:
        torch.testing.assert_close(
            result.tangent[method][1],
            family.full[1] @ result.projectors[method][1],
        )
        assert result.decomposition_error(method, 1) < 1e-5
