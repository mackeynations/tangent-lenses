from __future__ import annotations

import torch

from tangent_lenses.geometry import direct_pca_space, energy_rank, space_from_factor


def test_energy_rank_uses_squared_singular_values() -> None:
    singular_values = torch.tensor([3.0, 1.0, 1.0])
    assert energy_rank(singular_values, 0.80) == 1
    assert energy_rank(singular_values, 0.95) == 3


def test_factor_space_is_an_orthogonal_split() -> None:
    factor = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    space = space_from_factor(factor, energy_threshold=0.75)
    projector = space.projector
    assert space.rank == 1
    torch.testing.assert_close(projector, projector.T)
    torch.testing.assert_close(projector @ projector, projector)
    torch.testing.assert_close(projector + space.normal_projector, torch.eye(3))


def test_direct_pca_recovers_planar_neighbors() -> None:
    query = torch.zeros(3)
    bank = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    space = direct_pca_space(query, bank, k_neighbors=4, energy_threshold=0.99)
    assert space.rank == 2
    torch.testing.assert_close(space.projector @ torch.tensor([0.0, 0.0, 1.0]), torch.zeros(3))
