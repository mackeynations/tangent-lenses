"""Tangent-space estimators and orthogonal projectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def energy_rank(singular_values: torch.Tensor, threshold: float = 0.95) -> int:
    """Smallest rank whose squared singular values meet ``threshold`` energy."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("energy threshold must lie in (0, 1]")
    if singular_values.numel() == 0:
        return 0
    energy = singular_values.float().square()
    total = energy.sum()
    if total <= 0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return min(int(torch.searchsorted(cumulative, threshold).item()) + 1, singular_values.numel())


@dataclass(frozen=True)
class TangentSpace:
    """An orthonormal tangent basis and its hard projector."""

    basis: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    projector: torch.Tensor

    @property
    def normal_projector(self) -> torch.Tensor:
        return (
            torch.eye(
                self.projector.shape[0],
                dtype=self.projector.dtype,
                device=self.projector.device,
            )
            - self.projector
        )


@dataclass(frozen=True)
class DirectPCASpace(TangentSpace):
    neighbor_count: int
    neighbor_distances: torch.Tensor


def _space_from_basis_spectrum(
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    *,
    energy_threshold: float,
    rank: int | None,
) -> TangentSpace:
    available = min(basis.shape[1], singular_values.numel())
    if rank is not None and rank <= 0:
        raise ValueError("rank must be positive")
    chosen = (
        energy_rank(singular_values[:available], energy_threshold) if rank is None else int(rank)
    )
    chosen = max(0, min(chosen, available))
    selected = basis[:, :chosen]
    projector = selected @ selected.T
    return TangentSpace(
        basis=selected.float(),
        singular_values=singular_values.float(),
        rank=chosen,
        projector=projector.float(),
    )


def space_from_factor(
    factor: torch.Tensor,
    *,
    energy_threshold: float = 0.95,
    rank: int | None = None,
) -> TangentSpace:
    """Build the tangent span of a (possibly rectangular) pushforward factor."""

    if factor.ndim != 2:
        raise ValueError("factor must have shape [ambient_dim, n_directions]")
    ambient = factor.shape[0]
    if factor.shape[1] == 0:
        empty = factor.new_zeros((ambient, 0), dtype=torch.float32)
        return TangentSpace(
            empty, factor.new_zeros(0).float(), 0, factor.new_zeros((ambient, ambient)).float()
        )
    u, singular_values, _ = torch.linalg.svd(factor.float(), full_matrices=False)
    return _space_from_basis_spectrum(
        u,
        singular_values,
        energy_threshold=energy_threshold,
        rank=rank,
    )


def direct_pca_space(
    query: torch.Tensor,
    activation_bank: torch.Tensor,
    *,
    k_neighbors: int = 64,
    metric: str = "l2",
    energy_threshold: float = 0.95,
    rank: int | None = None,
) -> DirectPCASpace:
    """Estimate a local tangent by PCA of neighboring realized activations."""

    query = query.detach().float().cpu().flatten()
    bank = activation_bank.detach().float().cpu()
    if bank.ndim != 2 or bank.shape[1] != query.numel():
        raise ValueError("activation bank must have shape [n_samples, d_model]")
    if metric not in {"l2", "cosine"}:
        raise ValueError("metric must be 'l2' or 'cosine'")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    if bank.shape[0] == 0:
        raise ValueError("direct PCA requires a non-empty activation bank")

    n_candidates = min(bank.shape[0], max(1, k_neighbors + 1))
    if metric == "cosine":
        similarities = F.normalize(bank, dim=-1) @ F.normalize(query, dim=-1)
        indices = torch.topk(similarities, k=n_candidates).indices
        distances = 1.0 - similarities[indices]
    else:
        all_distances = torch.linalg.vector_norm(bank - query.unsqueeze(0), dim=-1)
        indices = torch.topk(all_distances, k=n_candidates, largest=False).indices
        distances = all_distances[indices]

    differences = bank[indices] - query.unsqueeze(0)
    keep = torch.linalg.vector_norm(differences, dim=-1) > 1e-8
    differences = differences[keep]
    distances = distances[keep]
    if differences.shape[0] == 0:
        raise ValueError("all direct-PCA neighbors coincide with the query")

    _, singular_values, vh = torch.linalg.svd(differences, full_matrices=False)
    space = _space_from_basis_spectrum(
        vh.T,
        singular_values,
        energy_threshold=energy_threshold,
        rank=rank,
    )
    return DirectPCASpace(
        basis=space.basis,
        singular_values=space.singular_values,
        rank=space.rank,
        projector=space.projector,
        neighbor_count=int(differences.shape[0]),
        neighbor_distances=distances,
    )


class EmbeddingGeometry:
    """Cached local k-NN PCA factors on the token embedding table."""

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        *,
        k_neighbors: int = 64,
        energy_threshold: float = 0.95,
        device: str | torch.device | None = None,
    ) -> None:
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        target = embedding_weight.device if device is None else torch.device(device)
        self.weight = embedding_weight.detach().float().to(target)
        self.normalized = F.normalize(self.weight, dim=-1)
        self.k_neighbors = int(k_neighbors)
        self.energy_threshold = float(energy_threshold)
        self._cache: dict[int, torch.Tensor] = {}

    def factor(self, token_id: int) -> torch.Tensor:
        """Return ``B`` such that ``BB^T`` is the local embedding projector."""

        if token_id in self._cache:
            return self._cache[token_id]
        center = self.weight[token_id]
        similarities = self.normalized @ self.normalized[token_id]
        count = min(self.weight.shape[0], max(1, self.k_neighbors))
        indices = torch.topk(similarities, k=count).indices
        differences = self.weight[indices] - center
        differences = differences[torch.linalg.vector_norm(differences, dim=-1) > 1e-8]
        if differences.shape[0] == 0:
            factor = torch.eye(self.weight.shape[1], device=self.weight.device)
        else:
            _, singular_values, vh = torch.linalg.svd(differences, full_matrices=False)
            chosen = energy_rank(singular_values, self.energy_threshold)
            factor = vh[:chosen].T.contiguous()
        self._cache[token_id] = factor
        return factor
