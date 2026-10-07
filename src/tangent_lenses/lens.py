"""Serializable collection of full, tangent, and normal average Jacobians."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

ProjectionPart = Literal["full", "tangent", "normal"]


class ProjectedLensFamily:
    r"""Matched estimates of ``E[J]``, ``E[J P_T]``, and ``E[J P_N]``.

    Tangent and normal estimates for every method are accumulated over the
    same prompt-position samples. This makes the decomposition auditable.
    """

    def __init__(
        self,
        full: dict[int, torch.Tensor],
        tangent: dict[str, dict[int, torch.Tensor]],
        normal: dict[str, dict[int, torch.Tensor]],
        *,
        n_samples: dict[int, int],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.full = _float_cpu(full)
        self.tangent = {method: _float_cpu(matrices) for method, matrices in tangent.items()}
        self.normal = {method: _float_cpu(matrices) for method, matrices in normal.items()}
        self.n_samples = {int(layer): int(count) for layer, count in n_samples.items()}
        self.metadata = dict(metadata or {})
        self.source_layers = sorted(self.full)
        if not self.source_layers:
            raise ValueError("a projected lens needs at least one layer")
        self.d_model = int(self.full[self.source_layers[0]].shape[0])
        self.methods = sorted(self.tangent)
        self._validate()

    def _validate(self) -> None:
        expected_layers = set(self.source_layers)
        for layer, matrix in self.full.items():
            if matrix.shape != (self.d_model, self.d_model):
                raise ValueError(f"full matrix at layer {layer} has shape {tuple(matrix.shape)}")
        if set(self.tangent) != set(self.normal):
            raise ValueError("tangent and normal methods differ")
        for method in self.methods:
            if (
                set(self.tangent[method]) != expected_layers
                or set(self.normal[method]) != expected_layers
            ):
                raise ValueError(f"method {method!r} does not cover every source layer")
            for layer in self.source_layers:
                tangent = self.tangent[method][layer]
                normal = self.normal[method][layer]
                if tangent.shape != (self.d_model, self.d_model) or normal.shape != tangent.shape:
                    raise ValueError(f"invalid matrix shape for method={method}, layer={layer}")

    def decomposition_error(self, method: str, layer: int) -> float:
        """Relative error in ``E[J] = E[JP_T] + E[JP_N]``."""

        residual = self.full[layer] - self.tangent[method][layer] - self.normal[method][layer]
        denominator = max(float(torch.linalg.norm(self.full[layer]).item()), 1e-12)
        return float(torch.linalg.norm(residual).item()) / denominator

    def matrix(
        self, layer: int, *, part: ProjectionPart = "full", method: str | None = None
    ) -> torch.Tensor:
        if part == "full":
            return self.full[layer]
        if method is None:
            raise ValueError(f"method is required for part={part!r}")
        collection = self.tangent if part == "tangent" else self.normal
        return collection[method][layer]

    def transport(
        self,
        residual: torch.Tensor,
        layer: int,
        *,
        part: ProjectionPart = "full",
        method: str | None = None,
    ) -> torch.Tensor:
        matrix = self.matrix(layer, part=part, method=method).to(
            device=residual.device,
            dtype=residual.dtype,
        )
        return residual @ matrix.T

    def view(self, *, part: ProjectionPart = "full", method: str | None = None) -> LensView:
        return LensView(self, part=part, method=method)

    def save(self, path: str | Path, *, dtype: torch.dtype = torch.float16) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "full": {layer: matrix.to(dtype) for layer, matrix in self.full.items()},
                "tangent": {
                    method: {layer: matrix.to(dtype) for layer, matrix in matrices.items()}
                    for method, matrices in self.tangent.items()
                },
                "normal": {
                    method: {layer: matrix.to(dtype) for layer, matrix in matrices.items()}
                    for method, matrices in self.normal.items()
                },
                "n_samples": self.n_samples,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ProjectedLensFamily:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported projected-lens checkpoint")
        return cls(
            payload["full"],
            payload["tangent"],
            payload["normal"],
            n_samples=payload["n_samples"],
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class LensView:
    """A fixed full/tangent/normal view with a conventional transport method."""

    family: ProjectedLensFamily
    part: ProjectionPart = "full"
    method: str | None = None

    @property
    def source_layers(self) -> list[int]:
        return self.family.source_layers

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        return self.family.transport(residual, layer, part=self.part, method=self.method)


def _float_cpu(matrices: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {int(layer): matrix.detach().float().cpu() for layer, matrix in matrices.items()}
