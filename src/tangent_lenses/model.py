"""Small model protocol used by the geometry code."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn


class LensModel(Protocol):
    """Interface required by the experiment implementation."""

    n_layers: int
    d_model: int
    layers: Sequence[nn.Module]
    tokenizer: Any
    input_device: torch.device
    embedding_weight: torch.Tensor

    def encode(self, text: str, *, max_length: int = ...) -> torch.Tensor: ...

    def forward(self, input_ids: torch.Tensor) -> Any: ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor: ...
