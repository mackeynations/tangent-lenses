from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class TinyTokenizer:
    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1 + (ord(character) % 15) for character in text] or [1]


class TinyBlock(nn.Module):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.matrix.T


class TinyModel(nn.Module):
    def __init__(self, d_model: int = 4, vocab_size: int = 16) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.d_model = d_model
        self.n_layers = 3
        self.tokenizer = TinyTokenizer()
        self.embedding = nn.Embedding(vocab_size, d_model)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(vocab_size, d_model, generator=generator))
        matrices = [
            torch.eye(d_model) + 0.1 * torch.randn(d_model, d_model, generator=generator)
            for _ in range(self.n_layers)
        ]
        self.layers = nn.ModuleList(TinyBlock(matrix) for matrix in matrices)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(vocab_size, d_model, generator=generator))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def input_device(self) -> torch.device:
        return self.embedding.weight.device

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.embedding.weight

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text)[:max_length]
        return torch.tensor([token_ids], dtype=torch.long)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.head(residual)
