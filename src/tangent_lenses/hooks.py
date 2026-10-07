"""Scoped hooks for recording and replacing residual-stream activations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn


def _hidden(output: Any) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    return hidden if torch.is_tensor(output) else (hidden, *output[1:])


class ActivationRecorder:
    """Record block outputs while retaining their autograd graph."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(
            set(at) | ({start_graph_at} if start_graph_at is not None else set())
        )
        self._start_graph_at = start_graph_at
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            hidden = _hidden(output)
            if index == self._start_graph_at:
                hidden.requires_grad_(True)
            self.activations[index] = hidden

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class ActivationReplacement:
    """Replace the complete residual tensor at one block output."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        layer: int,
        replacement: torch.Tensor,
    ) -> None:
        self._block = blocks[layer]
        self._replacement = replacement
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        hidden = _hidden(output)
        replacement = self._replacement.to(device=hidden.device, dtype=hidden.dtype)
        return _replace_hidden(output, replacement)

    def __enter__(self) -> ActivationReplacement:
        self._handle = self._block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
