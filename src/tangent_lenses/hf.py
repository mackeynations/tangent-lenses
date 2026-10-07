"""Hugging Face adapter for decoder-only language models."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _resolve(obj: Any, path: str) -> Any:
    return functools.reduce(getattr, path.split("."), obj)


@dataclass(frozen=True)
class Layout:
    """Locations of the decoder components inside a Hugging Face model."""

    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


_LAYOUTS = (
    Layout("model"),
    Layout("model.language_model"),
    Layout("language_model"),
    Layout("model", norm="final_layernorm"),
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),
    Layout("gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"),
)


def _find_layout(hf_model: nn.Module) -> Layout:
    for layout in _LAYOUTS:
        try:
            decoder = _resolve(hf_model, layout.path)
        except AttributeError:
            continue
        if all(
            hasattr(decoder, name) for name in (layout.layers, layout.norm, layout.embed)
        ) and hasattr(hf_model, layout.lm_head):
            return layout
    raise ValueError(f"Could not identify the residual stack in {type(hf_model).__name__}")


class HFLensModel:
    """Expose a loaded Hugging Face causal LM through :class:`LensModel`."""

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        layout: Layout | None = None,
        force_bos: bool = True,
    ) -> None:
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        if (
            force_bos
            and getattr(tokenizer, "bos_token_id", None) is not None
            and hasattr(tokenizer, "add_bos_token")
        ):
            tokenizer.add_bos_token = True

        hf_model.eval()
        for parameter in hf_model.parameters():
            parameter.requires_grad_(False)

        self.layout = layout or _find_layout(hf_model)
        self._decoder = _resolve(hf_model, self.layout.path)
        self.layers: nn.ModuleList = getattr(self._decoder, self.layout.layers)
        self._norm: nn.Module = getattr(self._decoder, self.layout.norm)
        self._embed: nn.Module = getattr(self._decoder, self.layout.embed)
        self._head: nn.Module = getattr(hf_model, self.layout.lm_head)

        config = hf_model.config.get_text_config()
        self.n_layers = int(config.num_hidden_layers)
        self.d_model = int(config.hidden_size)
        self._softcap = getattr(config, "final_logit_softcapping", None)

    @property
    def input_device(self) -> torch.device:
        return self._embed.weight.device

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self._embed.weight

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return encoded.input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self._decoder(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        logits = self._head(
            self._norm(residual.to(device=self._head.weight.device, dtype=self._head.weight.dtype))
        )
        if self._softcap is not None:
            logits = self._softcap * torch.tanh(logits / self._softcap)
        return logits


def from_hf(
    hf_model: nn.Module,
    tokenizer: Any,
    *,
    layout: Layout | None = None,
    force_bos: bool = True,
) -> HFLensModel:
    return HFLensModel(hf_model, tokenizer, layout=layout, force_bos=force_bos)


def load_hf_model(
    model_name: str,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
) -> HFLensModel:
    """Load and adapt a Hugging Face model without hiding placement choices."""

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    return from_hf(hf_model, tokenizer)
