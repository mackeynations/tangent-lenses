"""Rank, KL, and residual-alignment evaluation helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tangent_lenses.hooks import ActivationRecorder
from tangent_lenses.lens import ProjectedLensFamily
from tangent_lenses.model import LensModel
from tangent_lenses.study import LocalProjectionResult, MeanJacobianSplitResult


def load_evaluation_items(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"evaluation file {path} must contain a list or an 'items' list")
    return items


def benchmark_position(suite: str, seq_len: int, item: dict[str, Any]) -> int:
    if "probe_position" in item:
        raw = int(item["probe_position"])
        return raw + seq_len if raw < 0 else raw
    name = suite.removeprefix("lens-eval-")
    return max(0, seq_len - 1 if name in {"association", "typo"} else seq_len - 2)


def _first_token_ids(model: LensModel, text: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        return []
    ids: list[int] = []
    tokenizer_encode = getattr(model.tokenizer, "encode", None)
    if callable(tokenizer_encode):
        for variant in (f" {stripped}", stripped):
            try:
                encoded = tokenizer_encode(variant, add_special_tokens=False)
            except TypeError:
                encoded = []
            if encoded:
                ids.append(int(encoded[0]))
    if ids:
        return list(dict.fromkeys(ids))
    bos = getattr(model.tokenizer, "bos_token_id", None)
    for variant in (f" {stripped}", stripped):
        encoded = model.encode(variant).flatten().tolist()
        if encoded and bos is not None and encoded[0] == bos:
            encoded = encoded[1:]
        if encoded:
            ids.append(int(encoded[0]))
    return list(dict.fromkeys(ids))


def target_token_ids(model: LensModel, item: dict[str, Any]) -> list[int]:
    """Resolve the first token of every accepted target/intermediate string."""

    raw_targets = item.get("intermediates") or item.get("target") or item.get("expected") or []
    if not isinstance(raw_targets, list):
        raw_targets = [raw_targets]
    flattened: list[Any] = []
    for value in raw_targets:
        flattened.extend(value if isinstance(value, list) else [value])
    result: list[int] = []
    for value in flattened:
        result.extend(_first_token_ids(model, str(value)))
    return list(dict.fromkeys(result))


def _rank(logits: torch.Tensor, targets: Sequence[int]) -> int:
    flat = logits.float().flatten()
    return min(int((flat > flat[token]).sum().item()) + 1 for token in targets)


def _layer_score(
    model: LensModel,
    source: torch.Tensor,
    target: torch.Tensor,
    matrix: torch.Tensor,
    target_ids: Sequence[int],
) -> dict[str, float | int]:
    transported = source.float() @ matrix.float().T
    predicted_logits = (
        model.unembed(transported.to(model.input_device)).detach().float().cpu().flatten()
    )
    target_logits = model.unembed(target.to(model.input_device)).detach().float().cpu().flatten()
    target_probs = torch.softmax(target_logits, dim=-1)
    kl = torch.sum(
        target_probs
        * (torch.log_softmax(target_logits, dim=-1) - torch.log_softmax(predicted_logits, dim=-1))
    )
    return {
        "rank": _rank(predicted_logits, target_ids),
        "kl": float(kl.item()),
        "cosine": float(F.cosine_similarity(transported, target.float(), dim=0).item()),
    }


def score_local_result(
    model: LensModel,
    result: LocalProjectionResult,
    target_ids: Sequence[int],
) -> dict[str, dict[str, Any]]:
    variants = {"full": result.full}
    for method in sorted(result.tangent):
        variants[f"{method}_tangent"] = result.tangent[method]
        variants[f"{method}_normal"] = result.normal[method]
    return _score_variants(
        model,
        variants,
        result.source_activations,
        result.target_activation,
        target_ids,
    )


def score_mean_jacobian_split_result(
    model: LensModel,
    result: MeanJacobianSplitResult,
    target_ids: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Score ``E[J] P_T(h)`` and ``E[J] P_N(h)`` with unambiguous labels."""

    variants = {"mean_jacobian": result.full}
    for method in sorted(result.tangent):
        variants[f"{method}_mean_split_tangent"] = result.tangent[method]
        variants[f"{method}_mean_split_normal"] = result.normal[method]
    return _score_variants(
        model,
        variants,
        result.source_activations,
        result.target_activation,
        target_ids,
    )


def _score_variants(
    model: LensModel,
    variants: dict[str, dict[int, torch.Tensor]],
    source_activations: dict[int, torch.Tensor],
    target_activation: torch.Tensor,
    target_ids: Sequence[int],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, matrices in variants.items():
        layer_records = {
            str(layer): _layer_score(
                model,
                source_activations[layer],
                target_activation,
                matrices[layer],
                target_ids,
            )
            for layer in sorted(matrices)
        }
        output[name] = {
            "best_rank": min(int(record["rank"]) for record in layer_records.values()),
            "workspace_mean_kl": float(
                np.mean([record["kl"] for record in layer_records.values()])
            ),
            "workspace_mean_cosine": float(
                np.mean([record["cosine"] for record in layer_records.values()])
            ),
            "layers": layer_records,
        }
    return output


def evaluate_lens_family(
    model: LensModel,
    family: ProjectedLensFamily,
    items: Sequence[dict[str, Any]],
    *,
    suite: str,
    target_layer: int | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate all family variants on a fixed list of examples."""

    target = model.n_layers - 1 if target_layer is None else int(target_layer)
    variants = {"full": family.full}
    for method in family.methods:
        variants[f"{method}_tangent"] = family.tangent[method]
        variants[f"{method}_normal"] = family.normal[method]
    records: list[dict[str, Any]] = []
    for item_index, item in enumerate(items, start=start_index):
        targets = target_token_ids(model, item)
        if not targets:
            continue
        input_ids = model.encode(item["prompt"])
        position = benchmark_position(suite, int(input_ids.shape[1]), item)
        with (
            torch.no_grad(),
            ActivationRecorder(
                model.layers,
                at=[*family.source_layers, target],
            ) as recorder,
        ):
            model.forward(input_ids)
        source = {
            layer: recorder.activations[layer][0, position].detach().float().cpu()
            for layer in family.source_layers
        }
        target_activation = recorder.activations[target][0, position].detach().float().cpu()
        records.append(
            {
                "suite": suite,
                "item_index": item_index,
                "probe_position": position,
                "variants": _score_variants(model, variants, source, target_activation, targets),
            }
        )
    return records


def summarize_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    if not records:
        return {}
    variant_names = list(records[0]["variants"])
    summary: dict[str, dict[str, float | int | None]] = {}
    for variant in variant_names:
        ranks = np.asarray(
            [record["variants"][variant]["best_rank"] for record in records],
            dtype=np.int64,
        )
        kls = [record["variants"][variant]["workspace_mean_kl"] for record in records]
        cosines = [record["variants"][variant]["workspace_mean_cosine"] for record in records]
        summary[variant] = {
            "n_examples": int(ranks.size),
            "pass1": float(np.mean(ranks <= 1)),
            "pass5": float(np.mean(ranks <= 5)),
            "pass10": float(np.mean(ranks <= 10)),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "mean_workspace_kl": float(np.mean(kls)),
            "mean_workspace_cosine": float(np.mean(cosines)),
        }
    return summary
