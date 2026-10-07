"""Command-line entry points for the two paper experiment families."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from tangent_lenses.evaluation import (
    benchmark_position,
    evaluate_lens_family,
    load_evaluation_items,
    score_local_result,
    score_mean_jacobian_split_result,
    summarize_records,
    target_token_ids,
)
from tangent_lenses.geometry import EmbeddingGeometry
from tangent_lenses.hf import load_hf_model
from tangent_lenses.lens import ProjectedLensFamily
from tangent_lenses.study import (
    collect_activation_bank,
    compute_local_projections,
    compute_mean_jacobian_splits,
    fit_projected_lenses,
)

logger = logging.getLogger("tangent_lenses")


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _layers(value: str) -> list[int]:
    result = [int(part) for part in _csv(value)]
    if not result:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return result


def _dtype(value: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return mapping[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unknown dtype {value!r}") from error


def _load_prompts(path: str | Path, limit: int | None = None) -> list[str]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    prompts = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"{path} must contain a string list or a 'prompts' string list")
    return prompts if limit is None else prompts[:limit]


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def _load_or_build_bank(
    args: argparse.Namespace, model, prompts: list[str]
) -> dict[int, torch.Tensor] | None:
    if "direct_pca" not in args.methods:
        return None
    if args.activation_bank and Path(args.activation_bank).exists():
        payload = torch.load(args.activation_bank, map_location="cpu", weights_only=True)
        if payload.get("model_name") not in {None, args.model}:
            raise ValueError(
                f"activation bank was built for {payload['model_name']!r}, not {args.model!r}"
            )
        bank = {int(layer): tensor.float() for layer, tensor in payload["activations"].items()}
        if set(bank) != set(args.layers):
            raise ValueError(
                f"activation-bank layers {sorted(bank)} do not match requested layers {args.layers}"
            )
        if any(tensor.ndim != 2 or tensor.shape[1] != model.d_model for tensor in bank.values()):
            raise ValueError("activation-bank hidden size does not match the loaded model")
        logger.info("Loaded activation bank from %s", args.activation_bank)
        return bank
    if not args.bank_corpus and not prompts:
        raise ValueError("direct PCA requires --bank-corpus or an existing --activation-bank")
    bank_prompts = (
        _load_prompts(args.bank_corpus, args.n_bank) if args.bank_corpus else prompts[: args.n_bank]
    )
    bank = collect_activation_bank(
        model,
        bank_prompts,
        args.layers,
        samples_per_prompt=args.bank_samples_per_prompt,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
    )
    if args.activation_bank:
        path = Path(args.activation_bank)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": args.model,
                "layers": args.layers,
                "samples_per_prompt": args.bank_samples_per_prompt,
                "activations": bank,
            },
            path,
        )
        logger.info("Saved activation bank to %s", path)
    return bank


def _embedding_geometry(args: argparse.Namespace, model) -> EmbeddingGeometry | None:
    if "pushforward" not in args.methods:
        return None
    return EmbeddingGeometry(
        model.embedding_weight,
        k_neighbors=args.k_neighbors,
        energy_threshold=args.energy_threshold,
        device="cpu",
    )


def _load_model(args: argparse.Namespace):
    return load_hf_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )


def _run_fit(args: argparse.Namespace) -> None:
    prompts = _load_prompts(args.corpus, args.n_train)
    model = _load_model(args)
    bank = _load_or_build_bank(args, model, prompts)
    family = fit_projected_lenses(
        model,
        prompts,
        args.layers,
        target_layer=args.target_layer,
        methods=args.methods,
        embedding_geometry=_embedding_geometry(args, model),
        activation_bank=bank,
        samples_per_prompt=args.samples_per_prompt,
        prefix_lookback=args.prefix_lookback,
        k_neighbors=args.k_neighbors,
        direct_metric=args.direct_metric,
        energy_threshold=args.energy_threshold,
        rank=args.rank,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        broadcast=args.broadcast,
        forward_ad_attention=args.forward_ad_attention,
        metadata={"model_name": args.model, "target_layer": args.target_layer},
    )
    family.save(args.output)
    audit = {
        "checkpoint": str(args.output),
        "model_name": args.model,
        "layers": family.source_layers,
        "n_samples": family.n_samples,
        "decomposition_relative_error": {
            method: {
                str(layer): family.decomposition_error(method, layer)
                for layer in family.source_layers
            }
            for method in family.methods
        },
        "metadata": family.metadata,
    }
    _write_json(Path(args.output).with_suffix(".json"), audit)


def _run_evaluate(args: argparse.Namespace) -> None:
    family = _load_compatible_family(args)
    model = _load_model(args)
    all_records: list[dict[str, Any]] = []
    by_suite: dict[str, Any] = {}
    for eval_path in args.eval:
        suite = Path(eval_path).stem.removeprefix("lens-eval-")
        items = load_evaluation_items(eval_path)[args.start : args.start + args.limit]
        records = evaluate_lens_family(
            model,
            family,
            items,
            suite=suite,
            target_layer=args.target_layer,
            start_index=args.start,
        )
        all_records.extend(records)
        by_suite[suite] = summarize_records(records)
    _write_json(
        args.output,
        {
            "model_name": args.model,
            "checkpoint": args.checkpoint,
            "by_suite": by_suite,
            "overall": summarize_records(all_records),
            "records": all_records,
        },
    )


def _load_compatible_family(args: argparse.Namespace) -> ProjectedLensFamily:
    family = ProjectedLensFamily.load(args.checkpoint)
    checkpoint_model = family.metadata.get("model_name")
    if checkpoint_model is not None and checkpoint_model != args.model:
        raise ValueError(f"checkpoint was fit for {checkpoint_model!r}, not {args.model!r}")
    if family.source_layers != args.layers:
        raise ValueError(
            f"checkpoint layers {family.source_layers} do not match requested layers {args.layers}"
        )
    return family


def _run_mean_split(args: argparse.Namespace) -> None:
    """Evaluate state-dependent ``E[J] P_T(h)`` and ``E[J] P_N(h)``."""

    family = _load_compatible_family(args)
    model = _load_model(args)
    bank_prompts = _load_prompts(args.bank_corpus, args.n_bank) if args.bank_corpus else []
    bank = _load_or_build_bank(args, model, bank_prompts)
    embedding_geometry = _embedding_geometry(args, model)
    output_path = Path(args.output)
    config = {
        "experiment": "state_dependent_mean_jacobian_split",
        "definition": {
            "tangent": "E[J] P_T(h)",
            "normal": "E[J] P_N(h)",
        },
        "model_name": args.model,
        "checkpoint": str(args.checkpoint),
        "layers": args.layers,
        "target_layer": args.target_layer,
        "methods": args.methods,
        "start": args.start,
        "limit": args.limit,
        "prefix_lookback": args.prefix_lookback,
        "k_neighbors": args.k_neighbors,
        "direct_metric": args.direct_metric,
        "energy_threshold": args.energy_threshold,
        "rank": args.rank,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "forward_ad_attention": args.forward_ad_attention,
        "activation_bank": args.activation_bank,
        "evaluation_files": [str(path) for path in args.eval],
    }
    records: list[dict[str, Any]] = []
    if output_path.exists():
        with output_path.open() as handle:
            existing = json.load(handle)
        if existing.get("config") != config:
            raise ValueError("existing mean-split output has a different configuration")
        records = existing.get("records", [])
    completed = {(record["suite"], record["item_index"]) for record in records}

    for eval_path in args.eval:
        suite = Path(eval_path).stem.removeprefix("lens-eval-")
        items = load_evaluation_items(eval_path)[args.start : args.start + args.limit]
        for item_index, item in enumerate(items, start=args.start):
            if (suite, item_index) in completed:
                continue
            targets = target_token_ids(model, item)
            if not targets:
                continue
            input_ids = model.encode(item["prompt"], max_length=args.max_seq_len)
            position = benchmark_position(suite, int(input_ids.shape[1]), item)
            result = compute_mean_jacobian_splits(
                model,
                item["prompt"],
                family,
                position=position,
                target_layer=args.target_layer,
                methods=args.methods,
                embedding_geometry=embedding_geometry,
                activation_bank=bank,
                prefix_lookback=args.prefix_lookback,
                k_neighbors=args.k_neighbors,
                direct_metric=args.direct_metric,
                energy_threshold=args.energy_threshold,
                rank=args.rank,
                jvp_batch_size=args.dim_batch,
                max_seq_len=args.max_seq_len,
                forward_ad_attention=args.forward_ad_attention,
            )
            record = {
                "suite": suite,
                "item_index": item_index,
                "probe_position": position,
                "ranks": {
                    method: {str(layer): rank for layer, rank in by_layer.items()}
                    for method, by_layer in result.ranks.items()
                },
                "decomposition_relative_error": {
                    method: {
                        str(layer): result.decomposition_error(method, layer)
                        for layer in result.full
                    }
                    for method in result.tangent
                },
                "variants": score_mean_jacobian_split_result(model, result, targets),
            }
            records.append(record)
            completed.add((suite, item_index))
            _write_json(
                output_path,
                {
                    "config": config,
                    "by_suite": {
                        name: summarize_records(
                            [record for record in records if record["suite"] == name]
                        )
                        for name in sorted({record["suite"] for record in records})
                    },
                    "overall": summarize_records(records),
                    "records": records,
                },
            )
            logger.info("Completed mean-Jacobian split %s item %d", suite, item_index)


def _run_local_oracle(args: argparse.Namespace) -> None:
    model = _load_model(args)
    bank_prompts = _load_prompts(args.bank_corpus, args.n_bank) if args.bank_corpus else []
    bank = _load_or_build_bank(args, model, bank_prompts)
    embedding_geometry = _embedding_geometry(args, model)
    output_path = Path(args.output)
    config = {
        "model_name": args.model,
        "layers": args.layers,
        "target_layer": args.target_layer,
        "methods": args.methods,
        "start": args.start,
        "limit": args.limit,
        "prefix_lookback": args.prefix_lookback,
        "k_neighbors": args.k_neighbors,
        "direct_metric": args.direct_metric,
        "energy_threshold": args.energy_threshold,
        "rank": args.rank,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "broadcast": args.broadcast,
    }
    records: list[dict[str, Any]] = []
    if output_path.exists():
        with output_path.open() as handle:
            existing = json.load(handle)
        if existing.get("config") != config:
            raise ValueError("existing local-oracle output has a different configuration")
        records = existing.get("records", [])
    completed = {(record["suite"], record["item_index"]) for record in records}

    for eval_path in args.eval:
        suite = Path(eval_path).stem.removeprefix("lens-eval-")
        items = load_evaluation_items(eval_path)[args.start : args.start + args.limit]
        for item_index, item in enumerate(items, start=args.start):
            if (suite, item_index) in completed:
                continue
            targets = target_token_ids(model, item)
            if not targets:
                continue
            input_ids = model.encode(item["prompt"], max_length=args.max_seq_len)
            position = benchmark_position(suite, int(input_ids.shape[1]), item)
            result = compute_local_projections(
                model,
                item["prompt"],
                args.layers,
                position=position,
                target_layer=args.target_layer,
                methods=args.methods,
                embedding_geometry=embedding_geometry,
                activation_bank=bank,
                prefix_lookback=args.prefix_lookback,
                k_neighbors=args.k_neighbors,
                direct_metric=args.direct_metric,
                energy_threshold=args.energy_threshold,
                rank=args.rank,
                dim_batch=args.dim_batch,
                max_seq_len=args.max_seq_len,
                skip_first=args.skip_first,
                broadcast=args.broadcast,
                forward_ad_attention=args.forward_ad_attention,
            )
            record = {
                "suite": suite,
                "item_index": item_index,
                "probe_position": position,
                "ranks": {
                    method: {str(layer): rank for layer, rank in by_layer.items()}
                    for method, by_layer in result.ranks.items()
                },
                "decomposition_relative_error": {
                    method: {
                        str(layer): result.decomposition_error(method, layer)
                        for layer in result.full
                    }
                    for method in result.tangent
                },
                "variants": score_local_result(model, result, targets),
            }
            records.append(record)
            completed.add((suite, item_index))
            _write_json(
                output_path,
                {
                    "config": config,
                    "by_suite": {
                        name: summarize_records(
                            [record for record in records if record["suite"] == name]
                        )
                        for name in sorted({record["suite"] for record in records})
                    },
                    "overall": summarize_records(records),
                    "records": records,
                },
            )
            logger.info("Completed local oracle %s item %d", suite, item_index)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, type=_layers)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument("--attn-implementation", default=None)


def _add_geometry_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_jacobian_convention: bool = True,
) -> None:
    parser.add_argument("--methods", type=_csv, default=["pushforward", "direct_pca"])
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--prefix-lookback", type=int, default=4)
    parser.add_argument("--k-neighbors", type=int, default=64)
    parser.add_argument("--direct-metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--forward-ad-attention", choices=["math", "default"], default="math")
    if include_jacobian_convention:
        parser.add_argument(
            "--strict-local",
            dest="broadcast",
            action="store_false",
            help="Use d h_L[p] / d h_l[p] instead of the causal-broadcast Jacobian",
        )
        parser.set_defaults(broadcast=True)


def _add_bank_arguments(parser: argparse.ArgumentParser, *, require_corpus: bool) -> None:
    parser.add_argument("--bank-corpus", required=require_corpus, default=None)
    parser.add_argument("--n-bank", type=int, default=400)
    parser.add_argument("--bank-samples-per-prompt", type=int, default=8)
    parser.add_argument("--activation-bank", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tangent-lenses")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Fit E[J], E[J P_T], and E[J P_N]")
    _add_model_arguments(fit)
    _add_geometry_arguments(fit)
    _add_bank_arguments(fit, require_corpus=False)
    fit.add_argument("--corpus", required=True)
    fit.add_argument("--n-train", type=int, default=400)
    fit.add_argument("--samples-per-prompt", type=int, default=4)
    fit.add_argument("--output", required=True)
    fit.set_defaults(run=_run_fit)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a fitted lens family")
    _add_model_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--eval", nargs="+", required=True)
    evaluate.add_argument("--start", type=int, default=0)
    evaluate.add_argument("--limit", type=int, default=30)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(run=_run_evaluate)

    mean_split = subparsers.add_parser(
        "mean-split",
        help="Evaluate state-dependent E[J] P_T(h) and E[J] P_N(h)",
    )
    _add_model_arguments(mean_split)
    _add_geometry_arguments(mean_split, include_jacobian_convention=False)
    _add_bank_arguments(mean_split, require_corpus=False)
    mean_split.add_argument("--checkpoint", required=True)
    mean_split.add_argument("--eval", nargs="+", required=True)
    mean_split.add_argument("--start", type=int, default=0)
    mean_split.add_argument("--limit", type=int, default=30)
    mean_split.add_argument("--output", required=True)
    mean_split.set_defaults(run=_run_mean_split)

    local = subparsers.add_parser("local-oracle", help="Run prompt-local tangent/normal oracles")
    _add_model_arguments(local)
    _add_geometry_arguments(local)
    _add_bank_arguments(local, require_corpus=False)
    local.add_argument("--eval", nargs="+", required=True)
    local.add_argument("--start", type=int, default=0)
    local.add_argument("--limit", type=int, default=30)
    local.add_argument("--output", required=True)
    local.set_defaults(run=_run_local_oracle)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
