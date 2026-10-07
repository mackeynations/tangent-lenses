"""Exact downstream Jacobians and low-rank upstream pushforwards."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch.autograd import forward_ad

from tangent_lenses.hooks import ActivationRecorder, ActivationReplacement
from tangent_lenses.model import LensModel

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    SDPBackend = None
    sdpa_kernel = None


def valid_positions(seq_len: int, *, skip_first: int = 16) -> list[int]:
    """Positions used for fitting (attention sinks and final token excluded)."""

    if skip_first < 0:
        raise ValueError("skip_first must be non-negative")
    positions = list(range(skip_first, seq_len - 1))
    if not positions:
        raise ValueError(
            f"sequence of length {seq_len} has no positions after skip_first={skip_first}"
        )
    return positions


def sample_positions(positions: Sequence[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("samples_per_prompt must be positive")
    if len(positions) <= count:
        return list(positions)
    indices = np.linspace(0, len(positions) - 1, count, dtype=int)
    return [int(positions[index]) for index in indices]


@dataclass
class PromptState:
    """Residual states needed to estimate prompt-local tangent projectors."""

    input_ids: torch.Tensor
    token_ids: list[int]
    positions: list[int]
    source_activations: dict[int, dict[int, torch.Tensor]]
    target_activations: dict[int, torch.Tensor]
    layer0_activation: torch.Tensor


@dataclass
class PromptDerivatives(PromptState):
    """Prompt state plus exact downstream matrices at selected positions."""

    valid_positions: list[int]
    downstream: dict[int, dict[int, torch.Tensor]]


def capture_prompt_state(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    positions: Sequence[int] = (-1,),
    max_seq_len: int = 128,
) -> PromptState:
    """Capture only the states needed for ``P_T(h)`` and ``P_N(h)``.

    Unlike :func:`compute_prompt_derivatives`, this performs no reverse-mode
    downstream-Jacobian passes. It is the efficient path for studying
    ``E[J] P_T(h)`` and ``E[J] P_N(h)`` after ``E[J]`` has already been fit.
    """

    layers = sorted(set(int(layer) for layer in source_layers))
    if not layers:
        raise ValueError("at least one source layer is required")
    target = model.n_layers - 1 if target_layer is None else int(target_layer)
    if layers[0] < 0 or layers[-1] >= target or target >= model.n_layers:
        raise ValueError("source layers must satisfy 0 <= source < target < n_layers")

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    selected: list[int] = []
    for raw_position in positions:
        position = int(raw_position) + seq_len if int(raw_position) < 0 else int(raw_position)
        if not 0 <= position < seq_len:
            raise ValueError(
                f"position {raw_position} is out of range for sequence length {seq_len}"
            )
        if position not in selected:
            selected.append(position)
    if not selected:
        raise ValueError("no sample positions selected")

    with (
        torch.no_grad(),
        ActivationRecorder(model.layers, at=[0, *layers, target]) as recorder,
    ):
        model.forward(input_ids)

    return PromptState(
        input_ids=input_ids,
        token_ids=[int(token) for token in input_ids[0].tolist()],
        positions=selected,
        source_activations={
            layer: {
                position: recorder.activations[layer][0, position].detach().float().cpu()
                for position in selected
            }
            for layer in layers
        },
        target_activations={
            position: recorder.activations[target][0, position].detach().float().cpu()
            for position in selected
        },
        layer0_activation=recorder.activations[0][:1].detach(),
    )


def compute_prompt_derivatives(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    positions: Sequence[int] | None = None,
    samples_per_prompt: int = 4,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    broadcast: bool = True,
) -> PromptDerivatives:
    """Compute exact sample-local downstream Jacobian matrices.

    With ``broadcast=True``, ``J[l, p]`` maps the source at ``p`` to the sum of
    all valid final-layer targets, matching the causal-broadcast estimator used
    by the Jacobian lens. With ``broadcast=False``, it is the strict diagonal
    block ``d h_target[p] / d h_l[p]``.
    """

    layers = sorted(set(int(layer) for layer in source_layers))
    if not layers:
        raise ValueError("at least one source layer is required")
    target = model.n_layers - 1 if target_layer is None else int(target_layer)
    if layers[0] < 0 or layers[-1] >= target or target >= model.n_layers:
        raise ValueError("source layers must satisfy 0 <= source < target < n_layers")
    if dim_batch <= 0:
        raise ValueError("dim_batch must be positive")

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    try:
        fitting_positions = valid_positions(seq_len, skip_first=skip_first)
    except ValueError:
        if positions is None:
            raise
        fitting_positions = []

    if positions is None:
        selected = sample_positions(fitting_positions, samples_per_prompt)
    else:
        selected = []
        for raw_position in positions:
            position = int(raw_position) + seq_len if int(raw_position) < 0 else int(raw_position)
            if not 0 <= position < seq_len:
                raise ValueError(
                    f"position {raw_position} is out of range for sequence length {seq_len}"
                )
            if position not in selected:
                selected.append(position)
        fitting_positions = sorted(set(fitting_positions) | set(selected))
    if not selected:
        raise ValueError("no sample positions selected")

    d_model = model.d_model
    downstream = {
        layer: {position: torch.zeros(d_model, d_model) for position in selected}
        for layer in layers
    }
    replicated_ids = input_ids.expand(dim_batch, -1)
    batch_indices = torch.arange(dim_batch, device=model.input_device)

    with (
        ActivationRecorder(
            model.layers,
            at=[0, *layers, target],
            start_graph_at=min(layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(replicated_ids)
        target_activation = recorder.activations[target]
        source_tensors = [recorder.activations[layer] for layer in layers]
        cotangent = torch.zeros_like(target_activation)

        gradient_jobs = (
            [(selected, fitting_positions)]
            if broadcast
            else [([position], [position]) for position in selected]
        )
        for source_positions, target_positions in gradient_jobs:
            target_position_tensor = torch.tensor(target_positions, device=target_activation.device)
            for dim_start in range(0, d_model, dim_batch):
                n_dims = min(dim_batch, d_model - dim_start)
                active_batches = batch_indices[:n_dims]
                cotangent.zero_()
                cotangent[
                    active_batches[:, None],
                    target_position_tensor[None, :],
                    dim_start + active_batches[:, None],
                ] = 1.0
                gradients = torch.autograd.grad(
                    target_activation,
                    source_tensors,
                    grad_outputs=cotangent,
                    retain_graph=True,
                )
                for layer, gradient in zip(layers, gradients, strict=True):
                    for source_position in source_positions:
                        downstream[layer][source_position][dim_start : dim_start + n_dims] = (
                            gradient[:n_dims, source_position, :].detach().float().cpu()
                        )

        source_activations = {
            layer: {
                position: recorder.activations[layer][0, position].detach().float().cpu()
                for position in selected
            }
            for layer in layers
        }
        target_activations = {
            position: recorder.activations[target][0, position].detach().float().cpu()
            for position in selected
        }
        layer0_activation = recorder.activations[0][:1].detach()

    return PromptDerivatives(
        input_ids=input_ids,
        token_ids=[int(token) for token in input_ids[0].tolist()],
        positions=selected,
        source_activations=source_activations,
        target_activations=target_activations,
        layer0_activation=layer0_activation,
        valid_positions=fitting_positions,
        downstream=downstream,
    )


def _attention_context(policy: str):
    if policy == "default":
        return nullcontext()
    if policy != "math":
        raise ValueError("forward_ad_attention must be 'math' or 'default'")
    if sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    return sdpa_kernel([SDPBackend.MATH])


def compute_pushforward_factors(
    model: LensModel,
    derivatives: PromptState,
    source_layers: Sequence[int],
    prefix_factors: dict[int, dict[int, torch.Tensor]],
    *,
    jvp_batch_size: int = 8,
    forward_ad_attention: str = "math",
) -> dict[int, dict[int, torch.Tensor]]:
    """Push low-rank layer-0 directions into requested hidden layers.

    ``prefix_factors[p][q]`` is a ``[d_model, r]`` tangent factor injected at
    prefix position ``q`` and observed at target position ``p``. The returned
    factors concatenate every pushed-forward prefix direction.
    """

    layers = sorted(set(int(layer) for layer in source_layers))
    if not layers or layers[0] <= 0:
        raise ValueError("pushforward tangent estimation requires source layers above layer 0")
    if jvp_batch_size <= 0:
        raise ValueError("jvp_batch_size must be positive")

    specs: list[tuple[int, int, torch.Tensor]] = []
    widths: dict[int, int] = {position: 0 for position in prefix_factors}
    for position, by_prefix in prefix_factors.items():
        for prefix_position, factor in by_prefix.items():
            if factor.ndim != 2 or factor.shape[0] != model.d_model:
                raise ValueError("every prefix factor must have shape [d_model, rank]")
            for column in range(factor.shape[1]):
                specs.append((position, prefix_position, factor[:, column]))
                widths[position] += 1

    outputs = {
        position: {
            layer: torch.zeros(model.d_model, width, dtype=torch.float32) for layer in layers
        }
        for position, width in widths.items()
    }
    offsets = {position: 0 for position in widths}
    device = derivatives.layer0_activation.device
    dtype = derivatives.layer0_activation.dtype

    for start in range(0, len(specs), jvp_batch_size):
        chunk = specs[start : start + jvp_batch_size]
        batch_size = len(chunk)
        primal = derivatives.layer0_activation.expand(batch_size, -1, -1).clone()
        tangent = torch.zeros_like(primal)
        positions = torch.tensor([spec[0] for spec in chunk], device=device)
        batch_indices = torch.arange(batch_size, device=device)
        chunk_destinations: list[tuple[int, int]] = []
        for batch_index, (position, prefix_position, direction) in enumerate(chunk):
            tangent[batch_index, prefix_position] = direction.to(device=device, dtype=dtype)
            chunk_destinations.append((position, offsets[position]))
            offsets[position] += 1

        with forward_ad.dual_level():
            dual_replacement = forward_ad.make_dual(primal, tangent)
            with (
                ActivationReplacement(model.layers, 0, dual_replacement),
                _attention_context(forward_ad_attention),
                ActivationRecorder(model.layers, at=layers) as recorder,
            ):
                model.forward(derivatives.input_ids.expand(batch_size, -1))
                stacked = torch.stack(
                    [recorder.activations[layer][batch_indices, positions] for layer in layers],
                    dim=1,
                )
            _, tangent_output = forward_ad.unpack_dual(stacked)
        if tangent_output is None:
            raise RuntimeError("forward-mode AD returned no tangent")
        tangent_output = tangent_output.detach().float().cpu()
        for batch_index, (position, destination) in enumerate(chunk_destinations):
            for layer_index, layer in enumerate(layers):
                outputs[position][layer][:, destination] = tangent_output[batch_index, layer_index]

    return outputs
