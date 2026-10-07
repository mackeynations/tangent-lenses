from __future__ import annotations

import torch
from conftest import TinyModel

from tangent_lenses.jacobians import compute_prompt_derivatives


def test_strict_local_downstream_jacobian_matches_linear_model() -> None:
    model = TinyModel()
    derivatives = compute_prompt_derivatives(
        model,
        "abcdef",
        [1],
        target_layer=2,
        positions=[2],
        dim_batch=2,
        broadcast=False,
    )
    expected = model.layers[2].matrix
    torch.testing.assert_close(derivatives.downstream[1][2], expected)
