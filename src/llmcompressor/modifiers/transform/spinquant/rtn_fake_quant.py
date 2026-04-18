"""
RTN fake-quantization parametrization with straight-through estimator.

Used during Cayley rotation training to expose quantization loss to the
rotation parameters. Stacked on top of the rotation parametrization so the
effective weight seen by the forward pass is Q(R @ W_raw), with a straight-
through estimator preserving the gradient path to the rotation.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

__all__ = [
    "FakeQuantRTN",
    "add_rtn_fake_quant",
    "remove_rtn_fake_quant",
]


class FakeQuantRTN(nn.Module):
    """
    Per-group symmetric integer fake-quantization with a straight-through
    estimator on the backward pass.

    Intended to be registered via `torch.nn.utils.parametrize` so it composes
    on top of any existing parametrization (e.g., a rotation transform).

    :param num_bits: number of integer bits for the fake-quantized weight.
    :param group_size: number of input channels per scale group.
    """

    def __init__(
        self,
        num_bits: int = 4,
        group_size: int = 128,
        mse: bool = True,
        mse_grid_steps: int = 80,
        mse_min_ratio: float = 0.2,
        mse_norm: float = 2.4,
    ):
        super().__init__()
        if num_bits < 2:
            raise ValueError(f"num_bits must be >= 2, got {num_bits}")
        self.num_bits = num_bits
        self.group_size = group_size
        self.mse = mse
        self.mse_grid_steps = mse_grid_steps
        self.mse_min_ratio = mse_min_ratio
        self.mse_norm = mse_norm
        # symmetric int range: e.g. num_bits=4 -> max_int = 7, levels [-7, 7]
        self.max_int = 2 ** (num_bits - 1) - 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        out_features, in_features = w.shape
        if in_features % self.group_size != 0:
            raise ValueError(
                f"in_features {in_features} not divisible by group_size "
                f"{self.group_size}"
            )

        w_grouped = w.reshape(
            out_features, in_features // self.group_size, self.group_size
        )
        max_abs = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)

        if self.mse:
            # per-group grid search over clip ratios in [mse_min_ratio, 1.0];
            # pick the scale minimizing sum |w_q - w|^mse_norm. The 80
            # candidate quants are pure value-finding and stay outside the
            # autograd graph; the STE at the end of forward handles the
            # backward pass.
            with torch.no_grad():
                best_scale = max_abs / self.max_int
                best_err = torch.full_like(max_abs, float("inf"))
                for i in range(self.mse_grid_steps):
                    ratio = 1.0 - (i / self.mse_grid_steps) * (1.0 - self.mse_min_ratio)
                    scale = (max_abs * ratio / self.max_int).clamp(min=1e-10)
                    w_q_try = (w_grouped / scale).round().clamp(
                        -self.max_int, self.max_int
                    ) * scale
                    err = (
                        (w_q_try - w_grouped)
                        .abs()
                        .pow(self.mse_norm)
                        .sum(dim=-1, keepdim=True)
                    )
                    improved = err < best_err
                    best_err = torch.where(improved, err, best_err)
                    best_scale = torch.where(improved, scale, best_scale)
            scale = best_scale
        else:
            scale = max_abs / self.max_int

        w_q = (w_grouped / scale).round().clamp(-self.max_int, self.max_int) * scale
        w_q = w_q.reshape(out_features, in_features)

        # straight-through estimator: forward uses w_q, backward uses identity
        return w + (w_q - w).detach()

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # no true inverse for quantization; return value unchanged so that
        # assignments to the parametrized weight pass through unaffected
        return value


def add_rtn_fake_quant(
    model: nn.Module,
    num_bits: int = 4,
    group_size: int = 128,
    ignore: List[str] = ("lm_head",),
) -> List[nn.Linear]:
    """
    Register a :class:`FakeQuantRTN` parametrization on every `nn.Linear` weight
    in `model`, stacking on top of any existing parametrization.

    :return: the list of Linear modules that received the fake-quant
        parametrization.
    """
    patched: List[nn.Linear] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(skip) for skip in ignore):
            continue
        fq = FakeQuantRTN(num_bits=num_bits, group_size=group_size)
        parametrize.register_parametrization(module, "weight", fq)
        patched.append(module)
    return patched


def remove_rtn_fake_quant(patched: List[nn.Linear]) -> None:
    """
    Remove the last parametrization (assumed to be :class:`FakeQuantRTN`) from
    each module in `patched`, preserving any rotation parametrization underneath.
    """
    for module in patched:
        pl = module.parametrizations["weight"]
        # internal API - no public method to remove a single parametrization
        pl.pop(len(pl) - 1)
        if len(pl) == 0:
            # no more parametrizations - fully restore the raw parameter
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=False
            )
