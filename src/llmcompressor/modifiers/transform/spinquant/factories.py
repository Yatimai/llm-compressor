"""
Hadamard factory subclass for learnable SpinQuant rotations.

Extends compressed_tensors HadamardFactory to:

1. Accept Linear/Embedding subclasses (e.g. ParametrizedLinear that appears
   when stacking multiple parametrizations). The upstream
   ``apply_transform_weight`` uses strict type equality which rejects
   subclasses, blocking the R1+RTN parametrize stack used during Cayley
   training.
2. Bypass weight sharing for R2: each attention layer needs its own
   ``nn.Parameter`` so that Cayley SGD can optimize them independently.
   ``v_proj`` and ``o_proj`` of the same layer are still tied externally
   via :py:meth:`SpinQuantModifier._tie_r2_per_layer`.

These overrides are workarounds for compressed_tensors limitations that do
not yet support learnable transform stacking. Once upstream relaxes the
type check (issubclass) and exposes a ``tied`` option on TransformScheme,
this subclass can be simplified or removed.
"""

import torch
import torch.nn as nn
from compressed_tensors.transform import TransformArgs
from compressed_tensors.transform.factory.hadamard import (
    HadamardFactory,
    HadamardTransform,
)
from compressed_tensors.transform.utils.matrix import (
    apply_transform_weight,
    get_transform_size,
)
from compressed_tensors.utils import get_execution_device, get_offloaded_device

__all__ = ["LearnableHadamardFactory"]


class _SubclassAwareHadamardTransform(HadamardTransform):
    """
    HadamardTransform that widens its module type to the base
    ``nn.Linear`` / ``nn.Embedding`` so subclasses (e.g. ParametrizedLinear)
    are accepted by ``apply_transform_weight``.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.perm is not None:
            weight = weight[self.perm][:, self.perm]
        if self.args.inverse:
            weight = weight.T
        module_type = self.module_type
        if issubclass(module_type, nn.Linear):
            module_type = nn.Linear
        elif issubclass(module_type, nn.Embedding):
            module_type = nn.Embedding
        return (
            apply_transform_weight(
                weight.to(device=value.device),
                value.to(dtype=weight.dtype),
                self.args.location,
                module_type,
            )
            / self._scale
        ).to(value.dtype)


class LearnableHadamardFactory(HadamardFactory):
    """
    Hadamard factory for learnable SpinQuant rotations (Cayley SGD).

    Differences from the upstream :class:`HadamardFactory`:
    - returns :class:`_SubclassAwareHadamardTransform` so stacked
      parametrizations (R1 + RTN fake-quant) compose correctly
    - bypasses weight sharing for ``R2`` so each attention layer holds its
      own learnable Parameter
    """

    def create_transform(self, module: nn.Module, args: TransformArgs):
        size = get_transform_size(module, args.location, self.scheme.head_dim)
        exec_device = get_execution_device(module)
        device = get_offloaded_device(module)
        precision = self.scheme.precision if args.is_online() else torch.float64

        # for R2: create a fresh weight per module so per-layer optimization
        # is possible. v_proj <-> o_proj tying is handled externally.
        if self.name == "R2":
            weight = self._create_weight(
                size,
                device=device,
                construct_device=exec_device,
                precision=precision,
            )
            return _SubclassAwareHadamardTransform(
                weight,
                None,
                self.scheme,
                args,
                type(module),
            )

        # R1 / R3 / R4: keep the upstream caching (single shared rotation)
        factory_kwargs = {
            "device": device,
            "construct_device": exec_device,
            "precision": precision,
        }
        weight = self.weights.get(size, factory_kwargs=factory_kwargs)
        perm = self.perms[weight] if self.scheme.randomize else None
        return _SubclassAwareHadamardTransform(
            weight,
            perm,
            self.scheme,
            args,
            type(module),
        )
