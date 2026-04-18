"""
Unit tests for :class:`LearnableHadamardFactory`.

Covers the two behaviours that distinguish it from the upstream
:class:`compressed_tensors.transform.factory.hadamard.HadamardFactory`:
- accepts ``nn.Linear`` / ``nn.Embedding`` subclasses (e.g.
  ParametrizedLinear) by widening the module type before
  ``apply_transform_weight``
- creates a fresh ``nn.Parameter`` per module for ``R2`` (so each attention
  layer can hold its own learnable rotation), while keeping the upstream
  per-size cache for ``R1`` / ``R3`` / ``R4``
"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from compressed_tensors.transform import (
    TransformArgs,
    TransformScheme,
)

from llmcompressor.modifiers.transform.spinquant.factories import (
    LearnableHadamardFactory,
    _SubclassAwareHadamardTransform,
)


def _count_unique_param_ids(model: nn.Module) -> int:
    seen: set = set()
    for m in model.modules():
        plists = getattr(m, "parametrizations", None)
        if plists is None:
            continue
        for plist in plists.values():
            for transform in plist:
                w = getattr(transform, "weight", None)
                if w is None:
                    continue
                seen.add(id(w))
    return len(seen)


class _Tiny(nn.Module):
    """4 Linear layers, simulates 4 attention v_proj modules with same shape."""

    def __init__(self, n: int = 4, dim: int = 128):
        super().__init__()
        for i in range(n):
            setattr(self, f"linear{i}", nn.Linear(dim, dim, bias=False))


def _r2_scheme(head_dim: int = 128, requires_grad: bool = True) -> TransformScheme:
    return TransformScheme(
        type="hadamard",
        requires_grad=requires_grad,
        head_dim=head_dim,
        apply=[
            TransformArgs(targets=["re:linear.*"], location="weight_output"),
        ],
    )


def test_r2_creates_one_param_per_module():
    """LearnableHadamardFactory bypasses the per-size cache for R2."""
    model = _Tiny(n=4, dim=128)
    factory = LearnableHadamardFactory(name="R2", scheme=_r2_scheme(128))
    factory.apply_to_model(model, use_tqdm=False)
    assert _count_unique_param_ids(model) == 4, (
        "expected 4 distinct R2 Parameters (one per linear); "
        "the factory must bypass the per-size cache for R2"
    )


def test_r1_keeps_shared_param():
    """For non-R2 names the factory uses the upstream cache (shared Parameter)."""
    model = _Tiny(n=4, dim=128)
    factory = LearnableHadamardFactory(name="R1", scheme=_r2_scheme(128))
    factory.apply_to_model(model, use_tqdm=False)
    assert (
        _count_unique_param_ids(model) == 1
    ), "expected 1 shared R1 Parameter across all linears"


def test_subclass_aware_transform_accepts_parametrized_linear():
    """
    The transform must widen its module type so a ParametrizedLinear (subclass
    of nn.Linear) is accepted by ``apply_transform_weight``. Without this,
    stacking R2 on top of R1 raises NotImplementedError.
    """
    model = _Tiny(n=1, dim=128)
    # apply R1 first so linear0 becomes ParametrizedLinear
    f1 = LearnableHadamardFactory(name="R1", scheme=_r2_scheme(128))
    f1.apply_to_model(model, use_tqdm=False)
    assert parametrize.is_parametrized(model.linear0, "weight")

    # apply R2 on top via _apply_to_module directly (not apply_to_model)
    # to avoid match_named_modules matching parametrize submodules
    args = _r2_scheme(128).apply[0]
    f2 = LearnableHadamardFactory(name="R2", scheme=_r2_scheme(128))
    f2._apply_to_module(model, model.linear0, args)
    plist = model.linear0.parametrizations["weight"]
    assert len(plist) == 2, "expected 2 stacked parametrizations on linear0.weight"


def test_factory_returns_subclass_aware_transform():
    model = _Tiny(n=1, dim=128)
    factory = LearnableHadamardFactory(name="R2", scheme=_r2_scheme(128))
    factory.apply_to_model(model, use_tqdm=False)
    transforms = list(model.linear0.parametrizations["weight"])
    assert all(
        isinstance(t, _SubclassAwareHadamardTransform) for t in transforms
    ), "factory must return _SubclassAwareHadamardTransform instances"


def test_apply_then_load_state_dict_round_trip():
    """
    After applying R2 schemes, the rotation Parameters appear in state_dict
    under stable parametrize keys. Saving and re-loading those keys into a
    fresh model must produce bit-identical Parameters.
    """
    src = _Tiny(n=2, dim=128)
    LearnableHadamardFactory(name="R2", scheme=_r2_scheme(128)).apply_to_model(
        src, use_tqdm=False
    )
    src_state = {
        k: v.clone()
        for k, v in src.state_dict().items()
        if "parametrizations" in k and k.endswith(".weight") and "original" not in k
    }
    assert src_state, "expected R2 weights in state_dict"

    dst = _Tiny(n=2, dim=128)
    LearnableHadamardFactory(name="R2", scheme=_r2_scheme(128)).apply_to_model(
        dst, use_tqdm=False
    )
    result = dst.load_state_dict(src_state, strict=False)
    assert not result.unexpected_keys

    # check values match
    for k, v in src_state.items():
        loaded = dst.state_dict()[k]
        assert torch.equal(loaded, v), f"mismatch for key {k}"
