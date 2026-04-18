"""
Unit tests for :class:`FakeQuantRTN` (per-group symmetric W4 with STE).

Covers:
- w_clip MSE grid search reduces quantization error vs raw min/max
- straight-through estimator passes identity gradient
- input shape validation
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from llmcompressor.modifiers.transform.spinquant.rtn_fake_quant import (
    FakeQuantRTN,
    add_rtn_fake_quant,
    remove_rtn_fake_quant,
)


def test_w_clip_mse_beats_raw_minmax_on_outlier_weight():
    """w_clip MSE grid search should reduce quant error on weights with outliers."""
    torch.manual_seed(0)
    # Llama-like weight: ~N(0, 0.1) bulk plus a few outliers
    w = torch.randn(256, 256) * 0.1
    w[0, :8] = 5.0
    w[:, 0] = -3.0

    raw = FakeQuantRTN(num_bits=4, group_size=128, mse=False)
    mse = FakeQuantRTN(num_bits=4, group_size=128, mse=True)

    err_raw = (raw(w) - w).pow(2).mean().item()
    err_mse = (mse(w) - w).pow(2).mean().item()
    assert err_mse < err_raw, (
        f"w_clip MSE ({err_mse:.4e}) must beat raw min/max ({err_raw:.4e}) "
        f"on a weight with outliers"
    )


def test_ste_passes_identity_gradient():
    """STE: backward through FakeQuantRTN must be identity wrt input weight."""
    fq = FakeQuantRTN(num_bits=4, group_size=128, mse=False)
    w = torch.randn(128, 256, dtype=torch.float32, requires_grad=True)
    out = fq(w)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    assert torch.allclose(
        w.grad, upstream, atol=0.0
    ), "STE gradient should be identity (backward = upstream)"


def test_in_features_must_be_divisible_by_group_size():
    fq = FakeQuantRTN(num_bits=4, group_size=128)
    w_bad = torch.randn(64, 100)  # 100 not divisible by 128
    with pytest.raises(ValueError, match="not divisible by group_size"):
        fq(w_bad)


def test_invalid_num_bits_raises():
    with pytest.raises(ValueError, match="num_bits"):
        FakeQuantRTN(num_bits=1)


def test_add_and_remove_rtn_fake_quant():
    """add_rtn_fake_quant attaches parametrize on every Linear; remove restores."""
    model = nn.Sequential(
        nn.Linear(128, 128, bias=False),
        nn.Linear(128, 128, bias=False),
    )
    patched = add_rtn_fake_quant(model, num_bits=4, group_size=128, ignore=())
    assert len(patched) == 2
    for m in patched:
        assert parametrize.is_parametrized(m, "weight")
    remove_rtn_fake_quant(patched)
    for m in patched:
        assert not parametrize.is_parametrized(m, "weight")


def test_w_clip_default_is_true_via_add_rtn_fake_quant():
    """add_rtn_fake_quant() should activate w_clip MSE by default."""
    model = nn.Sequential(nn.Linear(128, 128, bias=False))
    patched = add_rtn_fake_quant(model, num_bits=4, group_size=128, ignore=())
    fq = patched[0].parametrizations["weight"][0]
    assert isinstance(fq, FakeQuantRTN)
    assert fq.mse is True
    assert fq.mse_grid_steps == 80
    assert fq.mse_min_ratio == pytest.approx(0.2)
    assert fq.mse_norm == pytest.approx(2.4)
    remove_rtn_fake_quant(patched)
