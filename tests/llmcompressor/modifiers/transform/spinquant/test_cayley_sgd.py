"""
Unit tests for the Cayley SGD optimizer on the Stiefel manifold.

These tests are CPU-only and run in seconds. They verify the algorithmic
correctness of the optimizer in isolation, independent of any model or
quantization setup.
"""

import pytest
import torch

from llmcompressor.modifiers.transform.spinquant.cayley_sgd import (
    CayleySGD,
    _induced_one_norm,
    _stiefel_retraction,
)


def _hadamard_4() -> torch.Tensor:
    """Sylvester Hadamard 4x4 / sqrt(4), i.e. row_norm = 1."""
    H = (
        torch.tensor(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
            dtype=torch.float64,
        )
        / 2.0
    )
    assert torch.allclose(H @ H.T, torch.eye(4, dtype=torch.float64), atol=1e-10)
    return H


def test_step_preserves_orthogonality_on_normalized_input():
    """Cayley step on a row_norm=1 matrix keeps row_norm=1 (~Stiefel manifold)."""
    torch.manual_seed(42)
    R = _hadamard_4().clone().requires_grad_(False)
    R.grad = torch.randn(4, 4, dtype=torch.float64) * 0.1
    opt = CayleySGD([R], lr=1.5)
    opt.step()
    # row norms should still be ~1 (stored matrix is at row_norm=1 here since
    # input was normalized; the rescaling logic preserves the scale)
    rn = R.norm(dim=1)
    assert torch.allclose(rn, torch.ones(4, dtype=torch.float64), atol=1e-3)


def test_step_handles_unnormalized_input():
    """
    compressed_tensors stores raw Hadamard (entries +/-1, row_norm=sqrt(n)).
    Cayley step must internally normalize, do manifold ops, then restore scale.
    """
    torch.manual_seed(42)
    H_raw = _hadamard_4() * 2.0  # row_norm = 2 = sqrt(4)
    R = H_raw.clone().requires_grad_(False)
    R.grad = torch.randn(4, 4, dtype=torch.float64) * 0.1
    opt = CayleySGD([R], lr=1.5)
    opt.step()
    # output row_norm should still be ~2 (scale restored after manifold ops)
    rn = R.norm(dim=1)
    assert torch.allclose(rn, torch.full((4,), 2.0, dtype=torch.float64), atol=2e-2)
    # effective rotation R/sqrt(n) should be ~orthogonal
    n = R.shape[0]
    R_eff = R / (n**0.5)
    err = (R_eff @ R_eff.T - torch.eye(n, dtype=torch.float64)).norm()
    assert err < 1e-2, f"orthogonality drift {err:.2e} too large"


def test_grad_scaling_chain_rule():
    """
    Chain rule for normalization: ``∂L/∂H = ∂L/∂R_eff / row_norm``.
    The optimizer must rescale the gradient by row_norms before manifold ops,
    otherwise the effective step is sqrt(n) times too small.

    We verify by checking that the same gradient applied to scaled vs unscaled
    storage produces equivalent effective rotations after one step.
    """
    torch.manual_seed(0)
    H = _hadamard_4()
    G = torch.randn(4, 4, dtype=torch.float64) * 0.1

    # case A: stored at row_norm=1 (normalized convention)
    R_norm = H.clone().requires_grad_(False)
    R_norm.grad = G.clone()
    CayleySGD([R_norm], lr=1.5).step()

    # case B: stored at row_norm=sqrt(n) (compressed_tensors convention)
    R_raw = (H * 2.0).clone().requires_grad_(False)
    # gradient wrt the raw-stored param is scaled DOWN by 1/sqrt(n) per chain rule
    R_raw.grad = G.clone() / 2.0
    CayleySGD([R_raw], lr=1.5).step()

    # effective rotations must match
    n = 4
    R_norm_eff = R_norm
    R_raw_eff = R_raw / (n**0.5)
    err = (R_norm_eff - R_raw_eff).abs().max()
    assert err < 1e-6, (
        f"effective rotations differ by {err:.2e}; gradient chain-rule "
        f"rescaling is broken"
    )


def test_stiefel_retraction_returns_orthogonal_matrix():
    X = torch.randn(8, 8, dtype=torch.float64)
    Q = _stiefel_retraction(X)
    assert torch.allclose(Q @ Q.T, torch.eye(8, dtype=torch.float64), atol=1e-12)


def test_induced_one_norm_is_max_abs_column_sum():
    M = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # column sums of |M|: [5, 7, 9] -> max = 9
    assert _induced_one_norm(M).item() == pytest.approx(9.0)


def test_invalid_lr_raises():
    R = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="learning rate"):
        CayleySGD([R], lr=0.0)


def test_invalid_momentum_raises():
    R = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="momentum"):
        CayleySGD([R], lr=1.5, momentum=1.5)


def test_only_2d_params_accepted():
    R = torch.zeros(3, dtype=torch.float64)
    R.requires_grad_(False)
    R.grad = torch.zeros(3, dtype=torch.float64)
    opt = CayleySGD([R], lr=1.5)
    with pytest.raises(ValueError, match="2D parameters"):
        opt.step()
