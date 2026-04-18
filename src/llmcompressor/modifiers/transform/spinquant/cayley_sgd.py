"""
Cayley SGD optimizer on the Stiefel manifold.

Implements Algorithm 1 from Li et al. 2020 (arXiv:2002.01113) for
Riemannian optimization on the Stiefel manifold via iterative Cayley
transform. Used by SpinQuant (arXiv:2405.16406) to learn
quantization-aware orthogonal rotations.

Adapted for compressed_tensors' row-unnormalized Hadamard storage
convention, which requires Stiefel normalization and gradient chain rule
rescaling inside the optimizer step.
"""

import random
from typing import Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["CayleySGD"]


EPS = 1e-8


def _induced_one_norm(M: torch.Tensor) -> torch.Tensor:
    """Induced 1-norm of a matrix: max absolute column sum."""
    return M.abs().sum(dim=0).max()


def _cayley_iteration(
    X: torch.Tensor,
    skew: torch.Tensor,
    init: torch.Tensor,
    step: float,
    num_iter: int,
) -> torch.Tensor:
    """
    Iterative estimation of the Cayley transform
    Y = (I - step/2 · skew)^-1 (I + step/2 · skew) · X
    via fixed-point iteration (Eq. 5 in Li et al. 2020).
    """
    Y = X + step * init
    half_step = 0.5 * step
    for _ in range(num_iter):
        Y = X + half_step * (skew @ (X + Y))
    return Y


def _stiefel_retraction(X: torch.Tensor) -> torch.Tensor:
    """
    Project X back onto the Stiefel manifold via QR decomposition.

    QR is applied to X.T (transposed-space convention used in the optimizer
    below) and the result transposed back, giving an orthonormal X with the
    sign of the diagonal of R fixed positive.
    """
    Q, R = torch.linalg.qr(X.T)
    sign = R.diagonal().sign()
    return (Q * sign.unsqueeze(0)).T


class CayleySGD(Optimizer):
    """
    SGD with momentum on the Stiefel manifold using an iterative Cayley transform.

    Implements Algorithm 1 from "Efficient Riemannian Optimization on the
    Stiefel Manifold via the Cayley Transform" (Li et al., ICLR 2020, arXiv:2002.01113).

    Each parameter must be a 2D matrix with orthonormal rows (or columns), as in the
    SpinQuant setting where parameters are square rotation matrices initialized
    from a Hadamard matrix.

    :param params: iterable of parameters to optimize. Each parameter must be a 2D
        matrix lying on the Stiefel manifold.
    :param lr: learning rate for the Cayley step. The effective step size is
        `min(lr, 2q / (||skew||_1 + eps))`, where ``skew`` is the
        skew-symmetrized tangent gradient computed at each step.
    :param momentum: heavy-ball momentum coefficient. `0.0` recovers plain Cayley
        gradient descent, as used in the SpinQuant setting.
    :param num_cayley_iter: number of fixed-point iterations for the Cayley transform.
    :param q: contraction parameter used for the adaptive step size. Must be in (0, 1).
    :param eps: numerical epsilon used to guard divisions.
    :param qr_retraction_prob: probability per step of re-orthogonalizing each
        parameter via QR decomposition to counteract floating-point drift.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float,
        momentum: float = 0.0,
        num_cayley_iter: int = 5,
        q: float = 0.5,
        eps: float = EPS,
        qr_retraction_prob: float = 1.0 / 100.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 < q < 1.0:
            raise ValueError(f"Invalid contraction parameter q: {q}")
        if not 0.0 <= qr_retraction_prob <= 1.0:
            raise ValueError(f"Invalid QR retraction probability: {qr_retraction_prob}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            num_cayley_iter=num_cayley_iter,
            q=q,
            eps=eps,
            qr_retraction_prob=qr_retraction_prob,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            num_cayley_iter = group["num_cayley_iter"]
            q = group["q"]
            eps = group["eps"]
            qr_prob = group["qr_retraction_prob"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim != 2:
                    raise ValueError(
                        f"CayleySGD expects 2D parameters, got shape {tuple(p.shape)}"
                    )

                # all manifold operations are done in float64 for numerical stability
                X = p.data.to(torch.float64)

                # compressed_tensors' HadamardFactory stores the unnormalized
                # Hadamard matrix (entries +/-1, row_norm = sqrt(n)) and divides
                # by sqrt(n) inside the transform forward. Normalize rows here
                # so manifold operations see X on the Stiefel manifold
                # (X @ X.T = I); the scale is restored before writing back.
                row_norms = X.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
                X = X / row_norms

                # scale gradient from stored-H space (row_norm=sqrt(n)) to
                # normalized-X space used by manifold ops. Chain rule:
                # since X_eff = H / row_norms, we have
                # ∂L/∂H[i,j] = ∂L/∂X_eff[i,j] / row_norms[i], so
                # ∂L/∂X_eff[i,j] = p.grad[i,j] * row_norms[i]. Without this
                # rescaling the effective step is sqrt(n) times too small
                # (~64x for R1), starving the rotation of useful movement.
                # gradient stored in transposed space throughout the step
                G_t = (p.grad.data.to(torch.float64) * row_norms).T

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(G_t)

                # periodic retraction BEFORE the step so X is exactly
                # orthonormal when the tangent projection is computed
                if random.random() < qr_prob:
                    X = _stiefel_retraction(X)

                momentum_buf = state["momentum_buffer"]

                # Euclidean momentum update in transposed space
                # (see Algorithm 1 line 4 of Li et al. 2020)
                momentum_buf = momentum * momentum_buf - G_t

                # auxiliary quantities for the tangent-space projection
                # (see Algorithm 1 lines 5-6 of Li et al. 2020).
                # Using g := momentum_buf in transposed space, we compute
                #   tangent_grad = g·X - (1/2) X^T·X·g·X
                # then skew-symmetrize it to get the infinitesimal generator.
                grad_X = momentum_buf @ X
                X_grad_X = X @ grad_X
                XtX_grad_X = X.T @ X_grad_X
                tangent_grad = grad_X - 0.5 * XtX_grad_X
                skew_grad = tangent_grad - tangent_grad.T

                # adaptive step size (see Algorithm 1 line 8 of Li et al. 2020)
                skew_norm = _induced_one_norm(skew_grad).item()
                alpha = min(lr, 2.0 * q / (skew_norm + eps))

                # iterative Cayley transform on X.T
                # (see Algorithm 1 lines 9-12 of Li et al. 2020).
                # initial guess uses the raw momentum (not skew_grad @ X.T)
                Y_t = _cayley_iteration(
                    X.T, skew_grad, momentum_buf, alpha, num_cayley_iter
                )
                X_new = Y_t.T

                # restore row scale so the transform forward (which divides by
                # sqrt(n)) still yields an orthogonal effective rotation matrix
                p.data.copy_((X_new * row_norms).to(p.data.dtype))

                # store tangent-projected momentum in transposed space
                state["momentum_buffer"] = skew_grad @ X.T

        return loss
