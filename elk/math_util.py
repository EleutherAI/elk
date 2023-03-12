from torch import Tensor
from typing import Optional
import math
import random
import torch


@torch.jit.script
def batch_cov(x: Tensor) -> Tensor:
    """Compute a batch of covariance matrices.

    Args:
        x: A tensor of shape [..., n, d].

    Returns:
        A tensor of shape [..., d, d].
    """
    x_ = x - x.mean(dim=-2, keepdim=True)
    return x_.mT @ x_ / x_.shape[-2]


def power_iteration(
    A: Tensor, b0: Optional[Tensor] = None, max_iter: int = 1000, tol: float = 1e-3
) -> tuple[Tensor, Tensor]:
    """
    Power iteration method for computing the dominant eigenpair of a Hermitian matrix.

    Args:
        A: Symmetric matrix whose dominant eigenpair we want to compute.
        b0: Initial vector for the power iteration. If None, a random vector is used.
        max_iter: The maximum number of iterations to perform before giving up.
        tol: The tolerance for the relative error of the dominant eigenvalue.
    Returns:
        largest_eigval: The largest eigenvalue of A
        largest_eigvec: The largest eigenvector of A
    """
    # Initialize a random vector
    if b0 is None:
        b_k = torch.randn(A.shape[0], 1, device=A.device, dtype=A.dtype)
    else:
        b_k = b0

    for k in range(max_iter):
        # Calculate the matrix-by-vector product Ab and normalize
        b_next = torch.nn.functional.normalize(A @ b_k, dim=0)

        # Check if the vector has converged every 25 iterations.
        # We don't check every iteration because we don't want to
        # cause a host-device sync each time.
        if k % 25 == 0 and torch.norm(b_next - b_k) < tol:
            b_k = b_next
            break

        b_k = b_next

    # Calculate the matrix-by-vector product Ab
    Ab = A @ b_k

    # Calculate the eigenvalue
    largest_eigval = b_k.mT @ Ab

    # Calculate the eigenvector
    largest_eigvec = Ab / largest_eigval
    return largest_eigval.squeeze(), largest_eigvec.squeeze()


def stochastic_round_constrained(x: list[float], rng: random.Random) -> list[int]:
    """Stochastic rounding under integer constraints.

    Given a list of floats which sum to an integer, stochastically round each float to
    an integer while maintaining the same sum. The expectation of the rounded values
    should be (nearly?) equal to the original values.

    Inspired by the algorithm described in https://arxiv.org/abs/1501.00014. Instead of
    deterministically allocating the total shortfall to the elements with the largest
    fractional parts, we randomly allocate it to elements with probability proportional
    to their fractional parts.
    """

    total = sum(x)
    assert math.isclose(total, round(total)), "Total must be an integer"

    rounded = [math.floor(x_) for x_ in x]
    fractional_parts = [x_ - r_ for x_, r_ in zip(x, rounded)]

    # Randomly choose where to allocate the shortfall. Note that currently we are
    # allocating the entire shortfall to a single element, but we could allocate
    # fractions of the shortfall to multiple elements. This would be lower variance
    # but requires an RNG that supports weighted sampling without replacement- this
    # exists in NumPy but not in the Python standard library so we'd need to switch
    # this function to use NumPy (and maintain a NumPy RNG).
    if any(fractional_parts):
        (index,) = rng.choices(range(len(x)), weights=fractional_parts)
        rounded[index] += round(sum(fractional_parts))

    return rounded
