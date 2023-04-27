from typing import Literal, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class ConvergenceError(Exception):
    """Raised when the Lanczos iteration fails to converge."""


class Eigendecomposition(NamedTuple):
    """A namedtuple containing eigenpairs of a matrix."""

    eigenvalues: Tensor
    eigenvectors: Tensor


def truncated_eigh(
    A: Tensor,
    k: int = 1,
    *,
    max_iter: Optional[int] = None,
    ncv: Optional[int] = None,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    which: Literal["LA", "SA"] = "LA",
    verbose: bool = False,
) -> Eigendecomposition:
    """Compute the leading `k` eigenpairs of `A` with the thick-restart Lanczos method.

    Algorithm proposed by Wu & Simon (1998) https://www.osti.gov/servlets/purl/895499.
    For matrices 256 x 256 or smaller, we short-circuit to the naive method of calling
    `torch.linalg.eigh` and discarding all but the requested number of eigenpairs.
    Empirically this is faster than our Lanczos implementation for such small matrices.

    Args:
        A (Tensor): The matrix or batch of matrices of shape `[..., n, n]` for which to
            compute eigenpairs. Must be symmetric, but need not be positive definite.
        k (int): The number of eigenpairs to compute.
        max_iter (int, optional): The maximum number of iterations to perform.
        ncv (int, optional): The number of Lanczos vectors generated. Must be
            greater than k and smaller than n - 1.
        tol (float, optional): The tolerance for the residual. Defaults to the machine
            precision of `A.dtype` or 1e-4, whichever is larger.
        seed (int, optional): The random seed to use for the starting vector.
        which (str, optional): Which k eigenvalues and eigenvectors to compute.
            Must be one of 'LA', or 'SA'.
            'LA': compute the k largest (algebraic) eigenvalues.
            'SA': compute the k smallest (algebraic) eigenvalues.
        verbose (bool, optional): Whether to print progress information.

    Returns:
        (Tensor, Tensor): A tuple containing the eigenvalues and eigenvectors.

    Raises:
        ConvergenceError: If the Lanczos iteration fails to converge.
    """
    *leading, n, m = A.shape
    assert n == m, "A must be a square matrix or a batch of square matrices."

    # Short circuit if the matrix is too small or if we're asked for too many
    # eigenpairs; we can't outcompete the naive method.
    if k > 10 or n <= 256:
        L, Q = torch.linalg.eigh(A)
        if which == "LA":
            return Eigendecomposition(L[..., -k:], Q[..., :, -k:])
        elif which == "SA":
            return Eigendecomposition(L[..., :k], Q[..., :, :k])

    if ncv is None:
        # This is the default used by SciPy; CuPy uses min(n - 1, max(2 * k, k + 32)).
        # Empirically the SciPy default seems to converge better.
        ncv = min(n, max(2 * k + 1, 20))
    else:
        ncv = min(max(ncv, k + 2), n - 1)

    if max_iter is None:
        max_iter = 10 * n

    # Diagonal and off-diagonal elements of the tridiagonal matrix
    alpha = A.new_zeros([*leading, ncv])
    beta = A.new_zeros([*leading, ncv])

    # Lanczos vector basis for the Krylov subspace
    Q = A.new_empty([*leading, ncv, n])

    # Initialize Lanczos vector
    rng = torch.Generator(A.device)
    if seed is not None:
        rng.manual_seed(seed)

    r_k = torch.randn(*leading, n, dtype=A.dtype, device=A.device, generator=rng)

    Q[..., 0, :] = F.normalize(r_k, dim=-1)
    _lanczos_inner_loop(A, Q, r_k, alpha, beta, 0, ncv)

    # Compute the Ritz vectors and values
    cur_iter = ncv
    w, s = _solve_ritz_pairs(alpha, beta, None, k, which)
    x = torch.einsum("...ij,...ik->...jk", Q, s)

    # Compute the residual. Note that we take the max over the batch dimensions,
    # to ensure that we don't terminate early for any element in the batch.
    beta_k = beta[..., -1, None] * s[..., -1, :]
    first_res = res = beta_k.norm(dim=-1).max()

    # Keep restarting until we converge or hit the iteration limit
    while res > tol and cur_iter < max_iter:
        # Setup for thick-restart
        alpha[..., :k] = w
        beta[..., :k] = 0
        Q[..., :k, :] = x.mT

        # Compute the next Lanczos vector
        _gram_schmidt(r_k, Q[..., :k, :])
        Q[..., k, :] = F.normalize(r_k, dim=-1)

        r_k[:] = torch.einsum("...ij,...j->...i", A, Q[..., k, :])
        alpha[..., k] = torch.einsum("...i,...i->...", Q[..., k, :], r_k)
        _gram_schmidt(r_k, Q[..., : k + 1, :])

        beta[..., k] = r_k.square().sum(dim=-1).sqrt()  # TorchScript-friendly norm
        Q[..., k + 1, :] = r_k / beta[..., k, None]

        # Inner loop
        _lanczos_inner_loop(A, Q, r_k, alpha, beta, k + 1, ncv)

        w, s = _solve_ritz_pairs(alpha, beta, beta_k, k, which)
        x = Q.mT @ s

        # Compute the residual
        beta_k = beta[..., -1, None] * s[..., -1, :]
        new_res = beta_k.square().sum(dim=-1).sqrt().max()  # TorchScript-friendly norm
        cur_iter += ncv - k

        # Check for divergence. This may happen in edge cases where our _gram_schmidt
        # is not run for a sufficient number of iterations. Most implementations would
        # use a dynamic number of iterations, but we don't have a good way to do that
        # in TorchScript, so we use a fixed number (2).
        if new_res > 2 * first_res:
            break
        else:
            res = new_res

        if verbose:
            print(f"Residual: {res} after {cur_iter} iterations.")

    if res > tol:
        raise ConvergenceError(
            f"Failed to converge after {cur_iter} iterations. "
            f"Residual: {res}, initial residual: {first_res}."
        )

    # We use the torch.autocast decorator above to speed up the algorithm, but
    # make sure the returned values are in the same dtype as the input.
    return Eigendecomposition(w.type_as(A), x.type_as(A))


@torch.jit.script
def _solve_ritz_pairs(diag, off_diag, beta_k: Optional[Tensor], k: int, which: str):
    """Solve the standard eigenvalue problem for the Ritz values and vectors.

    Args:
        diag (Tensor): The diagonal elements of the tridiagonal matrix.
        off_diag (Tensor): The off-diagonal elements of the tridiagonal matrix.
        beta_k (Tensor, optional): ???
        k (int): The number of eigenpairs to compute.
        which (str): Which k eigenvalues and eigenvectors to compute.
            Must be one of 'LA', or 'SA'.
    """
    # Create tri-diagonal matrix
    t = diag.diag_embed()
    t += off_diag[..., :-1].diag_embed(1)
    t += off_diag[..., :-1].diag_embed(-1)

    if beta_k is not None:
        t[..., k, :k] = beta_k
        t[..., :k, k] = beta_k

    # The eigenpairs are already sorted ascending by algebraic value
    w, s = torch.linalg.eigh(t)

    # Pick-up k ritz-values and ritz-vectors
    if which == "LA":
        wk = w[..., -k:]
        sk = s[..., -k:]
    elif which == "SA":
        wk = w[..., :k]
        sk = s[..., :k]
    else:
        raise ValueError("`which` must be LA or SA")

    return wk, sk


@torch.jit.script
def _gram_schmidt(z: Tensor, Q: Tensor, num_iter: int = 2):
    """Iteratively make vector `z` orthogonal to the semi-orthogonal basis `Q`.

    Wu & Simon (1998) define "semi-orthogonal" to mean that the largest off-diagonal
    element of the Gram matrix is no greater than sqrt(machine epsilon). They prove
    that if `Q` is semi-orthogonal, then applying Gram-Schmidt to `z` will make it
    closer to being orthogonal to the span of `Q`. Multiple iterations of this process
    may be necessary to achieve orthogonality up to machine precision. See pages 12-14
    of Wu & Simon (1998) for details.
    """
    for _ in range(num_iter):
        proj = torch.einsum("...ij,...j->...i", Q.conj(), z)
        z -= torch.einsum("...ij,...i->...j", Q, proj)


@torch.jit.script
def _lanczos_inner_loop(A, krylov, q, alpha, beta, k: int, end: int):
    """Step 2 of Algorithm 3 in Wu & Simon (1998)."""

    for i in range(k, end):
        # Compute the next matrix-vector product Au
        q[:] = torch.einsum("...ij,...j->...i", A, krylov[..., i, :])
        alpha[..., i] = torch.einsum("...i,...i->...", krylov[..., i, :], q)

        # Project away from the current Krylov subspace
        _gram_schmidt(q, krylov[..., : i + 1, :])

        # Record how much is left after projection
        beta[..., i] = q.square().sum(dim=-1).sqrt()  # TorchScript-friendly norm
        if i < end - 1:
            krylov[..., i + 1, :] = q / beta[..., i, None]
