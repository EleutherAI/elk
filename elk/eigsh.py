from torch import Tensor
from typing import Literal, Optional
import torch
import torch.nn.functional as F


def lanczos_eigsh(
    A: Tensor,
    k: int = 6,
    *,
    max_iter: Optional[int] = None,
    ncv: Optional[int] = None,
    tol: Optional[float] = None,
    seed: Optional[int] = None,
    v0: Optional[Tensor] = None,
    which: Literal["LA", "LM", "SA"] = "LA",
) -> tuple[Tensor, Tensor]:
    """Lanczos method for computing the top k eigenpairs of a symmetric matrix.

    Implementation adapted from `cupyx.scipy.sparse.linalg.eigsh`, which in turn is
    based on `scipy.sparse.linalg.eigsh`. Unlike the CuPy and SciPy functions, this
    function supports batched inputs with arbitrary leading dimensions.

    Unlike the above implementations, we use which='LA' as the default instead of
    which='LM' because we are interested in algebraic eigenvalues, not magnitude.
    Largest magnitude is also harder to implement in TorchScript.

    Args:
        A (Tensor): The matrix or batch of matrices of shape `[..., n, n]` for which to
            compute eigenpairs. Must be symmetric, but need not be positive definite.
        k (int): The number of eigenpairs to compute.
        max_iter (int, optional): The maximum number of iterations to perform.
        ncv (int, optional): The number of Lanczos vectors generated. Must be
            greater than k and smaller than n - 1.
        tol (float, optional): The tolerance for the residual.
        seed (int, optional): The random seed to use for the starting vector.
        v0 (Tensor, optional): The starting vector of shape `[n]`.
        which (str, optional): Which k eigenvalues and eigenvectors to compute.
            Must be one of 'LA', 'LM', or 'SA'.
            'LA': compute the k largest (algebraic) eigenvalues.
            'LM': compute the k largest (in magnitude) eigenvalues.
            'SA': compute the k smallest (algebraic) eigenvalues.

    Returns:
        (Tensor, Tensor): A tuple containing the eigenvalues and eigenvectors.
    """
    *leading, n, m = A.shape
    assert n == m, "A must be a square matrix or a batch of square matrices."

    # Short circuit if the matrix is too small; we can't outcompete the naive method.
    if n <= 32:
        L, Q = torch.linalg.eigh(A)
        if which == "LA":
            return L[..., -k:], Q[..., :, -k:]
        elif which == "LM":
            # Resort the eigenvalues and eigenvectors.
            idx = L.abs().argsort(dim=-1, descending=True)
            L = L.gather(-1, idx)
            Q = Q.gather(-1, idx.unsqueeze(-1).expand(*idx.shape, n))
            return L[..., :k], Q[..., :, :k]
        elif which == "SA":
            return L[..., :k], Q[..., :, :k]

    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)

    if max_iter is None:
        max_iter = 10 * n
    if tol is None:
        tol = torch.finfo(A.dtype).eps

    # We don't support which == 'LM' for batched inputs, because we can't do the
    # re-sorting properly in TorchScript.
    if len(leading) > 0 and which == "LM":
        raise NotImplementedError("`which='LM'` is not supported for batched inputs.")

    alpha = A.new_zeros([*leading, ncv])
    beta = A.new_zeros([*leading, ncv])
    V = A.new_empty([*leading, ncv, n])

    # Initialize Lanczos vector
    if v0 is None:
        if seed is not None:
            torch.manual_seed(seed)

        u = torch.randn(*leading, n, dtype=A.dtype, device=A.device)
    else:
        assert v0.shape == (
            *leading,
            n,
        )
        u = v0

    V[..., 0, :] = F.normalize(u, dim=-1)
    _inner_loop(A, V, u, alpha, beta, 0, ncv)

    # Compute the Ritz vectors and values
    cur_iter = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = torch.einsum("...ij,...ik->...jk", V, s)

    # Compute the residual. Note that we take the max over the batch dimensions,
    # to ensure that we don't terminate early for any element in the batch.
    beta_k = beta[..., -1, None] * s[..., -1, :]
    res = beta_k.norm(dim=-1).max()

    while res > tol and cur_iter < max_iter:
        w, x, beta_k, res = _outer_loop(
            A,
            V,
            u,
            alpha,
            beta,
            w,
            x,
            beta_k,
            res,
            which,
            k,
            ncv,
        )
        cur_iter += ncv - k

    return w, x


@torch.jit.script
def _eigsh_solve_ritz(alpha, beta, beta_k: Optional[Tensor], k: int, which: str):
    """Solve the standard eigenvalue problem for the Ritz values and vectors."""

    # Create tri-diagonal matrix
    t = alpha.diag_embed()
    t = t + beta[..., :-1].diag_embed(1)
    t = t + beta[..., :-1].diag_embed(-1)

    if beta_k is not None:
        t[..., k, :k] = beta_k
        t[..., :k, k] = beta_k

    # The eigenpairs are already sorted ascending by algebraic value
    w, s = torch.linalg.eigh(t)

    # Pick-up k ritz-values and ritz-vectors
    if which == "LA":
        wk = w[..., -k:]
        sk = s[..., -k:]
    elif which == "LM":
        # NOTE: We're assuming here that the inputs are not batched; this was already
        # checked in the main function.
        idx = w.abs().argsort()

        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == "SA":
        wk = w[..., :k]
        sk = s[..., :k]
    else:
        raise ValueError("which must be LA, LM, or SA")

    return wk, sk


@torch.jit.script
def _inner_loop(A, V, u, alpha, beta, i_start: int, i_end: int):
    """Lanczos iteration for symmetric matrix."""

    for i in range(i_start, i_end):
        # Batched matrix-vector product Av
        u[:] = torch.einsum("...ij,...j->...i", A, V[..., i, :])

        alpha[..., i] = torch.einsum("...i,...i->...", V[..., i, :], u)
        proj = torch.einsum("...ij,...j->...i", V[..., : i + 1, :].conj(), u)
        u -= torch.einsum("...ij,...i->...j", V[..., : i + 1, :], proj)
        beta[..., i] = u.square().sum(dim=-1).sqrt()  # TorchScript-friendly norm
        if i >= i_end - 1:
            break

        V[..., i + 1, :] = u / beta[..., i, None]


@torch.jit.script
def _outer_loop(
    A,
    V,
    u,
    alpha,
    beta,
    w,
    x,
    beta_k,
    res,
    which: str,
    k: int,
    ncv: int,
):
    """Outer loop for Lanczos iteration."""
    # Setup for thick-restart
    beta[..., :k] = 0
    alpha[..., :k] = w
    V[..., :k, :] = x.mT

    # Compute the next Lanczos vector
    proj = torch.einsum("...ij,...j->...i", V[..., :k, :].conj(), u)
    u -= torch.einsum("...ij,...i->...j", V[..., :k, :], proj)

    V[..., k, :] = F.normalize(u, dim=-1)

    u[:] = torch.einsum("...ij,...j->...i", A, V[..., k, :])
    alpha[..., k] = torch.einsum("...i,...i->...", V[..., k, :], u)

    proj = torch.einsum("...ij,...j->...i", V[..., : k + 1, :].conj(), u)
    u -= torch.einsum("...ij,...i->...j", V[..., : k + 1, :], proj)

    beta[..., k] = u.square().sum(dim=-1).sqrt()  # TorchScript-friendly norm
    V[..., k + 1, :] = u / beta[..., k, None]

    # Inner loop
    _inner_loop(A, V, u, alpha, beta, k + 1, ncv)

    w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
    x = V.mT @ s

    # Compute the residual
    beta_k = beta[..., -1, None] * s[..., -1, :]
    res = beta_k.square().sum(dim=-1).sqrt().max()  # TorchScript-friendly norm

    return w, x, beta_k, res
