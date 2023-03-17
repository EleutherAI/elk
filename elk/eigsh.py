from torch import Tensor
from typing import Literal, Optional
import torch


def lanczos_eigsh(
    A: Tensor,
    k: int = 6,
    *,
    max_iter: Optional[int] = None,
    ncv: Optional[int] = None,
    tol: Optional[float] = None,
    v0: Optional[Tensor] = None,
    which: Literal["LA", "LM", "SA"] = "LM",
) -> tuple[Tensor, Tensor]:
    """Lanczos method for computing the top k eigenpairs of a symmetric matrix.

    Implementation transliterated from `cupyx.scipy.sparse.linalg.eigsh`, which in turn
    is based on `scipy.sparse.linalg.eigsh`.

    Args:
        A (Tensor): The symmetric matrix to compute the eigenpairs of.
        k (int): The number of eigenpairs to compute.
        max_iter (int, optional): The maximum number of iterations to perform.
        ncv (int, optional): The number of Lanczos vectors generated. Must be
            greater than k and smaller than n - 1.
        tol (float, optional): The tolerance for the residual.
        v0 (Tensor, optional): The starting vector for the Lanczos iteration.
        which (str, optional): Which k eigenvalues and eigenvectors to compute.
            Must be one of 'LA', 'LM', or 'SA'.
            'LA': compute the k largest (algebraic) eigenvalues.
            'LM': compute the k largest (in magnitude) eigenvalues.
            'SA': compute the k smallest (algebraic) eigenvalues.

    Returns:
        (Tensor, Tensor): A tuple containing the eigenvalues and eigenvectors.
    """
    n = A.shape[0]

    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)

    if max_iter is None:
        max_iter = 10 * n
    if tol is None:
        tol = torch.finfo(A.dtype).eps

    alpha = A.new_zeros([ncv])
    beta = A.new_zeros([ncv])
    V = A.new_empty([ncv, n])

    # Initialize Lanczos vector
    if v0 is None:
        u = torch.randn(n, dtype=A.dtype, device=A.device)
    else:
        assert v0.shape == (n,)
        u = v0

    V[0] = u / torch.norm(u)

    _lanczos_asis(A, V, u, alpha, beta, 0, ncv)

    # Compute the Ritz vectors and values
    cur_iter = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.mT @ s

    # Compute the residual
    beta_k = beta[-1] * s[-1, :]
    res = beta_k.norm()

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

    idx = w.argsort()
    return w[idx], x[:, idx]


@torch.jit.script
def _eigsh_solve_ritz(alpha, beta, beta_k: Optional[Tensor], k: int, which: str):
    """Solve the standard eigenvalue problem for the Ritz values and vectors."""
    t = torch.diag(alpha)
    t = t + torch.diag(beta[:-1], 1)
    t = t + torch.diag(beta[:-1], -1)
    if beta_k is not None:
        t[k, :k] = beta_k
        t[:k, k] = beta_k

    w, s = torch.linalg.eigh(t)

    # Pick-up k ritz-values and ritz-vectors
    if which == "LA":
        idx = torch.argsort(w)
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == "LM":
        idx = torch.argsort(torch.absolute(w))
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == "SA":
        idx = torch.argsort(w)
        wk = w[idx[:k]]
        sk = s[:, idx[:k]]
    else:
        raise ValueError("which must be LA, LM, or SA")

    return wk, sk


# @torch.jit.script
def _lanczos_asis(a, V, u, alpha, beta, i_start: int, i_end: int):
    """Lanczos iteration for symmetric matrix."""

    assert len(u.shape) == 1
    for i in range(i_start, i_end):
        u[...] = a @ V[i]
        torch.dot(V[i], u, out=alpha[i])
        u -= u @ V[: i + 1].conj().mT @ V[: i + 1]
        torch.norm(u, out=beta[i])
        if i >= i_end - 1:
            break

        V[i + 1] = u / beta[i]


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
    beta[:k] = 0
    alpha[:k] = w
    V[:k] = x.T

    # Compute the next Lanczos vector
    u -= u @ V[:k].conj().mT @ V[:k]

    V[k] = u / u.norm()

    u[:] = A @ V[k]
    torch.dot(V[k], u, out=alpha[k])
    u -= alpha[k] * V[k]
    u -= V[:k].mT @ beta_k
    torch.norm(u, out=beta[k])
    V[k + 1] = u / beta[k]

    # Lanczos iteration
    _lanczos_asis(A, V, u, alpha, beta, k + 1, ncv)

    w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
    x = V.mT @ s

    # Compute the residual
    beta_k = beta[-1] * s[-1, :]
    res = beta_k.norm()

    return w, x, beta_k, res
