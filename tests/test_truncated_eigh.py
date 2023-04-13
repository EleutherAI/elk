import numpy as np
import pytest
import torch
from scipy.sparse.linalg import eigsh

from elk.truncated_eigh import truncated_eigh


def random_symmetric_matrix(n: int, k: int) -> torch.Tensor:
    """Random symmetric matrix with `k` nonzero eigenvalues centered around zero."""
    assert k <= n, "Rank k should be less than or equal to the matrix size n."

    # Generate random n x k matrix A with elements drawn from a uniform distribution
    A = torch.rand(n, k) / k**0.5

    # Create a diagonal matrix D with k eigenvalues evenly distributed around zero
    eigenvalues = torch.linspace(-1, 1, k)
    D = torch.diag(eigenvalues)

    # Compute the product A * D * A.T to obtain a symmetric matrix with the desired
    # eigenvalue distribution
    symm_matrix = A @ D @ A.T

    return symm_matrix


@pytest.mark.parametrize("n", [32, 768, 6144])
@pytest.mark.parametrize("full_rank", [False, True])
@pytest.mark.parametrize("which", ["LA", "SA"])
def test_truncated_eigh(n: int, full_rank: bool, which):
    torch.manual_seed(42)

    if full_rank:
        A = torch.randn(n, n)
    else:
        # Generate a random symmetric matrix with rank n // 2
        A = random_symmetric_matrix(n, n // 2)

    A = A + A.T

    # Compute the top k eigenpairs using our implementation
    w, v = truncated_eigh(A, k=6, which=which, tol=1e-5)

    # Compute the top k eigenpairs using scipy
    w_scipy, v_scipy = eigsh(A.numpy(), which=which)

    # Check that the eigenvalues match to within the tolerance
    torch.testing.assert_close(w, torch.from_numpy(w_scipy), atol=1e-3, rtol=1e-3)

    # Normalize the sign of the eigenvectors
    for i in range(v.shape[-1]):
        if v[torch.argmax(torch.abs(v[:, i])), i] < 0:
            v[:, i] *= -1
        if v_scipy[np.argmax(np.abs(v_scipy[:, i])), i] < 0:
            v_scipy[:, i] *= -1

    # Check that the eigenvectors match to within the tolerance
    torch.testing.assert_close(v, torch.from_numpy(v_scipy), atol=1e-3, rtol=1e-3)
