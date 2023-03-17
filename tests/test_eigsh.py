from elk.eigsh import lanczos_eigsh
from scipy.sparse.linalg import eigsh
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("which", ["LA", "LM", "SA"])
def test_lanczos_eigsh(which):
    torch.manual_seed(42)

    # Generate a random symmetric matrix
    n = 10
    A = torch.randn(n, n)
    A = A + A.T

    # Compute the top k eigenpairs using our implementation
    k = 3
    w, v = lanczos_eigsh(A, k=k, which=which)

    # Compute the top k eigenpairs using scipy
    w_scipy, v_scipy = eigsh(A.numpy(), k=k, which=which)

    # Check that the eigenvalues match to within the tolerance
    assert np.allclose(w, w_scipy, rtol=1e-3)

    # Normalize the sign of the eigenvectors
    for i in range(k):
        if v[torch.argmax(torch.abs(v[:, i])), i] < 0:
            v[:, i] *= -1
        if v_scipy[np.argmax(np.abs(v_scipy[:, i])), i] < 0:
            v_scipy[:, i] *= -1

    # Check that the eigenvectors match to within the tolerance
    assert np.allclose(v.numpy(), v_scipy, rtol=1e-3)
