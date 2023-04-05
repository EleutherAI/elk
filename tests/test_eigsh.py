import numpy as np
import pytest
import torch
from scipy.sparse.linalg import eigsh

from elk.eigsh import lanczos_eigsh


@pytest.mark.parametrize("n", [20, 40])
@pytest.mark.parametrize("which", ["LA", "SA"])
def test_lanczos_eigsh(n, which):
    torch.manual_seed(42)

    A = torch.randn(n, n)
    A = A + A.T

    # Compute the top k eigenpairs using our implementation
    w, v = lanczos_eigsh(A, which=which)

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
