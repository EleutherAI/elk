import math
from random import Random

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from elk.utils import batch_cov, cov_mean_fused, stochastic_round_constrained


def test_cov_mean_fused():
    X = torch.randn(10, 500, 100, dtype=torch.float64)
    cov_gt = batch_cov(X).mean(dim=0)
    cov_fused = cov_mean_fused(X)
    assert torch.allclose(cov_gt, cov_fused)


@given(
    # Let Hypothesis generate the number of floats...
    st.integers(min_value=1, max_value=100),
    # ...and the total sum of the floats
    st.integers(min_value=1, max_value=int(np.finfo(np.float32).max)),
)
def test_stochastic_rounding(num_parts: int, total: int):
    # Randomly sample the breakdown of the total into floats
    rng = np.random.default_rng(42)
    x = rng.dirichlet(np.ones(num_parts), size=1) * total

    # Stochastically round the floats
    rounded = stochastic_round_constrained(x[0].tolist(), Random(42))

    # Check that the rounded floats sum to the total
    assert math.isclose(sum(rounded), total)

    # Check that the rounded floats are never smaller than np.floor(x)
    assert all(math.floor(x_) <= r_ for x_, r_ in zip(x[0], rounded))

    # TODO: Check that the rounded floats are never larger than np.ceil(x),
    # once our implementation actually has this property.
