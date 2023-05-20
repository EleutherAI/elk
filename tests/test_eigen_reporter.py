import torch
from einops import rearrange

from elk.training import EigenReporter, EigenReporterConfig


def test_eigen_reporter():
    num_clusters = 5
    hidden_size = 10
    n = 100

    x = torch.randn(n, num_clusters, 2, hidden_size, dtype=torch.float64)
    x1, x2 = x.chunk(2, dim=0)

    x = rearrange(x, "n v k d -> (n v) k d")
    x = x - x.mean(dim=(0, 1))
    x_neg, x_pos = x.unbind(1)

    reporter = EigenReporter(
        EigenReporterConfig(),
        hidden_size,
        dtype=torch.float64,
        num_variants=num_clusters,
    )
    reporter.update(x1)
    reporter.update(x2)

    # Check that the streaming negative covariance is correct
    true_xcov = x_neg.mT @ x_pos / x.shape[0]
    true_xcov = 0.5 * (true_xcov + true_xcov.mT)
    torch.testing.assert_close(reporter.contrastive_xcov, true_xcov)

    assert reporter.n == n * num_clusters
