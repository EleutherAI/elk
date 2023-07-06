import torch

from elk.training import EigenFitter, EigenFitterConfig
from elk.utils import batch_cov, cov_mean_fused


def test_eigen_reporter():
    num_clusters = 5
    hidden_size = 10
    N = 100

    x = torch.randn(N, num_clusters, 2, hidden_size, dtype=torch.float64)
    x1, x2 = x.chunk(2, dim=0)
    x_neg, x_pos = x.unbind(2)

    reporter = EigenFitter(
        EigenFitterConfig(),
        hidden_size,
        dtype=torch.float64,
        num_variants=num_clusters,
    )
    reporter.update(x1)
    reporter.update(x2)

    # Check that the streaming mean is correct
    neg_mu, pos_mu = x_neg.mean(dim=(0, 1)), x_pos.mean(dim=(0, 1))

    assert reporter.class_means is not None
    torch.testing.assert_close(reporter.class_means[0], neg_mu)
    torch.testing.assert_close(reporter.class_means[1], pos_mu)

    # Check that the streaming covariance is correct
    neg_centroids, pos_centroids = x_neg.mean(dim=1), x_pos.mean(dim=1)
    true_cov = 0.5 * (batch_cov(neg_centroids) + batch_cov(pos_centroids))
    torch.testing.assert_close(reporter.intercluster_cov, true_cov)

    # Check that the streaming negative covariance is correct
    true_xcov = (neg_centroids - neg_mu).mT @ (pos_centroids - pos_mu) / N
    true_xcov = 0.5 * (true_xcov + true_xcov.mT)
    torch.testing.assert_close(reporter.contrastive_xcov, true_xcov)

    # Check that the streaming invariance (intra-cluster variance) is correct.
    # This is actually the same whether or not we track class means.
    expected_invariance = 0.5 * (cov_mean_fused(x_neg) + cov_mean_fused(x_pos))
    torch.testing.assert_close(reporter.intracluster_cov, expected_invariance)

    assert reporter.n == N
