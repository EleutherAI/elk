import torch

from elk.training import EigenReporter, EigenReporterConfig
from elk.utils import batch_cov, cov_mean_fused


def test_eigen_reporter():
    cluster_size = 5
    hidden_size = 10
    num_clusters = 100

    x = torch.randn(num_clusters, cluster_size, 2, hidden_size, dtype=torch.float64)
    x1, x2 = x.chunk(2, dim=0)

    reporter = EigenReporter(EigenReporterConfig(), hidden_size, dtype=torch.float64)
    reporter.update(x1)
    reporter.update(x2)

    # Check that the streaming mean is correct
    x_neg, x_pos = x.unbind(2)
    pos_mu, neg_mu = x_pos.mean(dim=(0, 1)), x_neg.mean(dim=(0, 1))
    torch.testing.assert_close(reporter.class_means[0], neg_mu)
    torch.testing.assert_close(reporter.class_means[1], pos_mu)

    # Check that the streaming covariance is correct
    pos_centroids, neg_centroids = x_pos.mean(dim=1), x_neg.mean(dim=1)
    expected_var = 0.5 * (batch_cov(pos_centroids) + batch_cov(neg_centroids))
    torch.testing.assert_close(reporter.intercluster_cov, expected_var)

    # Check that the streaming invariance (intra-cluster variance) is correct
    expected_invariance = 0.5 * (cov_mean_fused(x_pos) + cov_mean_fused(x_neg))
    torch.testing.assert_close(reporter.intracluster_cov, expected_invariance)

    # Check that the streaming negative covariance is correct
    cross_cov = (pos_centroids - pos_mu).mT @ (neg_centroids - neg_mu) / num_clusters
    cross_cov = 0.5 * (cross_cov + cross_cov.mT)
    torch.testing.assert_close(reporter.contrastive_xcov, cross_cov)

    assert reporter.n == num_clusters
