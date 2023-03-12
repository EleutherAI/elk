from elk.math_util import batch_cov, cov_mean_fused
from elk.training import EigenReporter, EigenReporterConfig
import torch


def test_eigen_reporter():
    cluster_size = 5
    hidden_size = 10
    num_clusters = 100

    x_pos = torch.randn(num_clusters, cluster_size, hidden_size, dtype=torch.float64)
    x_neg = torch.randn(num_clusters, cluster_size, hidden_size, dtype=torch.float64)
    x_pos1, x_pos2 = x_pos.chunk(2, dim=0)
    x_neg1, x_neg2 = x_neg.chunk(2, dim=0)

    reporter = EigenReporter(hidden_size, EigenReporterConfig(), dtype=torch.float64)
    reporter.update(x_pos1, x_neg1)
    reporter.update(x_pos2, x_neg2)

    # Check that the streaming mean is correct
    pos_mu, neg_mu = x_pos.mean(dim=(0, 1)), x_neg.mean(dim=(0, 1))
    assert torch.allclose(reporter.pos_mean, pos_mu)
    assert torch.allclose(reporter.neg_mean, neg_mu)

    # Check that the streaming covariance is correct
    pos_centroids, neg_centroids = x_pos.mean(dim=1), x_neg.mean(dim=1)
    expected_var = batch_cov(pos_centroids) + batch_cov(neg_centroids)
    assert torch.allclose(reporter.variance, expected_var)

    # Check that the streaming invariance (intra-cluster variance) is correct
    expected_invariance = cov_mean_fused(x_pos) + cov_mean_fused(x_neg)
    assert torch.allclose(reporter.invariance, expected_invariance)

    # Check that the streaming consistency (inter-cluster variance) is correct
    cross_cov = (pos_centroids - pos_mu).mT @ (neg_centroids - neg_mu) / num_clusters
    cross_cov = cross_cov + cross_cov.mT
    assert torch.allclose(reporter.neg_covariance, cross_cov)

    assert reporter.n == num_clusters
