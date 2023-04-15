import pytest
import torch

from elk.training import EigenReporter, EigenReporterConfig
from elk.utils import batch_cov, cov_mean_fused


@pytest.mark.parametrize("track_class_means", [True, False])
def test_eigen_reporter(track_class_means: bool):
    cluster_size = 5
    hidden_size = 10
    N = 100

    x = torch.randn(N, cluster_size, 2, hidden_size, dtype=torch.float64)
    x1, x2 = x.chunk(2, dim=0)
    x_neg, x_pos = x.unbind(2)

    reporter = EigenReporter(
        EigenReporterConfig(),
        hidden_size,
        dtype=torch.float64,
        num_classes=2 if track_class_means else None,
    )
    reporter.update(x1)
    reporter.update(x2)

    if track_class_means:
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
    else:
        assert reporter.class_means is None

        # Check that the covariance matrices are correct. When we don't track class
        # means, we expect intercluster_cov and contrastive_xcov to simply be averaged
        # over each batch passed to update().
        true_xcov = 0.0
        true_cov = 0.0
        for x_i in (x1, x2):
            x_neg_i, x_pos_i = x_i.unbind(2)
            neg_centroids, pos_centroids = x_neg_i.mean(dim=1), x_pos_i.mean(dim=1)
            true_cov += 0.5 * (batch_cov(neg_centroids) + batch_cov(pos_centroids))

            neg_mu_i, pos_mu_i = x_neg_i.mean(dim=(0, 1)), x_pos_i.mean(dim=(0, 1))
            xcov_asym = (neg_centroids - neg_mu_i).mT @ (pos_centroids - pos_mu_i)
            true_xcov += 0.5 * (xcov_asym + xcov_asym.mT)

        torch.testing.assert_close(reporter.intercluster_cov, true_cov / 2)
        torch.testing.assert_close(reporter.contrastive_xcov, true_xcov / N)

    # Check that the streaming invariance (intra-cluster variance) is correct.
    # This is actually the same whether or not we track class means.
    expected_invariance = 0.5 * (cov_mean_fused(x_neg) + cov_mean_fused(x_pos))
    torch.testing.assert_close(reporter.intracluster_cov, expected_invariance)

    assert reporter.n == N
