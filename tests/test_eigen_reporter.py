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
    assert torch.allclose(reporter.intercluster_cov, expected_var)

    # Check that the streaming invariance (intra-cluster variance) is correct
    expected_invariance = cov_mean_fused(x_pos) + cov_mean_fused(x_neg)
    assert torch.allclose(reporter.intracluster_cov, expected_invariance)

    # Check that the streaming negative covariance is correct
    cross_cov = (pos_centroids - pos_mu).mT @ (neg_centroids - neg_mu) / num_clusters
    cross_cov = cross_cov + cross_cov.mT
    assert torch.allclose(reporter.contrastive_xcov, cross_cov)

    assert reporter.n == num_clusters


def test_supervised_eigen_reporter():
    cluster_size = 5
    hidden_size = 10
    num_clusters = 100

    x_pos = torch.randn(num_clusters, cluster_size, hidden_size, dtype=torch.float64)
    x_neg = torch.randn(num_clusters, cluster_size, hidden_size, dtype=torch.float64)
    labels = torch.randint(0, 2, (num_clusters,), dtype=torch.long)
    x_true = torch.cat([x_pos[labels == 1], x_neg[labels == 0]], dim=0)
    x_false = torch.cat([x_pos[labels == 0], x_neg[labels == 1]], dim=0)

    cfg = EigenReporterConfig(supervised_inv_weight=1, supervised_var_weight=1)
    reporter = EigenReporter(hidden_size, cfg, dtype=torch.float64)
    reporter.update(x_pos, x_neg, labels)

    # Check that the streaming supervised mean is correct
    true_mu, false_mu = x_true.mean(dim=(0, 1)), x_false.mean(dim=(0, 1))
    assert torch.allclose(reporter.true_mean, true_mu)
    assert torch.allclose(reporter.false_mean, false_mu)

    x_pos1, x_pos2 = x_pos.chunk(2, dim=0)
    x_neg1, x_neg2 = x_neg.chunk(2, dim=0)
    labels1, labels2 = labels.chunk(2, dim=0)

    cfg = EigenReporterConfig(supervised_inv_weight=1, supervised_var_weight=1)
    reporter = EigenReporter(hidden_size, cfg, dtype=torch.float64)
    reporter.update(x_pos1, x_neg1, labels1)
    reporter.update(x_pos2, x_neg2, labels2)

    # Check that the streaming mean is correct
    pos_mu, neg_mu = x_pos.mean(dim=(0, 1)), x_neg.mean(dim=(0, 1))
    assert torch.allclose(reporter.pos_mean, pos_mu)
    assert torch.allclose(reporter.neg_mean, neg_mu)

    # Check that the streaming supervised mean is correct
    true_mu, false_mu = x_true.mean(dim=(0, 1)), x_false.mean(dim=(0, 1))
    assert torch.allclose(reporter.true_mean, true_mu)
    assert torch.allclose(reporter.false_mean, false_mu)

    # Check that the streaming covariance is correct
    pos_centroids, neg_centroids = x_pos.mean(dim=1), x_neg.mean(dim=1)
    expected_var = batch_cov(pos_centroids) + batch_cov(neg_centroids)
    assert torch.allclose(reporter.intercluster_cov, expected_var)

    # Check that the streaming supervised covariance (interclass) is correct
    cat = torch.cat([true_mu.unsqueeze(1), false_mu.unsqueeze(1)], dim=1).mT  # (2, d)
    expected_var = cat.mT @ cat / 2
    assert torch.allclose(reporter.supervised_interclass_cov, expected_var)

    # Check that the streaming invariance (intra-cluster variance) is correct
    expected_invariance = cov_mean_fused(x_pos) + cov_mean_fused(x_neg)
    assert torch.allclose(reporter.intracluster_cov, expected_invariance)

    # Check that the streaming supervised invariance (intra-class variance) is correct
    expected_var = batch_cov(x_true.mean(1)) + batch_cov(x_false.mean(1))
    assert torch.allclose(reporter.supervised_intraclass_cov, expected_var)

    # Check that the streaming negative covariance is correct
    cross_cov = (pos_centroids - pos_mu).mT @ (neg_centroids - neg_mu) / num_clusters
    cross_cov = cross_cov + cross_cov.mT
    assert torch.allclose(reporter.contrastive_xcov, cross_cov)

    assert reporter.n == num_clusters
