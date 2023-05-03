import torch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from elk.training import SpectralNorm


def test_stats():
    num_features = 3
    num_classes = 2
    batch_size = 10
    num_batches = 5

    # Initialize the SpectralNorm
    norm = SpectralNorm(num_features, num_classes)

    # Generate random data
    torch.manual_seed(42)
    x_data = [torch.randn(batch_size, num_features) for _ in range(num_batches)]
    y_data = [
        torch.randint(0, num_classes, (batch_size, num_classes))
        for _ in range(num_batches)
    ]

    # Compute cross-covariance matrix using batched updates
    for x, y in zip(x_data, y_data):
        norm.update(x, y)

    # Compute the expected cross-covariance matrix using the whole dataset
    x_all = torch.cat(x_data)
    y_all = torch.cat(y_data)
    mean_x = x_all.mean(dim=0)
    mean_y = y_all.type_as(x_all).mean(dim=0)
    x_centered = x_all - mean_x
    y_centered = y_all - mean_y
    expected_var = x_all.var(dim=0, unbiased=False)
    expected_xcov = x_centered.t().mm(y_centered) / (batch_size * num_batches)

    # Compare the computed cross-covariance matrix with the expected one
    torch.testing.assert_close(norm.var_x, expected_var)
    torch.testing.assert_close(norm.xcov, expected_xcov)


def test_projection():
    n, d = 1000, 20

    X, Y = make_classification(n_samples=n, n_features=d, random_state=42)
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    norm = SpectralNorm(d, 1).update(X_t, Y_t)
    X_ = norm(X_t)

    # Means should be equal before and after the projection
    torch.testing.assert_close(X_t.mean(dim=0), X_.mean(dim=0) + norm.mean_x)

    # Logistic regression should not be able to learn anything
    null_lr = LogisticRegression().fit(X_.numpy(), Y)
    beta = torch.from_numpy(null_lr.coef_)
    assert beta.norm(p=torch.inf) < 5e-5

    # But it should learn something before the projection
    real_lr = LogisticRegression().fit(X, Y)
    beta = torch.from_numpy(real_lr.coef_)
    assert beta.norm(p=torch.inf) > 5e-5
