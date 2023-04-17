import math

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.distributions.normal import Normal

from elk.metrics import accuracy_ci, roc_auc


def test_auroc_and_acc():
    # Generate 1D binary classification dataset
    X_1d, y_true_1d = make_classification(n_samples=1000, random_state=42)

    # Generate 2D matrix of binary classification datasets
    X_2d_1, y_true_2d_1 = make_classification(n_samples=1000, random_state=43)
    X_2d_2, y_true_2d_2 = make_classification(n_samples=1000, random_state=44)

    # Fit LR models and get predicted probabilities for 1D and 2D cases
    lr_1d = LogisticRegression(random_state=42).fit(X_1d, y_true_1d)
    y_scores_1d = lr_1d.predict_proba(X_1d)[:, 1]

    lr_2d_1 = LogisticRegression(random_state=42).fit(X_2d_1, y_true_2d_1)
    y_scores_2d_1 = lr_2d_1.predict_proba(X_2d_1)[:, 1]

    lr_2d_2 = LogisticRegression(random_state=42).fit(X_2d_2, y_true_2d_2)
    y_scores_2d_2 = lr_2d_2.predict_proba(X_2d_2)[:, 1]

    # Stack the datasets into 2D matrices
    y_true_2d = np.vstack((y_true_2d_1, y_true_2d_2))
    y_scores_2d = np.vstack((y_scores_2d_1, y_scores_2d_2))

    # Convert to PyTorch tensors
    y_true_1d_torch = torch.tensor(y_true_1d)
    y_scores_1d_torch = torch.tensor(y_scores_1d)
    y_true_2d_torch = torch.tensor(y_true_2d)
    y_scores_2d_torch = torch.tensor(y_scores_2d)

    # Calculate ROC AUC score using batch_roc_auc_score function for 1D and 2D cases
    roc_auc_1d_torch = roc_auc(y_true_1d_torch, y_scores_1d_torch).item()
    roc_auc_2d_torch = roc_auc(y_true_2d_torch, y_scores_2d_torch).numpy()

    # Calculate ROC AUC score with sklearn's roc_auc_score function for 1D and 2D cases
    roc_auc_1d_sklearn = roc_auc_score(y_true_1d, y_scores_1d)
    roc_auc_2d_sklearn = np.array(
        [
            roc_auc_score(y_true_2d_1, y_scores_2d_1),
            roc_auc_score(y_true_2d_2, y_scores_2d_2),
        ]
    )

    # Assert that the results from the two implementations are almost equal
    np.testing.assert_almost_equal(roc_auc_1d_torch, roc_auc_1d_sklearn)
    np.testing.assert_almost_equal(roc_auc_2d_torch, roc_auc_2d_sklearn)

    ### Test accuracy_ci function ###
    # Compute accuracy confidence interval
    level = 0.95
    hard_preds = y_scores_1d_torch > 0.5
    acc_ci = accuracy_ci(y_true_1d_torch, hard_preds, level=level)

    # Point estimate of the accuracy
    acc = hard_preds.eq(y_true_1d_torch).float().mean()

    # Compute the CI quantiles
    alpha = (1 - level) / 2
    q = acc.new_tensor([alpha, 1 - alpha])

    # Normal approximation to the binomial distribution
    stderr = (acc * (1 - acc) / len(y_true_1d_torch)) ** 0.5
    lower, upper = Normal(acc, stderr).icdf(q).tolist()

    # Assert that the results from the two implementations are close
    assert math.isclose(acc_ci.lower, lower, rel_tol=2e-3)
    assert math.isclose(acc_ci.upper, upper, rel_tol=2e-3)
