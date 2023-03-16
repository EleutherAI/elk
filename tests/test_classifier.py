from elk.training.classifier import Classifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import torch


@torch.no_grad()
def test_classifier_roughly_same_sklearn():
    input_dims: int = 100
    torch.manual_seed(0)

    # make a classification problem of 1000 samples with input_dims features
    features, truths = make_classification(
        n_samples=1000, n_features=input_dims, random_state=0
    )

    # train a logistic regression model on the data. No regularization
    model = LogisticRegression(penalty=None, solver="lbfgs")  # type: ignore[call-arg]
    model.fit(features, truths)

    # train a classifier on the data
    classifier = Classifier(input_dim=input_dims, device="cpu", dtype=torch.float64)
    classifier.fit(
        torch.from_numpy(features),
        torch.from_numpy(truths),
        l2_penalty=0.0,
    )
    # check that the weights are roughly the same
    sklearn_coef = torch.from_numpy(model.coef_)
    torch_coef = classifier.linear.weight.data
    torch.testing.assert_close(sklearn_coef, torch_coef, atol=1e-2, rtol=1e-2)

    # check that on a new sample, the predictions are roughly the same
    new_sample = torch.randn(10, input_dims, dtype=torch.float64)
    # 2d array, need to index into the first row and the second column
    sklearn_pred = torch.from_numpy(
        model.predict_proba(new_sample.numpy())[:, 1]
    ).squeeze()
    # 1d array, need to index into the first element
    torch_pred = classifier(new_sample).sigmoid().squeeze()

    torch.testing.assert_close(sklearn_pred, torch_pred, atol=1e-2, rtol=1e-2)
