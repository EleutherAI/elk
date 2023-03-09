import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from elk.training.classifier import Classifier


def test_classifier_roughly_same_sklearn():
    input_dims: int = 10
    # make a classification problem of 1000 samples with input_dims features
    features: np.ndarray
    truths: np.ndarray
    _features, _truths = make_classification(
        n_samples=1000, n_features=input_dims, random_state=0
    )
    # use float32 for the features so it's the same as the default dtype for torch
    features = _features.astype(np.float32)
    truths = _truths.astype(np.float32)
    # train a logistic regression model on the data. No regularization
    model = LogisticRegression(penalty="none", solver="lbfgs")
    model.fit(features, truths)
    # train a classifier on the data
    classifier = Classifier(input_dim=input_dims, device="cpu")
    # float32 is the default dtype for torch tensors
    classifier.fit(torch.from_numpy(features), torch.from_numpy(truths))
    # check that the weights are roughly the same
    sklearn_coef = model.coef_
    torch_coef = classifier.linear.weight.detach().numpy()
    assert np.allclose(sklearn_coef, torch_coef, atol=1e-2)

    # check that on a new sample, the predictions are roughly the same
    new_sample = np.random.randn(input_dims).astype(np.float32)
    # 2d array, need to index into the first row and the second column
    sklearn_pred = model.predict_proba(new_sample.reshape(1, -1))[0][1]
    # 1d array, need to index into the first element
    torch_pred = classifier(torch.from_numpy(new_sample)).sigmoid().detach().numpy()[0]
    assert np.allclose(sklearn_pred, torch_pred, atol=1e-2)
