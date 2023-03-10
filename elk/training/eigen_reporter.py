"""An ELK reporter network."""

from ..math_util import batch_cov
from dataclasses import dataclass
from pathlib import Path
from simple_parsing.helpers import Serializable
from sklearn.metrics import roc_auc_score
from torch import Tensor
from typing import Literal, NamedTuple, Optional, Union
import torch
import torch.nn as nn


class EvalResult(NamedTuple):
    """The result of evaluating a reporter on a dataset.

    The `.score()` function of a reporter returns an instance of this class,
    which contains the loss, accuracy, calibrated accuracy, and AUROC.
    """

    loss: float
    acc: float
    cal_acc: float
    auroc: float


@dataclass
class EigenReporterConfig(Serializable):
    """ """

    inv_weight: float = 1.0
    neg_cov_weight: float = 1.0
    solver: Literal["arpack", "dense", "power"] = "dense"


class EigenReporter(nn.Module):
    """A linear reporter whose weights are computed via eigendecomposition.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    def __init__(
        self, in_features: int, cfg: EigenReporterConfig, device: Optional[str] = None
    ):
        super().__init__()

        self.config = cfg
        self.linear = nn.Linear(in_features, 1, bias=False, device=device)

    # TODO: These methods will do something fancier in the future
    @classmethod
    def load(cls, path: Union[Path, str]):
        """Load a reporter from a file."""
        return torch.load(path)

    def save(self, path: Union[Path, str]):
        # TODO: Save separate JSON and PT files for the reporter.
        torch.save(self, path)

    def forward(self, x: Tensor) -> Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.linear(x)

    def validate_data(self, data):
        """Validate that the data's shape is valid."""
        assert len(data) == 2 and data[0].shape == data[1].shape

    def fit(
        self,
        contrast_pair: tuple[Tensor, Tensor],
    ) -> float:
        """Fit the probe to the contrast pair (x0, x1).

        Args:
            contrast_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations.
            labels: The labels of the contrast pair. Defaults to None.
            lr: The learning rate for Adam. Defaults to 1e-2.
            num_epochs: The number of epochs to train for. Defaults to 1000.
            num_tries: The number of times to repeat the procedure. Defaults to 10.
            optimizer: The optimizer to use. Defaults to "adam".
            verbose: Whether to print out information at each step. Defaults to False.
            weight_decay: The weight decay for Adam. Defaults to 0.01.

        Returns:
            best_loss: The best loss obtained.

        Raises:
            ValueError: If `optimizer` is not "adam" or "lbfgs".
            RuntimeError: If the best loss is not finite.
        """
        self.validate_data(contrast_pair)

        x_pos, x_neg = contrast_pair
        assert x_pos.shape == x_neg.shape

        # Variance
        pos_bar, neg_bar = x_pos.mean(1), x_neg.mean(1)  # [batch, d]
        inter_variance = batch_cov(pos_bar) + batch_cov(neg_bar)  # [d, d]

        # Invariance
        intra_variance = batch_cov(x_pos).mean(0) + batch_cov(x_neg).mean(0)  # [d, d]

        # Negative covariance
        contrastive_variance = pos_bar.mT @ neg_bar + neg_bar.mT @ pos_bar  # [d, d]

        alpha, beta = self.config.inv_weight, self.config.neg_cov_weight
        A = inter_variance - alpha * intra_variance - beta * contrastive_variance

        L, Q = torch.linalg.eigh(A)
        self.linear.weight.data = Q[:, -1, None]

        return L[-1]

    @torch.no_grad()
    def score(
        self,
        contrast_pair: tuple[Tensor, Tensor],
        labels: Tensor,
    ) -> EvalResult:
        """Score the probe on the contrast pair (x0, x1).

        Args:
            contrast_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations.
            labels: The labels of the contrast pair.

        Returns:
            an instance of EvalResult containing the loss, accuracy, calibrated
                accuracy, and AUROC of the probe on the contrast pair (x0, x1).
        """

        self.validate_data(contrast_pair)

        logit0, logit1 = map(self, contrast_pair)
        p0, p1 = logit0.sigmoid(), logit1.sigmoid()
        pred_probs = 0.5 * (p0 + (1 - p1))

        # Calibrated accuracy
        cal_thresh = pred_probs.float().quantile(labels.float().mean())
        cal_preds = pred_probs.gt(cal_thresh).squeeze(1).to(torch.int)
        raw_preds = pred_probs.gt(0.5).squeeze(1).to(torch.int)

        # makes `num_variants` copies of each label, all within a single
        # dimension of size `num_variants * n`, such that the labels align
        # with pred_probs.flatten()
        broadcast_labels = labels.repeat_interleave(pred_probs.shape[1])
        # roc_auc_score only takes flattened input
        auroc = float(roc_auc_score(broadcast_labels.cpu(), pred_probs.cpu().flatten()))
        cal_acc = cal_preds.flatten().eq(broadcast_labels).float().mean()
        raw_acc = raw_preds.flatten().eq(broadcast_labels).float().mean()

        return EvalResult(
            loss=0.0,
            acc=torch.max(raw_acc, 1 - raw_acc).item(),
            cal_acc=torch.max(cal_acc, 1 - cal_acc).item(),
            auroc=max(auroc, 1 - auroc),
        )
