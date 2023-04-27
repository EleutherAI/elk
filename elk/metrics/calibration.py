from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class CalibrationEstimate:
    ece: float
    num_bins: int


@dataclass
class CalibrationError:
    """Monotonic Sweep Calibration Error for binary problems.

    This method estimates the True Calibration Error (TCE) by searching for the largest
    number of bins into which the data can be split that preserves the monotonicity
    of the predicted confidence -> empirical accuracy mapping. We use equal mass bins
    (quantiles) instead of equal width bins. Roelofs et al. (2020) show that this
    estimator has especially low bias in simulations where the TCE is analytically
    computable, and is hyperparameter-free (except for the type of norm used).

    Paper: "Mitigating Bias in Calibration Error Estimation" by Roelofs et al. (2020)
    Link: https://arxiv.org/abs/2012.08668
    """

    labels: list[Tensor] = field(default_factory=list)
    pred_probs: list[Tensor] = field(default_factory=list)

    def update(self, labels: Tensor, probs: Tensor) -> "CalibrationError":
        labels, probs = labels.detach().flatten(), probs.detach().flatten()
        assert labels.shape == probs.shape
        assert torch.is_floating_point(probs)

        self.labels.append(labels)
        self.pred_probs.append(probs)
        return self

    def compute(self, p: int = 2) -> CalibrationEstimate:
        """Compute the expected calibration error.

        Args:
            p: The norm to use for the calibration error. Defaults to 2 (Euclidean).
        """
        labels = torch.cat(self.labels)
        pred_probs = torch.cat(self.pred_probs)

        n = len(pred_probs)
        if n < 2:
            raise ValueError("Not enough data to compute calibration error.")

        # Sort the predictions and labels
        pred_probs, indices = pred_probs.sort()
        labels = labels[indices].float()

        # Search for the largest number of bins which preserves monotonicity.
        # Based on Algorithm 1 in Roelofs et al. (2020).
        # Using a single bin is guaranteed to be monotonic, so we start there.
        b_star, accs_star = 1, labels.mean().unsqueeze(0)
        for b in range(2, n + 1):
            # Split into (nearly) equal mass bins
            freqs = torch.stack([h.mean() for h in labels.tensor_split(b)])

            # This binning is not strictly monotonic, let's break
            if not torch.all(freqs[1:] > freqs[:-1]):
                break

            elif not torch.all(freqs * (1 - freqs)):
                break

            # Save the current binning, it's monotonic and may be the best one
            else:
                accs_star = freqs
                b_star = b

        # Split into (nearly) equal mass bins. They won't be exactly equal, so we
        # still weight the bins by their size.
        conf_bins = pred_probs.tensor_split(b_star)
        w = pred_probs.new_tensor([len(c) / n for c in conf_bins])

        # See the definition of ECE_sweep in Equation 8 of Roelofs et al. (2020)
        mean_confs = torch.stack([c.mean() for c in conf_bins])
        ece = torch.sum(w * torch.abs(accs_star - mean_confs) ** p) ** (1 / p)

        return CalibrationEstimate(float(ece), b_star)
