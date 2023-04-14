from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import brentq
from torch import Tensor

from elk.metrics import to_one_hot
from elk.training import Classifier


@dataclass
class RlaceResult:
    """The result of applying R-LACE."""

    P: Tensor
    """The orthogonal projection matrix."""
    P_relaxed: Tensor
    """The relaxed projection matrix."""
    subspace: Tensor
    """The subspace erased by the projection matrix."""
    clf: Classifier
    """The best-response classifier."""
    clf_loss: float
    """The loss of the best-response classifier."""


@torch.no_grad()
def fantope_project(A: Tensor, d: int = 1) -> Tensor:
    """Project `A` to the Fantope."""
    L, Q = torch.linalg.eigh((A + A.T) / 2)

    # Solve the eigenvalue constraint on the CPU
    L_cpu = L.cpu()
    L -= brentq(
        lambda theta: torch.clamp(L_cpu - theta, 0, 1).sum() - d,
        a=L_cpu.max(),
        b=L_cpu.min() - 1,
    )
    return Q @ L.clamp(0, 1).diag() @ Q.T


def sal(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    rank: int = 1,
):
    """Spectral Attribute Removal <https://arxiv.org/abs/2203.07893>."""
    # Compute the direction of highest covariance with the labels
    # and use this to initialize the projection matrix. This usually
    # gets us most of the way to the optimal solution.
    y_one_hot = to_one_hot(y, num_classes).float() if num_classes > 2 else y
    cross_cov = (X - X.mean(0)).T @ (y_one_hot - y_one_hot.mean(0)) / X.shape[0]
    if num_classes > 2:
        u, _, _ = torch.svd_lowrank(cross_cov, q=rank)
    else:
        # We can skip the SVD entirely for binary classification
        u = F.normalize(cross_cov, dim=0).unsqueeze(1)

    return u @ u.T


def rlace(
    X: torch.Tensor,
    y: torch.Tensor,
    rank: int = 1,
    *,
    max_iter: int = 100,
    lr: float = 1e-2,
    tolerance_grad: float = 1e-5,
    tolerance_loss: float = 1e-2,
) -> RlaceResult:
    """
    Apply Relaxed Linear Adversarial Concept Erasure (R-LACE) to `X` and `y`.

    R-LACE locates a rank k projection matrix P which maximizes the loss of the
    optimal classifier on the projected data. Method from Ravfogel et al. (2022)
    <https://arxiv.org/abs/2201.12091>.

    Args:
        X: The data matrix (n x d)
        y: The labels (n)
        rank: The rank of the projection matrix
        max_iter: The maximum number of iterations
        lr: The learning rate for the projection matrix
        tolerance_loss: How close the classifier loss must be to the random baseline
            before we break.
        tolerance_grad: The tolerance for the squared gradient norm
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D tensor.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D tensor.")

    n, d = X.shape
    if n < d:
        raise ValueError("Must have n >= d.")
    if n != len(y):
        raise ValueError("Number of labels must match number of rows in X.")

    class_sizes = torch.bincount(y.long())
    num_classes = len(class_sizes)
    if num_classes == 1:
        raise ValueError("Must have at least two classes.")
    elif num_classes == 2:
        loss_fn = F.binary_cross_entropy_with_logits
        y = y.float()
    else:
        loss_fn = F.cross_entropy
        y = y.long()

    # Compute entropy of the labels
    fracs = class_sizes / n
    eps = torch.finfo(fracs.dtype).eps
    H = -torch.sum(fracs * fracs.add(eps).log())

    # Initialize with Spectral Attribute Removal
    P = sal(X, y, num_classes, rank)
    P.requires_grad = True

    # We use a small learning rate for the projection matrix instead of strong Wolfe
    # line search because the projection matrix is usually quite close to the optimal
    # solution at initialization, and line search seems to end up overshooting and
    # causing divergence.
    adv_opt = torch.optim.LBFGS([P], lr=lr, tolerance_grad=1e-4)
    clf = Classifier(d, num_classes=num_classes, device=X.device)

    def adv_closure():
        adv_opt.zero_grad()
        P.data = fantope_project(P, rank)

        eye = torch.eye(d, device=X.device)
        loss_P = -loss_fn(clf(X @ (eye - P)).squeeze(), y)
        loss_P.backward()
        return float(loss_P)

    # "best" here means HIGHEST classifier loss; we're trying to maximize the loss of
    # the best-response classifier. We want to ensure that even if we start diverging
    # for some reason, we can still recover the best solution we've seen.
    best_loss: float = -torch.inf
    best_P = P.detach().clone()

    for _ in range(max_iter):
        # Alternate between optimizing the projection matrix and the classifier
        clf.requires_grad_(True)
        clf_loss = clf.fit(
            X @ (torch.eye(d, device=X.device) - P.detach()), y, l2_penalty=0.0
        )
        if clf_loss > best_loss:
            best_loss = clf_loss
            best_P.copy_(P.detach())

            # Check if we've reached the random baseline
            if H - clf_loss < tolerance_loss:
                break

        clf.requires_grad_(False)
        adv_opt.step(adv_closure)

        # Check if we're very close to a saddle point
        grads = [p.grad for p in clf.parameters()] + [P.grad]
        grad_norm_sq = (
            torch.cat([g.view(-1) for g in grads if g is not None]).square().sum()
        )
        if grad_norm_sq < tolerance_grad:
            break

    # Make P an actual orthogonal projection matrix
    _, U = torch.linalg.eigh(best_P)
    U = U.T
    W = U[-rank:]
    P_final = torch.eye(d, device=W.device) - W.T @ W

    return RlaceResult(
        P=P_final.detach(),
        P_relaxed=torch.eye(d, device=U.device) - best_P,
        subspace=W.detach(),
        clf=clf,
        clf_loss=best_loss,
    )
