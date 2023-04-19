"""Loss functions for training reporters."""

import warnings
from inspect import signature

import torch
from torch import Tensor

LOSSES = dict()  # Registry of loss functions


def register(name):
    """A decorator to register a function to LOSSES"""

    def decorate(func):
        assert signature(func).parameters.keys() == {"logit0", "logit1", "coef"}, (
            f"Loss function {func.__name__} must take arguments "
            "`logit0`, `logit1`, and `coef`."
        )
        assert (
            name not in LOSSES
        ), f"Loss function {name} conflicts with existing function."
        LOSSES[name] = func
        return func

    return decorate


def H(p: Tensor) -> Tensor:
    """Entropy of Bernoulli distribution(s) with success probability `p`."""
    return torch.nn.functional.binary_cross_entropy(p, p)


@register("ccs")
def ccs_squared_loss(logit0: Tensor, logit1: Tensor, coef: float = 1.0) -> Tensor:
    """CCS loss from original paper, with squared differences between probabilities.

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logit0: The log odds for the original proposition.
        logit1: The log odds for the negated proposition.
        coef: The coefficient to multiply the loss by.
    Returns:
        The sum of the consistency and confidence losses.
    """
    loss = consistency_squared_loss(logit0, logit1) + confidence_squared_loss(
        logit0, logit1
    )
    return coef * loss


@register("ccs_prompt_var")
def ccs_prompt_var_loss(logit0: Tensor, logit1: Tensor, coef: float = 1.0) -> Tensor:
    """CCS loss with prompt variance regularization.

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logit0: The log odds for the original proposition. Shape ([batch,] n_variants)
        logit1: The log odds for the negated proposition. Shape ([batch,] n_variants)
        coef: The coefficient to multiply the loss by.
    Returns:
        The sum of the consistency and confidence losses.
    """
    loss = (
        consistency_squared_loss(logit0, logit1)
        + confidence_squared_loss(logit0, logit1)
        + prompt_var_loss(logit0, logit1)
    )
    return coef * loss


@register("js")
def js_loss(
    logit0: Tensor,
    logit1: Tensor,
    coef: float = 1.0,
) -> Tensor:
    """Negation consistency loss based on the Jensen-Shannon divergence.

    Note that by default we use the base 2 logarithm, so the value is measured in bits.
    This ensures the divergence is in the range [0, 1]."""
    p0, neg_p1 = logit0.sigmoid(), 1 - logit1.sigmoid()
    nats = H((p0 + neg_p1) / 2) - (H(p0) + H(neg_p1)) / 2
    return coef * nats


@register("js_confidence")
def js_confidence_loss(
    logit0: Tensor,
    logit1: Tensor,
    coef: float = 1.0,
) -> Tensor:
    """Confidence loss based on the Jensen-Shannon divergence. This is the same as the
    entropy of the 50/50 mixture of the two distributions.

    Note that by default we use the base 2 logarithm, so the value is measured in bits.
    This ensures the divergence is in the range [0, 1]."""
    p0, neg_p1 = logit0.sigmoid(), 1 - logit1.sigmoid()
    nats = H((p0 + neg_p1) / 2)
    return coef * nats


@register("consistency_squared")
def consistency_squared_loss(
    logit0: Tensor,
    logit1: Tensor,
    coef: float = 1.0,
) -> Tensor:
    """Negation consistency loss based on the squared difference between the
    two distributions."""
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()
    return coef * p0.sub(1 - p1).square().mean()


@register("confidence_squared")
def confidence_squared_loss(
    logit0: Tensor,
    logit1: Tensor,
    coef: float = 1.0,
) -> Tensor:
    """Confidence loss based on the squared difference between the two distributions."""
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()
    return coef * torch.min(p0, p1).square().mean()


@register("prompt_var_squared")
def prompt_var_loss(logit0: Tensor, logit1: Tensor, coef: float = 1.0) -> Tensor:
    """
    Prompt-variance loss: the squared difference between the probability
    of a proposition and the mean probability over all variants of that
    proposition (templates).

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logit0: The log odds for the original proposition. shape ([batch,] n_variants)
        logit1: The log odds for the negated proposition. shape ([batch,] n_variants)
        coef: The coefficient to multiply the loss by.
    """
    assert logit0.shape == logit1.shape
    assert len(logit0.shape) in [1, 2]
    if logit0.shape[-1] == 1:
        warnings.warn(
            "Only one variant provided. Prompt variance loss will cause errors."
        )
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()

    var0 = p0.var(dim=-1, unbiased=False).mean()
    var1 = p1.var(dim=-1, unbiased=False).mean()
    prompt_variance = var0 + var1
    return coef * prompt_variance
