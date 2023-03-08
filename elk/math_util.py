import math
import random


def stochastic_round_constrained(x: list[float], rng: random.Random) -> list[int]:
    """Stochastic rounding under integer constraints.

    Given a list of floats which sum to an integer, stochastically round each float to
    an integer while maintaining the same sum. The expectation of the rounded values
    should be (nearly?) equal to the original values.

    Inspired by the algorithm described in https://arxiv.org/abs/1501.00014. Instead of
    deterministically allocating the total shortfall to the elements with the largest
    fractional parts, we randomly allocate it to elements with probability proportional
    to their fractional parts.
    """

    total = sum(x)
    assert math.isclose(total, round(total)), "Total must be an integer"

    rounded = [math.floor(x_) for x_ in x]
    fractional_parts = [x_ - r_ for x_, r_ in zip(x, rounded)]

    # Randomly choose where to allocate the shortfall. Note that currently we are
    # allocating the entire shortfall to a single element, but we could allocate
    # fractions of the shortfall to multiple elements. This would be lower variance
    # but requires an RNG that supports weighted sampling without replacement- this
    # exists in NumPy but not in the Python standard library so we'd need to switch
    # this function to use NumPy (and maintain a NumPy RNG).
    if any(fractional_parts):
        (index,) = rng.choices(range(len(x)), weights=fractional_parts)
        rounded[index] += round(sum(fractional_parts))

    return rounded
