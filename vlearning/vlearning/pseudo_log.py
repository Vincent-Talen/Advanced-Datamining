"""This module has a function that doesn't crash calculating the log of values below 0.

Typical usage example:

    from vlearning import pseudo_log

    x = pseudo_log(0.5)
"""
from math import log


def pseudo_log(x, *, epsilon: float = 0.0001) -> float:
    """When the logarithm of a value is needed but the value is close to, or below, 0.

    Because the normal logarithmic function is only defined for > 0, this can result in
    errors when the value in question can coincidentally be close to, or below, 0. This
    pseudo logarithmic function replaces the asymptotic tail of the logarithm below
    `epsilon` with a steep linear function, to avoid errors.

    Args:
        x: The value to calculate the pseudo logarithmic for.

    Keyword Args:
        epsilon: The value where the logarithm's asymptotic tail is replaced below.

    Returns:
        The
    """
    if x < epsilon:
        return log(epsilon) + (x - epsilon) / epsilon
    return log(x)
