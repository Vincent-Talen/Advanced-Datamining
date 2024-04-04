"""Module defining various activation functions.

Available activation functions:
    - linear
    - sign
    - tanh
"""
from math import tanh as math_tanh
# from math import e as math_e, tanh as math_tanh

__all__ = ["linear", "sign", "tanh"]


def linear(a: float) -> float:
    """Linear "Identity" Activation Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    return a


def sign(a: float) -> float:
    """Sign (a.k.a. Signum) Activation Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    if a > 0.0:
        return 1.0
    if a < 0.0:
        return -1.0
    return 0.0


def tanh(a: float) -> float:
    """Hyperbolic Tangent Activation Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    # e_a = math_e ** a
    # e_neg_a = math_e ** -a
    # return (e_a - e_neg_a) / (e_a + e_neg_a)
    return math_tanh(a)


def softsign(a: float) -> float:
    """Softsign Activation Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    return a / (1 + abs(a))
