from math import e as math_e, tanh as math_tanh
from numpy import sign as np_sign

__all__ = ["linear", "sign", "tanh"]


def linear(a: float) -> float:
    """
    Linear "Identity" Activation Function

    Args:
        a: pre-activation value

    Returns:
        post-activation value
    """
    return a


def sign(a: float) -> float:
    """
    Sign, or Signum, Activation Function

    Args:
        a: pre-activation value

    Returns:
        post-activation value
    """
    # if a > 0.0:
    #     return 1.0
    # elif a < 0.0:
    #     return -1.0
    # else:
    #     return 0.0
    return np_sign(a)


def tanh(a: float) -> float:
    """
    Hyperbolic Tangent Activation Function

    Args:
        a: pre-activation value

    Returns:
        post-activation value
    """
    # e_a = math_e ** a
    # e_neg_a = math_e ** -a
    # return (e_a - e_neg_a) / (e_a + e_neg_a)
    return math_tanh(a)
