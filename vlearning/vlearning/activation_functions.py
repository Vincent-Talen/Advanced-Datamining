"""Module defining various activation functions.

Available activation functions:
    - linear
    - sign
    - tanh
    - softsign
    - sigmoid
    - softplus
    - relu
    - swish
    - nipuna
"""
from math import tanh as math_tanh, exp, log1p

__all__ = [
    "linear",
    "sign",
    "tanh",
    "softsign",
    "sigmoid",
    "softplus",
    "relu",
    "swish",
    "nipuna",
]


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
    # e_a = exp(a)
    # e_neg_a = exp(-a)
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


def sigmoid(a: float) -> float:
    """Sigmoid Activation Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    # Normal logistic sigmoid function
    if a >= 0:
        return 1 / (1 + exp(-a))
    # Equivalent formula to avoid overflow
    e_a = exp(a)
    return e_a / (1 + e_a)


def softplus(a: float) -> float:
    """Softplus Activation Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    return log1p(exp(-abs(a))) + max(a, 0.0)


def relu(a: float) -> float:
    """Rectified Linear Unit (ReLU) Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    return max(0.0, a)


def swish(a: float, *, beta: float = 1.0) -> float:
    """Sigmoid-weighted Linear Unit (swish) Activation Function

    Args:
        a: Pre-activation value.

    Keyword Args:
        beta: The beta parameter for the swish function.

    Returns:
        The calculated post-activation value.
    """
    return a * sigmoid(beta * a)


def nipuna(a: float, *, beta: float = 1.0) -> float:
    """NIPUNA Activation Function

    Args:
        a: Pre-activation value.

    Keyword Args:
        beta: The beta parameter to control the sharpness of the function.

    Returns:
        The calculated post-activation value.
    """
    if a >= 0:
        x = a / (1 + exp(-beta * a))
    else:
        exp_beta_a = exp(beta * a)
        x = a * exp_beta_a / (1 + exp_beta_a)
    return max(x, a)
