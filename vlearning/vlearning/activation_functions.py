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
    - elish
    - hardelish
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
    "elish",
    "hardelish",
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
    return a / (1.0 + abs(a))


def sigmoid(a: float) -> float:
    """Sigmoid Activation Function

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    # Normal logistic sigmoid function
    if a >= 0.0:
        return 1.0 / (1.0 + exp(-a))
    # Equivalent formula to avoid overflow
    e_a = exp(a)
    return e_a / (1.0 + e_a)


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
    if a >= 0.0:
        return a
    exp_beta_a = exp(beta * a)
    return a * exp_beta_a / (1.0 + exp_beta_a)


def elish(a: float) -> float:
    """ELiSH (Exponential Linear Sigmoid SquasHing) Activation Function

    The ELiSH activation function uses a multiplication of ELU (Exponential Linear
    Unit) and Sigmoid in its negative part and its positive part is the same as Swish.

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    if a >= 0.0:
        return swish(a)
    exp_a = exp(a)
    return (exp_a * (exp_a - 1.0)) / (1.0 + exp_a)


def hardelish(a: float) -> float:
    """HardELiSH (Hard Exponential Linear Sigmoid SquasHing) Activation Function

    The HardELiSH activation function uses a multiplication of HardSigmoid and ELU
    (Exponential Linear Unit) in negative part and HardSigmoid and Linear in its
    positive part.

    Args:
        a: Pre-activation value.

    Returns:
        The calculated post-activation value.
    """
    max_min_part = max(0.0, min(1.0, (a + 1.0) / 2.0))
    if a >= 0.0:
        return a * max_min_part
    return (exp(a) - 1.0) * max_min_part
