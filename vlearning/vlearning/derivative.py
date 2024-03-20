"""This module contains a function that can create numerical derivatives of functions.

Typical usage example:

    def f(x):
        return x ** 2

    f_prime = derivative(f)
"""
from collections.abc import Callable


def derivative(function: Callable, *, delta: float = 0.01) -> Callable:
    """Creates a numerical derivative function for the given input function.

    The resulting numerical derivative function is able to calculate a
    numerical approximation of the slope at a given position.

    Args:
        function: The function for which the derivative function needs to be made.

    Keyword Args:
        delta: The step size slope is calculated for.

    Returns:
        Numerical derivative function of the input function.
    """
    def wrapper_derivative(x, *args) -> float:
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = f"{function.__name__}’"
    wrapper_derivative.__qualname__ = f"{function.__qualname__}’"
    return wrapper_derivative
