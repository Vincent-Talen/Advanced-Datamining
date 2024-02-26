from typing import Callable


def derivative(function: Callable, *, delta: float = 0.01):
    """
    Creates the derivative function for the given input function.
    The resulting derivative function is able to calculate a
    numerical approximation of the slope at a given position.

    Args:
        function: The function the derivative function is made from
        delta: The step size slope is calculated for

    Returns:
        Derivative function of the input function
    """
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative
