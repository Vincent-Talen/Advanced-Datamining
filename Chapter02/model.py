import math
from typing import Callable


# Activation Functions
def linear(a: float) -> float:
    """
    Linear "Identity" Activation Function

    Args:
        a: pre-activation value

    Returns:
        post-activation value
    """
    return a


def signum(a: float) -> float:
    """
    Signum Activation Function

    Args:
        a: pre-activation value

    Returns:
        post-activation value
    """
    if a > 0.0:
        return 1.0
    elif a < 0.0:
        return -1.0
    else:
        return 0.0


def tanh(a: float) -> float:
    """
    Hyperbolic Tangent Activation Function

    Args:
        a: pre-activation value

    Returns:
        post-activation value
    """
    # return (math.e ** a - math.e ** -a) / (math.e ** a + math.e ** -a)
    return math.tanh(a)


# Loss functions
def mean_squared_error(yhat: float, y: float) -> float:
    """
    Mean Squared Error Loss-function - Calculates the loss of an instance

    Args:
        yhat: predicted classification of the instance
        y: actual classification of the instance

    Returns:
        loss of instance
    """
    return (yhat - y) ** 2


def mean_absolute_error(yhat: float, y: float) -> float:
    """
    Mean Absolute Error Loss-function

    Args:
        yhat: predicted classification of the instance
        y: actual classification of the instance

    Returns:
        loss of instance
    """
    return abs(yhat - y)


def hinge(yhat: float, y: float) -> float:
    """
    Hinge Loss-function

    Args:
        yhat: predicted classification of the instance
        y: actual classification of the instance

    Returns:
        loss of instance
    """
    return max(1 - yhat * y, 0)


# Derivative function
def derivative(function: Callable, *, delta: float = 0.01):
    """
    Creates the derivative function for the input function.
    The resulting function can calculate a numerical approximation of the slope at a position.

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


# Neuron class
class Neuron:
    """
    Generalized model that computes an activation value from inputs it receives.
    This is done by weighting the inputs, adding bias, using activation functions and loss functions.
    Training is done by iterating over the data and updating the model's weights and bias using gradient descent,
    which minimizes the Loss, resulting in the model becoming more accurate.
    """
    def __init__(self, dim: int, activation: Callable = linear, loss: Callable = mean_squared_error):
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]

    def __repr__(self):
        return f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'

    def predict(self, xs: list) -> list:
        """
        Calculates prediction value for each instance in the list of instances given.

        Args:
            xs: List of input data/instances

        Returns:
            List with predicted (yhat) values/classes
        """
        return [self.activation(self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))) for x in xs]

    def partial_fit(self, xs: list, ys: list, *, alpha: int = 0.001) -> None:
        """
        Update/fit the model with a single iteration over the given data.

        Args:
            xs: List of input data/instances
            ys: List of target values
            alpha: Learning rate
        """
        for x, y in zip(xs, ys):
            pre_activation = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            yhat = self.activation(pre_activation)
            slope = derivative(self.loss)(yhat, y) * derivative(self.activation)(pre_activation)
            self.bias -= alpha * slope
            self.weights = [wi - alpha * slope * xi for wi, xi in zip(self.weights, x)]

    def fit(self, xs: list, ys: list, *, alpha: int = 0.001, epochs: int = 1000) -> None:
        """
        Update/fit the model with {epochs} amount of iterations over the given data.

        Args:
            xs: List of input instances
            ys: List of target values
            alpha: Learning rate
            epochs: Maximum amount of iterations/epochs the model should perform
        """
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)
