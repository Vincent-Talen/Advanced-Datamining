from typing import Callable

from vlearning import derivative
from vlearning.activation_functions import linear
from vlearning.loss_functions import mean_squared_error


class Neuron:
    """
    The Neuron class is a generalized model that computes an activation value from
    inputs that it receives. It achieves this by weighting the inputs, adding bias,
    using an activation function and a loss functions.

    It trains by iterating over data and updating its weights and bias accordingly
    through the use of gradient descent, which is a method to minimize loss.
    """
    def __init__(
        self,
        dim: int,
        activation: Callable = linear,
        loss: Callable = mean_squared_error
    ):
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]

    def __repr__(self):
        return (
            "Neuron("
            f"dim={self.dim}, "
            f"activation={self.activation.__name__}, "
            f"loss={self.loss.__name__}"
            ")"
        )

    def predict(self, xs: list) -> list:
        """
        Calculates prediction value for each instance in the list of instances given.

        Args:
            xs: List of input data/instances

        Returns:
            List with predicted (yhat) values/classes
        """
        return [
            self.activation(
                self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            )
            for x in xs
        ]

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
            loss_derivative = derivative(self.loss)
            activation_derivative = derivative(self.activation)
            slope = loss_derivative(yhat, y) * activation_derivative(pre_activation)
            self.bias -= alpha * slope
            self.weights = [wi - alpha * slope * xi for wi, xi in zip(self.weights, x)]

    def fit(
        self, xs: list, ys: list, *, alpha: int = 0.001, epochs: int = 1000
    ) -> None:
        """
        Update/fit the model for <epochs> amount of iterations over the given data.

        Args:
            xs: List of input instances
            ys: List of target values
            alpha: Learning rate
            epochs: Amount of iterations/epochs the model should perform for training
        """
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)
