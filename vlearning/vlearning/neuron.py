"""This module contains the implementation of a generalized model of a neuron.

Typical usage example:

    my_neuron = Neuron(dim=2)
    my_neuron.fit(xs, ys)
"""
from collections.abc import Callable

from vlearning import derivative
from vlearning.activation_functions import linear
from vlearning.loss_functions import mean_squared_error


class Neuron:
    """Generalized model that computes an activation value from inputs that it receives.

    It achieves this by weighting the inputs, adding a bias and then using an
    activation function to produce a prediction as output. Using its loss function
    it can calculate the error between the value it predicts and the actual value.

    It fits/trains by iterating over the input data and then updating its weights and
    bias accordingly by the use of gradient descent, which is a method to minimize loss.

    Attributes:
        dim (int): Amount of input features.
        activation (Callable): Activation function.
        loss (Callable): Loss function.
        bias (float): Bias value.
        weights (list[float]): List with a weight for each input feature.
    """
    def __init__(
        self,
        dim: int,
        activation: Callable[[float], float] = linear,
        loss: Callable[[float, float], float] = mean_squared_error
    ):
        """
        Args:
            dim: Amount of input features.
            activation: Activation function.
            loss: Loss function.
        """
        self.dim: int = dim
        self.activation: Callable[[float], float] = activation
        self.loss: Callable[[float, float], float] = loss
        self.bias: float = 0.0
        self.weights: list[float] = [0.0 for _ in range(dim)]

    def __repr__(self) -> str:
        return (
            "Neuron("
            f"dim={self.dim}, "
            f"activation={self.activation.__name__}, "
            f"loss={self.loss.__name__}"
            ")"
        )

    def predict(self, xs: list[list[float]]) -> list[float]:
        """Calculates the predicted value for every instance of the input list.

        Args:
            xs: List of input data/instances.

        Returns:
            List with predicted (yhat) values/classes.
        """
        return [
            self.activation(
                self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            )
            for x in xs
        ]

    def partial_fit(
        self, xs: list[list[float]], ys: list[float], *, alpha: int = 0.001
    ) -> None:
        """Fit/train the neuron with a single iteration over the given data.

        Args:
            xs: List of input data/instances.
            ys: List of target values.

        Keyword Args:
            alpha: Learning rate.
        """
        for x, y in zip(xs, ys):
            # Calculate the pre-activation and the post-activation (yhat) values
            pre_activation = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            yhat = self.activation(pre_activation)

            # Get the derivative functions and calculate the slope
            activation_prime: Callable[[float], float] = derivative(self.activation)
            loss_prime: Callable[[float, float], float] = derivative(self.loss)
            slope = loss_prime(yhat, y) * activation_prime(pre_activation)

            # Update the bias and weights
            self.bias -= alpha * slope
            self.weights = [wi - alpha * slope * xi for wi, xi in zip(self.weights, x)]

    def fit(
        self,
        xs: list[list[float]],
        ys: list[float],
        *,
        alpha: int = 0.001,
        epochs: int = 1000
    ) -> None:
        """Fit/train the neuron for <epochs> amount of iterations over the given data.

        Args:
            xs: List of input instances.
            ys: List of target values.

        Keyword Args:
            alpha: Learning rate.
            epochs: Amount of iterations/epochs the model should perform for training.
        """
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)
