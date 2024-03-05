"""This module contains the implementation of a perceptron that predicts number values.

Normally a perceptron predicts classes, but the LinearRegression class is an adaptation
that can predict number values.

Typical usage example:

    my_linearregression = LinearRegression(dim=2)
    my_linearregression.fit(xs, ys)
"""
from overrides import override


class LinearRegression:
    """
    Adaptation of the Perceptron where the signum activation function is replaced by
    the so-called 'identity function', meaning that the post-activation is now simply
    the pre-activation value. Because error values will now be larger during
    fitting/training, the alpha parameter has been added, which is the 'Learning Rate'.

    Attributes:
        dim (int): Amount of input features
        bias (float): Bias value
        weights (list[float]): List of weights
    """
    def __init__(self, dim: int):
        """
        Args:
            dim: Amount of input features
        """
        self.dim: int = dim
        self.bias: float = 0.0
        self.weights: list[float] = [0.0 for _ in range(dim)]

    def __repr__(self) -> str:
        return f"LinearRegression(dim={self.dim})"

    def predict(self, xs: list) -> list:
        """Calculates prediction value for each instance in the list of instances given.

        Args:
            xs: List of input data/instances

        Returns:
            List with predicted (yhat) values/classes
        """
        return [self.bias + sum(wi * xi for wi, xi in zip(self.weights, x)) for x in xs]

    def partial_fit(self, xs: list, ys: list, *, alpha: int = 0.01) -> None:
        """Fit/train the model with a single iteration over the given data.

        Args:
            xs: List of input data/instances
            ys: List of target values

        Keyword Args:
            alpha: Learning rate
        """
        for x, y in zip(xs, ys):
            yhat = self.predict([x])[0]
            error = yhat - y
            self.bias -= alpha * error
            self.weights = [wi - alpha * error * xi for wi, xi in zip(self.weights, x)]

    def fit(self, xs: list, ys: list, *, alpha: int = 0.01, epochs: int = 1000) -> None:
        """Fit/train the model with {epochs} amount of iterations over the given data.

        Args:
            xs: List of input instances
            ys: List of target values

        Keyword Args:
            alpha: Learning rate
            epochs: Maximum amount of iterations/epochs the model should perform
        """
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)
