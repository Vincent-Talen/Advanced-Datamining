"""This module contains an implementation of Rosenblatt's Perceptron.

Typical usage example:

    my_perceptron = Perceptron(dim=2)
    my_perceptron.fit(xs, ys)
"""


class Perceptron:
    """Implementation of Rosenblatt's Perceptron, a binary single neuron model.

    It can solve linear classification problems through simple learning algorithms
    and is considered the first generation of neural networks.

    Weights are used for inputs which are obtained during the training stage,
    after performing addition between the input values and weights
    it uses the signum activation function to return the predicted classification.

    Attributes:
        dim (int): Amount of input features.
        bias (float): Bias value.
        weights (list[float]): List of weights.
        fitted (bool): Whether the model has been fully fitted.
    """
    def __init__(self, dim: int):
        """
        Args:
            dim: The amount of input features the perceptron should work on.
        """
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.fitted = False

    def __repr__(self) -> str:
        return f"Perceptron(dim={self.dim})"

    def predict(self, xs: list[list[float]]) -> list[float]:
        """Calculates prediction value for each instance in the list of instances given.

        Args:
            xs: List of input data/instances.

        Returns:
            List with predicted (yhat) values/classes.
        """
        y_hats = []
        for x in xs:
            # Compute pre-activation value
            a = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            # Save post-activation value using signum activation function
            y_hats.append(1.0 if a > 0.0 else -1.0 if a < 0.0 else 0.0)
        return y_hats

    def partial_fit(self, xs: list[list[float]], ys: list[float]) -> None:
        """Update/fit the model with a single iteration over the given data.

        Args:
            xs: List of input data/instances.
            ys: List of target values.
        """
        model_updated = False
        for x, y in zip(xs, ys):
            # Get predicted value (yhat)
            yhat = self.predict([x])[0]

            # Update bias and weights if the error is not 0
            if error := yhat - y:
                model_updated = True
                self.bias -= error
                self.weights = [wi - error * xi for wi, xi in zip(self.weights, x)]

        if not model_updated:
            self.fitted = True

    def fit(self, xs: list[list[float]], ys: list[float], *, epochs: int = 0) -> None:
        """Update/fit the model with <epochs> amount of iterations over the given data.

        When epochs=0, iteration will continue until the model is fully fitted.

        Args:
            xs: List of input instances.
            ys: List of target values.

        Keyword Args:
            epochs: Amount of iterations/epochs the model should update itself for.
        """
        epochs_completed = 0
        while not self.fitted and (epochs == 0 or epochs_completed < epochs):
            self.partial_fit(xs, ys)
            epochs_completed += 1
        if self.fitted:
            print(f"Model has been fully fitted after {epochs_completed} epochs")
        else:
            print(f"Model after fitting for {epochs_completed} epochs")
