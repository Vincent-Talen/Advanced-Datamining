"""This module contains the InputLayer class for a neural network.

An InputLayer is a special type of layer that is the first layer in a network and
the only layer instance you actually interact with when using the network. Meaning
it is the only layer that you can call the `predict` and `fit` methods on.
"""
from overrides import override

from vlearning.layers import Layer


class InputLayer(Layer):
    """The first layer in a neural network, thus does not have any inputs.

    It is a special type of layer that is the first layer in a network and the only
    layer instance you actually interact with when using the network. Meaning
    it is the only layer that you can call the `predict` and `fit` methods on.

    Attributes:
        num_inputs (int): The number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer | None): The next layer in the network.
        training_history (dict[str, list[float]]): Dictionary storing training history
    """
    @override
    def __init__(
        self,
        num_outputs: int,
        *,
        name: str | None = None,
        next_layer: Layer | None = None,
    ):
        """Overrides parent's method to add a training_history instance attribute.

        To keep track of the loss, accuracy, etc. of the network during training,
        the `training_history` attribute can be used and accessed by the fit methods.

        Args:
            num_outputs: The number of outputs from the layer.

        Keyword Args:
            name: The name of the layer.
            next_layer: The next layer in the network.
        """
        super().__init__(num_outputs, name=name, next_layer=next_layer)
        self.training_history = {"loss": []}

    @override
    def __call__(self, xs, labels=None, *, alpha=None):
        return self.next_layer(xs, labels, alpha=alpha)

    @override
    def _set_inputs(self, num_inputs: int) -> None:
        raise TypeError("An InputLayer does not have, nor accept, inputs.")

    def predict(self, xs: list[list[float]]) -> list[list[float]]:
        """Get the predicted values for the given dataset.

        This method is an accessibility/ease-of-use wrapper that simply calls
        the instance itself, returning the predicted values of the network.

        Args:
            xs: The instances the network should predict values for.

        Returns:
            The predicted values for the given dataset.
        """
        y_hats, _, _ = self(xs)
        return y_hats

    def evaluate(self, xs: list[list[float]], labels: list[list[float]]) -> float:
        """Get the mean loss of the network for the instances of the given dataset.

        Args:
            xs: The instances the network should predict values for.
            labels: A list containing the correct labels for all instances

        Returns:
            The mean loss of the network for the instances of the given dataset.
        """
        _, losses, _ = self(xs, labels)
        return sum(losses) / len(losses)

    def partial_fit(
        self, xs: list[list[float]], labels: list[list[float]], *, alpha: float = 0.001
    ) -> None:
        """Fit/train the network to the given dataset for a single epoch.

        After training this one epoch the loss of the network is calculated and stored
        in the `training_history` instance attribute.

        Args:
            xs:
                The instances the network should predict values for.
            labels:
                A list containing the correct label per feature of each instance if the
                loss should be returned, otherwise `None`.

        Keyword Args:
            alpha:
                The learning rate of the network.
        """
        _, losses, _ = self(xs, labels, alpha=alpha)
        self.training_history["loss"].append(sum(losses) / len(losses))

    def fit(
        self,
        xs: list[list[float]],
        labels: list[list[float]] = None,
        *,
        alpha: float = 0.001,
        epochs: int = 100
    ) -> dict[str, list[float]]:
        """Fit/train the network to the given dataset for a given number of epochs.

        Args:
            xs: The instances the network should predict values for.
            labels: A list containing the correct labels for all instances if
                the loss should be returned, otherwise `None`.

        Keyword Args:
            alpha: The learning rate of the network.
            epochs: The number of epochs to train the network for.

        Returns:
            A dictionary containing the training history of the network.
        """
        for _ in range(epochs):
            self.partial_fit(xs, labels, alpha=alpha)
        return self.training_history
