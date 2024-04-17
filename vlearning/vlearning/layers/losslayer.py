"""This module contains the LossLayer class for a neural network.

A LossLayer instance is the last layer in a network, it thus has no outputs and no
layer after it, it does add a parameter and instance attribute for a loss function.
Using this loss function it calculates the loss of the network it is part of.
"""
from collections.abc import Callable

from overrides import override

from vlearning import derivative
from vlearning.layers import Layer
from vlearning.loss_functions import mean_squared_error


class LossLayer(Layer):
    """The final layer in a neural network that computes the loss of the network.

    Because it is the last layer in a network it has no outputs and no layer after it.
    It does add a parameter for a loss function and stores it as an instance attribute,
    this function is then used to compute the loss of the network.

    Attributes:
        num_inputs (int): Number of inputs to the layer.
        name (str): The name of the layer.
        loss (Callable[[float, float], float]): The loss function of the layer.
        loss_prime (Callable[[float, float], float]): Derivative of the loss function.
    """
    @override
    def __init__(
        self,
        *,
        name: str | None = None,
        loss: Callable = mean_squared_error
    ):
        """Overrides parent's method to change parameter signature and attributes.

        The `num_outputs` and `next_layer` parameters have been removed since it is the
        last layer in a network and the `loss` parameter and instance attribute have
        been added to store the loss function that the instance should use.

        Keyword Args:
            name: The name of the layer.
            loss: The loss function of the layer.
        """
        super().__init__(None, name=name)
        self.loss: Callable[[float, float], float] = loss
        self.loss_prime: Callable[[float, float], float] = derivative(loss)

    @override
    def __call__(self, y_hats, labels=None, *, alpha=None):
        """Makes `LossLayer`s callable and implements forward- & back-propagation.

        This layer always returns the predicted values for the given dataset, if the
        expected labels are passed through the `labels` argument it also returns the
        loss for each instance. If the network should train, meaning that a learning
        rate was passed to the `alpha` parameter, then the gradient of the loss to the
        predicted value will also be calculated for every feature (output value) of
        each instance.

        Args:
            y_hats:
                A list with the predicted values by the network for all instances.
            labels:
                A list containing the correct label per feature of each instance if the
                loss should be returned, otherwise `None`.

        Keyword Args:
            alpha:
                The learning rate if the network should train, otherwise `None`.

        Returns:
            A tuple with 3 elements
            `(list[list[float]], list[float] | None, list[list[float]] | None)`.

            The first element is always the network's predicted values for the
            instances, the second element has the loss for each instance if `labels` is
            used, otherwise `None`. The third element is `None` if `alpha` is not used,
            otherwise it is a list that contains a list for every instance, where each
            list contains the gradient of the loss to the predicted value by the network
            for every feature (output layer neuron) of that instance.
        """
        # If no labels were given then return early with only the predicted values
        if not labels:
            return y_hats, None, None

        # Calculate the total loss for every instance
        losses: list[float] = [
            sum(self.loss(y_hat[i], y[i]) for i in range(self.num_inputs))
            for y_hat, y in zip(y_hats, labels)
        ]
        if not alpha:
            return y_hats, losses, None

        # Calculate the gradient of the loss to the predicted value
        gradients: list[list[float]] = [
            [self.loss_prime(y_hat[i], label[i]) for i in range(self.num_inputs)]
            for y_hat, label in zip(y_hats, labels)
        ]
        return y_hats, losses, gradients

    @override
    def __repr__(self) -> str:
        text = (
            f"LossLayer("
            f"num_inputs={self.num_inputs}, "
            f"name='{self.name}', "
            f"loss='{self.loss.__name__}'"
            ")"
        )

        if self.next_layer is not None:
            text += f" +\n\t{self.next_layer!r}"
        return text

    @override
    def add(self, new_layer: Layer) -> None:
        """Overrides parent's method to raise an error.

        Raises:
            TypeError: Indicates that a LossLayer can not have a layer added after it.
        """
        raise TypeError(
            "A LossLayer is a network's last layer, so it can't have a layer after it."
        )
