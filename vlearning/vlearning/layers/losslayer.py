"""This module contains the LossLayer class for a neural network.

A LossLayer instance is the last layer in a network, it thus has no outputs and no
layer after it, it does add a parameter and instance attribute for a loss function.
Using this loss function it calculates the loss of the network it is part of.
"""
from typing import Callable

from overrides import override

from . import Layer
from vlearning.loss_functions import mean_squared_error


class LossLayer(Layer):
    """The final layer in a neural network that computes the loss of the network.

    Because it is the last layer in a network it has no outputs and no layer after it.
    It does add a parameter for a loss function and stores it as an instance attribute,
    this function is then used to compute the loss of the network.

    Attributes:
        num_inputs (int): The number of inputs to the layer.
        name (str): The name of the layer.
        loss (Callable): The loss function of the layer.
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
        self.loss: Callable = loss

    @override
    def __call__(self, xs, ys=None):
        y_hats = xs

        if not ys:
            return y_hats, None, None

        # Calculate the loss for every instance for each neuron from the output layer
        losses: list[float] = [
            sum(self.loss(y_hat[i], y[i]) for i in range(self.num_inputs))
            for y_hat, y in zip(y_hats, ys)
        ]
        return y_hats, losses

    @override
    def __repr__(self) -> str:
        text = (
            f"{type(self).__name__}("
            f"num_inputs={self.num_inputs}, "
            f"name='{self.name}', "
            f"loss='{self.loss.__name__}'"
            ")"
        )

        if self.next_layer is not None:
            text += f" + {self.next_layer!r}"
        return text
