"""This module contains the ActivationLayer class for a neural network.

The ActivationLayer is the second class of two that make up a hidden layer in a network.
It adds a parameter for an activation function and stores it as an instance attribute,
this function is then used to compute the post-activation values from the pre-activation
values computed by the DenseLayer instance that comes before it.
"""
from typing import Callable

from overrides import override

from . import Layer
from vlearning.activation_functions import linear


class ActivationLayer(Layer):
    """The second half of a hidden layer and computes the pre-activation values.

    In this package a hidden layer is made up out of two classes, the first is the
    DenseLayer class and this class is normally the second one. After the pre-activation
    values have been calculated by a DenseLayer instance, an instance of this class
    applies its activation function to calculate the post-activation values.

    Attributes:
        num_inputs (int): The number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer | None): The next layer in the network.
        activation (Callable): The activation function of the layer.
    """
    @override
    def __init__(
        self,
        num_outputs: int,
        *,
        name: str | None = None,
        next_layer: Layer | None = None,
        activation: Callable = linear
    ):
        """Overrides parent's method to add an activation function parameter+attribute.

        The goal for this layer is to apply an activation function to the pre-activation
        values, it thus has an extra `activation` parameter to accept one and stores the
        passed function to the corresponding `activation` attribute of the instance.

        Args:
            num_outputs: The number of outputs from the layer.

        Keyword Args:
            name: The name of the layer.
            next_layer: The next layer in the network.
            activation: The activation function of the layer.
        """
        super().__init__(num_outputs, name=name, next_layer=next_layer)
        self.activation: Callable = activation

    @override
    def __call__(self, xs, ys=None, *, alpha=None):
        hh = []
        for x in xs:
            h = [self.activation(x[o]) for o in range(self.num_outputs)]
            hh.append(h)

        y_hats, losses, gradients = self.next_layer(hh, ys, alpha=alpha)

        if not alpha:
            return y_hats, losses, None

        new_gradients = []
        return y_hats, losses, new_gradients

    @override
    def __repr__(self) -> str:
        text = (
            f"{type(self).__name__}("
            f"num_outputs={self.num_outputs}, "
            f"name='{self.name}', "
            f"activation='{self.activation.__name__}'"
            ")"
        )
        if self.next_layer is not None:
            text += f" + {self.next_layer!r}"
        return text
