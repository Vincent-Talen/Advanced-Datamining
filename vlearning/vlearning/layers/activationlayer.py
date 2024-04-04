"""This module contains the ActivationLayer class for a neural network.

The ActivationLayer is the second class of two that make up a hidden layer in a network.
It adds a parameter for an activation function and stores it as an instance attribute,
this function is then used to compute the post-activation values from the pre-activation
values computed by the DenseLayer instance that comes before it.
"""
from collections.abc import Callable

from overrides import override

from vlearning import derivative
from vlearning.activation_functions import linear
from vlearning.layers import Layer


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
        activation_prime (Callable): The derivative of the activation function.
    """
    @override
    def __init__(
        self,
        num_outputs: int,
        *,
        name: str | None = None,
        next_layer: Layer | None = None,
        activation: Callable[[float], float] = linear
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
        self.activation: Callable[[float], float] = activation
        self.activation_prime: Callable[[float], float] = derivative(activation)

    @override
    def __call__(self, xs, labels=None, *, alpha=None):
        """Makes `ActivationLayer`s callable and implements forward- & back-propagation.

        An ActivationLayer applies the activation function it got assigned to the
        pre-activation values it receives from the previous (dense) layer. The results
        are the outputs of the current hidden layer this instance is part of, when it is
        the last layer in a network then the outputs are considered the final
        predictions. If the correct labels were passed then it will simply return them
        to the next layer again. If the network should train, then all the gradients of
        the loss to the pre-activation values will also be calculated.

        Args:
            xs:
                A list with a pre-activation value per neuron for all instances.
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
            instances, the second element contains the loss for each instance if
            `labels` is used, otherwise `None`. The third element is `None` if `alpha`
            is not used, otherwise it is a list that contains a list for every instance,
            where each list contains the gradient of the loss to the pre-activation
            values it receives from the layer before it, for every feature (neuron of
            current layer) of that instance.
        """
        # Apply the activation function to each neuron's value for every instance
        hs: list[list[float]] = [
            [self.activation(x[o]) for o in range(self.num_outputs)]
            for x in xs
        ]

        # Feed forward and receive the next layer's results and back-propagation values
        y_hats, losses, gradients = self.next_layer(hs, labels, alpha=alpha)

        if not alpha:
            return y_hats, losses, None

        # Calculate the gradient of the loss to the pre-activation values
        new_gradients: list[list[float]] = [
            [self.activation_prime(a[i]) * g[i] for i in range(self.num_inputs)]
            for a, g in zip(xs, gradients)
        ]
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
            text += f" +\n\t{self.next_layer!r}"
        return text
