"""This module contains the DenseLayer class for a neural network.

A DenseLayer is one of two layer class that make up a hidden layer in a network. It
initializes and stores the weights and biases and uses these to compute the
pre-activation values, after which it is normally combined with an ActivationLayer that
applies its activation function to create the post-activation value.
"""
import random
from math import sqrt
from typing import TypeAlias

from overrides import override

from . import Layer

DataInstanceValues: TypeAlias = list[float]


class DenseLayer(Layer):
    """The first half of a hidden layer and computes the pre-activation values.

    In this package a hidden layer is made up out of two classes, this is the first
    class and the second class is normally the ActivationLayer. A DenseLayer instance
    computes the pre-activation values using the weights and biases it stores, after
    which, normally the ActivationLayer applies the activation function to get the
    post-activation values.

    Attributes:
        num_inputs (int): The number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer | None): The next layer in the network.
        biases (list[float]): The biases of the neurons in the layer.
        weights (list[list[float]] | None): A list with the weights of each neuron.
    """
    @override
    def __init__(
        self,
        num_outputs: int,
        *,
        name: str | None = None,
        next_layer: Layer | None = None
    ):
        """Overrides the parent Layer class to also initialize weights and biases.

        Because this layer computes the pre-activation values, it also needs to store
        and initialize the weights and biases used to compute these values. There is
        however no difference in the parameters/arguments having to be passed.

        The biases are immediately initialized to 0.0 but the weights are only
        initialized by the _set_inputs method when it is called.

        Args:
            num_outputs: The number of outputs from the layer.

        Keyword Args:
            name: The name of the layer.
            next_layer: The next layer in the network.
        """
        super().__init__(num_outputs, name=name, next_layer=next_layer)
        self.biases: list[float] = [0.0 for _ in range(self.num_outputs)]
        self.weights: list[list[float]] | None = None

    @override
    def __call__(self, xs: list[DataInstanceValues]) -> list[DataInstanceValues]:
        aa: list[DataInstanceValues] = []
        for x in xs:
            a: DataInstanceValues = []
            for o in range(self.num_outputs):
                # For this neuron calculate the pre-activation values for the instances
                a.append(
                    self.biases[o] + sum(wi * xi for wi, xi in zip(self.weights[o], x))
                )
            aa.append(a)
        y_hats: list[DataInstanceValues] = self.next_layer(aa)
        return y_hats

    @override
    def _set_inputs(self, num_inputs: int) -> None:
        """Sets the number of inputs for the layer and initializes the weights.

        This method overrides the parent method to also initialize the weights, whose
        initial values are set using the Normalized Xavier Initialization method.

        Args:
            num_inputs: The number of inputs.
        """
        self.num_inputs = num_inputs
        limit = sqrt(6 / (num_inputs + self.num_outputs))
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(self.num_inputs)]
            for _ in range(self.num_outputs)
        ]
