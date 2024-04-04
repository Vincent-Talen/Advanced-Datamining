"""This module contains the DenseLayer class for a neural network.

A DenseLayer is one of two layer class that make up a hidden layer in a network. It
initializes and stores the weights and biases and uses these to compute the
pre-activation values, after which it is normally combined with an ActivationLayer that
applies its activation function to create the post-activation value.
"""
import random
from math import sqrt

from overrides import override

from vlearning.layers import Layer


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
    def __call__(self, xs, labels=None, *, alpha=None):
        aa: list[list[float]] = []
        for x in xs:
            a = [
                self.biases[o] + sum(wi * xi for wi, xi in zip(self.weights[o], x))
                for o in range(self.num_outputs)
            ]
            aa.append(a)

        # Feed forward and receive the next layer's results and back-propagation values
        y_hats, losses, gradients = self.next_layer(aa, labels, alpha=alpha)
        if not alpha:
            return y_hats, losses, None

        scaled_alpha = alpha / len(xs)
        new_gradients: list[list[float]] = []
        for n, x in enumerate(xs):
            instance_gradients: list[float] = []
            for i in range(self.num_inputs):
                neuron_in_gradient: float = 0.0
                for o in range(self.num_outputs):
                    neuron_out_gradient = gradients[n][o]
                    neuron_in_gradient += self.weights[o][i] * neuron_out_gradient
                    self.biases[o] -= scaled_alpha * neuron_out_gradient
                    self.weights[o][i] -= scaled_alpha * neuron_out_gradient * x[i]
                instance_gradients.append(neuron_in_gradient)
            new_gradients.append(instance_gradients)

        return y_hats, losses, new_gradients

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
