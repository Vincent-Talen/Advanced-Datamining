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
        """Makes `DenseLayer`s callable and implements forward- & back-propagation.

        This layer performs a weighted linear combination of the inputs it receives and
        its weights and biases to compute the pre-activation (linear output) values. It
        then passes these values to the next layer in the network, which is normally an
        ActivationLayer. If the correct labels were passed then it will simply return
        them to the next layer again. If the network should train, meaning that a
        learning rate was passed to the `alpha` parameter, then the weights and biases
        will be updated using the gradient it received of the loss to the pre-activation
        values had calculated. In case this layer is not the first hidden layer in the
        network, it will also calculate and return new gradients of the loss to the
        input values it received from the layer before.

        Args:
            xs:
                A list with values for all instances and their attributes.
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
            instances, calculated by the current (hidden) layer. The second element
            contains the loss for each instance if `labels` is used, otherwise `None`.
            The third element is `None` if `alpha` is not used, otherwise it is a list
            that contains a list for every instance, where each list contains the
            gradient of the loss to the input it receives from the layer before, for
            every feature (neuron of current layer) of that instance.
        """
        linear_outputs: list[list[float]] = []
        for x in xs:
            a = [
                self.biases[o] + sum(wi * xi for wi, xi in zip(self.weights[o], x))
                for o in range(self.num_outputs)
            ]
            linear_outputs.append(a)

        # Feed forward and receive the next layer's results and back-propagation values
        y_hats, losses, gradients = self.next_layer(linear_outputs, labels, alpha=alpha)
        if not alpha:
            return y_hats, losses, None

        # Calculate the gradients of this layer's output to the input it received
        new_gradients: list[list[float]] = []
        for gradient in gradients:
            instance_gradients: list[float] = [
                sum(self.weights[o][i] * gradient[o] for o in range(self.num_outputs))
                for i in range(self.num_inputs)
            ]
            new_gradients.append(instance_gradients)

        # Update the weights and biases based on the gradients
        scaled_alpha: float = alpha / len(xs)
        for x, gradient in zip(xs, gradients):
            for o in range(self.num_outputs):
                update_size: float = scaled_alpha * gradient[o]
                self.biases[o] -= update_size
                for i in range(self.num_inputs):
                    self.weights[o][i] -= update_size * x[i]

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
