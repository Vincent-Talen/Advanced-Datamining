"""This module contains the SoftmaxLayer class for a neural network.

When a network has 2 or more output values per instance, the SoftmaxLayer can be used
to ensure that the predicted values are returned as probabilities; none below 0 with all
of them summing up to 1. This is done by applying the softmax function to the final
output values of the last/output layer.
"""
from math import exp

from overrides import override

from vlearning.layers import Layer


class SoftmaxLayer(Layer):
    """A layer that can be used for multinomial classification to return probabilities.

    Can be seen as a vector function that is applied to the output of the network.

    Attributes:
        num_inputs (int): Number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer | None): The next layer in the network.
    """
    @override
    def __init__(
        self,
        num_outputs: int,
        *,
        name: str | None = None,
        next_layer: Layer | None = None
    ):
        """Overrides parent's method to remove unneeded/unsupported parameters.

        ???

        Args:
            num_outputs: The number of outputs from the layer.

        Keyword Args:
            name: The name of the layer.
            next_layer: The next layer in the network.
        """
        super().__init__(num_outputs, name=name, next_layer=next_layer)

    @override
    def __call__(self, linear_outputs, labels=None, *, alpha=None):
        """Makes `SoftmaxLayer`s callable and implements forward- & back-propagation.

        ???

        Args:
            linear_outputs:
                A list with the final output values/labels for all instances.
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
            where each list contains the gradient of the loss to the post-activation
            values it receives from the layer before, for every label of that instance.
        """
        # Apply the softmax function to the output values of every instance
        y_hats: list[list[float]] = []
        for instance_vector in linear_outputs:
            # Calculate the exponential of each output value and their sum
            euler_vector = [exp(x - max(instance_vector)) for x in instance_vector]
            euler_output_sum = sum(euler_vector)
            # Divide each exponential by the sum to get the probabilities
            y_hats.append([value / euler_output_sum for value in euler_vector])

        # Feed forward and receive the next layer's results and back-propagation values
        _, losses, gradients = self.next_layer(y_hats, labels, alpha=alpha)

        if not alpha:
            return y_hats, losses, None

        # Calculate the gradient of the loss to the pre-activation values
        new_gradients: list[list[float]] = []
        for y_hat, gradient in zip(y_hats, gradients):
            instance_gradients = [
                sum(
                    gradient[o] * y_hat[o] * ((i == o) - y_hat[i])
                    for o in range(self.num_outputs)
                )
                for i in range(self.num_inputs)
            ]
            new_gradients.append(instance_gradients)

        return y_hats, losses, new_gradients

    @override
    def __repr__(self) -> str:
        text = f"SoftmaxLayer(num_outputs={self.num_outputs}, name='{self.name}')"
        if self.next_layer is not None:
            text += f" +\n\t{self.next_layer!r}"
        return text
