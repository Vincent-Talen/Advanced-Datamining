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

    The SoftmaxLayer is can be seen as a vector function that is applied to the output
    of the network to convert the values to probabilities. This is done by applying the
    softmax function to the output values of every instance.

    Attributes:
        num_inputs (int): Number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer | None): The next layer in the network.
    """
    @override
    def __call__(self, linear_outputs, labels=None, *, alpha=None):
        """Makes `SoftmaxLayer`s callable and implements forward- & back-propagation.

        To prevent numerical under- and overflow every value is subtracted by the
        maximum value of the instance. Only after this correction the exponential is
        calculated for each value and then the sum of them for the current instance.
        Using the exponential values and their sum the probabilities are calculated by
        performing the softmax formula.

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
            max_value = max(instance_vector)
            euler_vector = [exp(x - max_value) for x in instance_vector]
            euler_output_sum = sum(euler_vector)
            # Divide each exponential by the sum to get the probabilities
            y_hats.append([value / euler_output_sum for value in euler_vector])

        # Feed forward and receive the next layer's results and back-propagation values
        _, losses, gradients = self.next_layer(y_hats, labels, alpha=alpha)

        if not alpha:
            return y_hats, losses, None

        # Calculate the gradient of the loss to the predicted probabilities
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
