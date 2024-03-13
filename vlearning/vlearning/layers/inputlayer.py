"""This module contains the InputLayer class for a neural network.

An InputLayer is a special type of layer that is the first layer in a network and
the only layer instance you actually interact with when using the network. Meaning
it is the only layer that you can call the `predict` and `fit` methods on.
"""
from typing import TypeAlias

from overrides import override

from . import Layer

DataInstanceValues: TypeAlias = list[float]


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
    """
    @override
    def __call__(self, xs: list[DataInstanceValues]) -> list[DataInstanceValues]:
        y_hats: list[DataInstanceValues] = self.next_layer(xs)
        return y_hats

    @override
    def _set_inputs(self, num_inputs: int) -> None:
        raise TypeError("An InputLayer does not have, nor accept, inputs.")

    def predict(self, xs: list[DataInstanceValues]) -> list[DataInstanceValues]:
        """Get the predicted values for the given dataset.

        This method is an accessibility/ease-of-use wrapper that simply calls
        the instance itself, returning the predicted values of the network.

        Args:
            xs: The data the layer should predict values for.

        Returns:
            The predicted values for the given dataset.
        """
        y_hats = self(xs)
        return y_hats
