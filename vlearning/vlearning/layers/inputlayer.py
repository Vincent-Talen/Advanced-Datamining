"""This module contains the InputLayer class for a neural network.

An InputLayer is a special type of layer that is the first layer in a network and
the only layer instance you actually interact with when using the network. Meaning
it is the only layer that you can call the `predict` and `fit` methods on.
"""
from overrides import override

from . import Layer


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
    def _set_inputs(self, num_inputs: int) -> None:
        raise TypeError("An InputLayer does not have, nor accept, inputs.")

