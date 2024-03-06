"""This module contains the InputLayer class for a neural network.

An InputLayer is a special type of layer that is the first layer in a network and
the only layer instance you actually interact with when using the network. Meaning
it is the only layer that you can call the `predict` and `fit` methods on.
"""
from overrides import override

from . import Layer


class InputLayer(Layer):
    @override
    def __repr__(self) -> str:
        text = f"InputLayer(num_outputs={self.num_outputs}, name={repr(self.name)})"
        if self.next_layer is not None:
            text += f" + {repr(self.next_layer)}"
        return text

    @override
    def _set_inputs(self, num_inputs: int) -> None:
        raise TypeError("An InputLayer does not have, or accept, inputs.")

