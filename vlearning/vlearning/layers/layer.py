from collections import Counter
from copy import deepcopy


class Layer:
    """
    Base implementation of a layer for a neural network, only defines layer structure.

    Attributes:
        num_inputs (int): The number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer): The next layer in the network.
    """
    _counter: Counter = Counter()
    """Counter: Counter keeping track of the number of instances of each layer class."""

    def __init__(self, num_outputs: int, *, name: str = None, next_layer=None):
        """
        Parameters:
            num_outputs (int): The number of outputs from the layer.
            name (str, optional): The name of the layer. Default = None.
            next_layer (Layer, optional): The next layer in the network. Default = None.
        """
        Layer._counter[type(self)] += 1
        self.num_inputs: int = 0
        self.num_outputs: int = num_outputs
        self.name: str = name or f"{type(self).__name__}_{Layer._counter[type(self)]}"
        self.next_layer = next_layer

    def __add__(self, new_layer):
        """
        Overloads the '+' operator to add a new layer to the network.
        NOTE: The instances of the previous layers and the new layer are
        not used as references, but are newly instantiated as deep copies.

        Parameters:
            new_layer (Layer): The new layer to be added after the current layer.

        Returns:
            Layer: An updated network with the new layer added.
        """
        result = deepcopy(self)
        result.add(deepcopy(new_layer))
        return result

    def __getitem__(self, index: [int | str]):
        """
        Overloads the '[]' operator to get a layer by its index or name.
        Searches for a layer matching the index by recursively
        calling this method on layers down the chain.

        Parameters:
            index (int | str): The index or name of the layer.

        Returns:
            Layer: The layer at the given index or with the given name.

        Raises:
            IndexError: If the index is out of range.
            KeyError: If the name does not exist.
            TypeError: If the index is not an integer or string.
        """
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next_layer is None:
                raise IndexError("Layer index out of range")
            return self.next_layer[index - 1]
        if isinstance(index, str):
            if self.next_layer is None:
                raise KeyError(index)
            return self.next_layer[index]
        raise TypeError(
            f"Layer indices must be integers or strings, not {type(index).__name__}"
        )

    def __iadd__(self, new_layer):
        """
        Overloads the '+=' operator to add a new layer to the network in-place.

        Parameters:
            new_layer (Layer): The new layer to be added after the current layer.

        Returns:
            Layer: The updated network with the new layer added.
        """
        self.add(new_layer)
        return self

    def __iter__(self):
        """
        Makes the Layer instances iterable.

        Yields:
            Layer: The current layer in the network.
        """
        current_layer = self
        while current_layer is not None:
            yield current_layer
            current_layer = current_layer.next_layer

    def __len__(self):
        """
        Returns the number of layers in the network.

        Returns:
            int: The number of layers.
        """
        count = 1
        current_layer = self.next_layer
        while current_layer is not None:
            count += 1
            current_layer = current_layer.next_layer
        return count

    def __repr__(self):
        text = (
            f"{type(self).__name__}("
            f"num_inputs={self.num_inputs}, "
            f"num_outputs={self.num_outputs}, "
            f"name={repr(self.name)}"
            ")"
        )
        if self.next_layer is not None:
            text += f" + {repr(self.next_layer)}"
        return text

    def _set_inputs(self, num_inputs: int):
        """
        Sets the number of inputs for the layer.

        Parameters:
            num_inputs (int): The number of inputs.
        """
        self.num_inputs = num_inputs

    def add(self, new_layer):
        """
        Adds a new layer at the end of the network.

        Parameters:
            new_layer (Layer): The new layer to be added.
        """
        if self.next_layer is None:
            self.next_layer = new_layer
            new_layer._set_inputs(self.num_outputs)
        else:
            self.next_layer.add(new_layer)
