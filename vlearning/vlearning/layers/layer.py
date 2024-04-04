"""This module contains the base implementation of a layer for a neural network.

The Layer class is not meant to be actually used as a layer in a network, but
is instead meant to be subclassed by other layer classes. It defines the basic
structure of a layer and provides methods for adding layers to a network and
accessing layers by index or name.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy


class Layer:
    """Base implementation of a neural network layer only defining network structure.

    The Layer class is not meant to be actually used as a layer in a network, but
    is instead meant to be subclassed by other layer classes. It defines the basic
    structure of a layer and provides methods for adding layers to a network and
    accessing layers by index or name.

    Because in some earlier notebooks of the course the Layer class is instantiated,
    it can not be made an abstract class that inherits from the `abc.ABC` class.
    In turn the `@abc.abstractmethod` decorator is not used for methods intended to be
    abstract, but the methods are instead implemented to raise a `NotImplementedError`.

    Attributes:
        num_inputs (int): The number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer | None): The next layer in the network.
    """
    _counter: Counter = Counter()
    """Counter: Counter keeping track of the number of instances of each layer class."""

    def __init__(
        self,
        num_outputs: int | None,
        *,
        name: str | None = None,
        next_layer: Layer | None = None
    ):
        """
        Args:
            num_outputs: The number of outputs from the layer.

        Keyword Args:
            name: The name of the layer.
            next_layer: The next layer in the network.
        """
        Layer._counter[type(self)] += 1
        self.num_inputs: int | None = None
        self.num_outputs: int | None = num_outputs
        self.name: str = name or f"{type(self).__name__}_{Layer._counter[type(self)]}"
        self.next_layer: Layer | None = next_layer

    def __add__(self, new_layer: Layer) -> Layer:
        """Implements the '+' operator to add a new layer to the network.

        The instances of the previous layers and the new layer are
        not used as references, but are newly instantiated as deep copies.

        Args:
            new_layer: The new layer to be added after the current layer.

        Returns:
            An updated network with the new layer added.
        """
        result = deepcopy(self)
        result.add(deepcopy(new_layer))
        return result

    def __call__(
        self,
        xs: list[list[float]],
        labels: list[list[float]] = None,
        *,
        alpha: float = None
    ) -> tuple[list[list[float]], list[float] | None, list[list[float]] | None]:
        """Makes layer instances callable, used for forward- and back-propagation.

        This method is meant to be overridden by subclasses and each subclass
        needs to implement its own version of this method.

        When an instance is called it performs its part of the calculations for
        forward-propagation and backpropagation and then calls the next layer in the
        network with these results. Every instance also passes the `labels` argument
        to the next layer until the LossLayer is reached, which then returns the
        loss of the network. If `alpha` is passed then the network will train by
        updating the weights and biases of DenseLayer instances, in order to do this
        every other layer will calculate and return the gradients of the loss to the
        values it receives from the layer before.

        Args:
            xs:
                The instances the network should predict values for.
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
            instances, the second element has the loss for each instance if `labels` is
            used, otherwise `None`. The third element is `None` if `alpha` is not used,
            otherwise it is a list that contains a list for every instance, where each
            list contains the gradient of the loss to the input it receives from the
            layer before, for every feature (neuron of current layer) of that instance.
        """
        raise NotImplementedError("Abstract __call__ method")

    def __getitem__(self, index: int | str) -> Layer:
        """Implements the '[]' operator to get a layer by its index or name.

        Searches for a layer matching the index by recursively
        calling this method on layers down the chain.

        Args:
            index: The index or name of the layer.

        Returns:
            The layer at the given index or with the given name.

        Raises:
            IndexError: If the index is out of range.
            KeyError: If the name does not exist.
            TypeError: If the index is not an integer or string.
        """
        if index in (0, self.name):
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

    def __iadd__(self, new_layer: Layer) -> Layer:
        """Implements the '+=' operator to add a new layer to the network in-place.

        Args:
            new_layer: The new layer to be added after the current layer.

        Returns:
            The updated network with the new layer added.
        """
        self.add(new_layer)
        return self

    def __iter__(self) -> Layer:
        """Makes the Layer instances iterable.

        Yields:
            The current layer in the network.
        """
        current_layer = self
        while current_layer is not None:
            yield current_layer
            current_layer = current_layer.next_layer

    def __len__(self) -> int:
        """Returns the number of layers in the network.

        Returns:
            The number of layers.
        """
        count = 1
        current_layer = self.next_layer
        while current_layer is not None:
            count += 1
            current_layer = current_layer.next_layer
        return count

    def __repr__(self) -> str:
        text = (
            f"{type(self).__name__}("
            f"num_outputs={self.num_outputs}, "
            f"name='{self.name}'"
            ")"
        )
        if self.next_layer is not None:
            text += f" +\n\t{self.next_layer!r}"
        return text

    def _set_inputs(self, num_inputs: int) -> None:
        """Sets the number of inputs for the layer.

        Args:
            num_inputs: The number of inputs.
        """
        self.num_inputs = num_inputs

    def add(self, new_layer: Layer) -> None:
        """Adds a new layer at the end of the network.

        Args:
            new_layer: The new layer to be added.
        """
        if self.next_layer is None:
            self.next_layer = new_layer
            new_layer._set_inputs(self.num_outputs)
        else:
            self.next_layer.add(new_layer)
