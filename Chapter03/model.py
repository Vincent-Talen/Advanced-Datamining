from collections import Counter
from copy import deepcopy


class Layer:
    layer_counter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.layer_counter[type(self)] += 1
        if name is None:
            name = f"{type(self).__name__}_{Layer.layer_counter[type(self)]}"
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self):
        text = f"Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})"
        if self.next is not None:
            text += " + " + repr(self.next)
        return text

    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        # If the key requests is this layer (self), return self
        if index == 0 or index == self.name:
            return self
        # If the requested layer is not this layer, continue to the next layer in the chain
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')
    
    def __len__(self):
        return sum(Layer.layer_counter.values())
        # if self.next is None:
        #     return 1
        # else:
        #     return 1 + len(self.next)

    def add(self, next):
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs


class InputLayer(Layer):
    def __repr__(self):
        text = f"InputLayer(outputs={self.outputs}, name={repr(self.name)})"
        if self.next is not None:
            text += " + " + repr(self.next)
        return text


class DenseLayer(Layer):
    def __repr__(self):
        text = f"DenseLayer(outputs={self.outputs}, name={repr(self.name)})"
        if self.next is not None:
            text += " + " + repr(self.next)
        return text


class ActivationLayer(Layer):
    def __repr__(self):
        text = f"ActivationLayer(outputs={self.outputs}, name={repr(self.name)})"
        if self.next is not None:
            text += " + " + repr(self.next)
        return text
