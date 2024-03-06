"""A subpackage containing various layer classes for building neural networks.

The Layer class is the base implementation of a layer for a neural network, and
sort of acts as an abstract class the other Layer classes inherit from. Because of
the design specifications of the course, the hidden layers of a network are split up
into different classes, namely the DenseLayer and ActivationLayer classes. As the name
DenseLayer suggests this layer is densely- or fully-connected, meaning that each neuron
in the layer is connected to all neurons in the previous layer.

To actually build a neural network, the first layer should always be an instance of
the InputLayer class, after that you can configure the network as you see fit by adding
any of the other provided layer classes using activation- and loss functions from the
respective modules they are defined in.

Classes:
    Layer: The base implementation of a layer for a neural network.
    InputLayer: The first layer of a neural network.
    DenseLayer: A layer representing a dense layer of a neural network.
    ActivationLayer: A layer representing an activation function in a neural network.
    LossLayer: A layer representing a loss function in a neural network.

Typical usage example:
    from vlearning import activation_functions

    my_network = (
        InputLayer(2, name='Input') +
        DenseLayer(5, name='Dense') +
        ActivationLayer(5, activation=activation_functions.tanh, name='Activation') +
        DenseLayer(1, name='Output') +
        LossLayer(name='Loss') +
    )
    my_network.fit(xs, ys, alpha=0.1, epochs=200)
    y_hats = my_network.predict(xs)
"""
from .layer import Layer
from .inputlayer import InputLayer

__all__ = [
    "Layer",
    "InputLayer",
]
