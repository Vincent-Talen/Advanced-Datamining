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
    DenseLayer: The first of two layer representing a hidden layer of a neural network.
    ActivationLayer: The second layer class of a hidden layer in a neural network.
    SoftmaxLayer: Applies softmax activation function to the neural network's outputs.
    LossLayer: A layer representing a loss function in a neural network.

Typical usage example:
    from vlearning import activation_functions, layers

    my_network = (
        layers.InputLayer(2) +
        layers.DenseLayer(5, name='Hidden') +
        layers.ActivationLayer(5, activation=activation_functions.tanh) +
        layers.DenseLayer(4, name='Output') +
        layers.SoftmaxLayer() +
        layers.LossLayer()
    )
    my_network.fit(xs, ys, alpha=0.1, epochs=200)
    y_hats = my_network.predict(xs)
"""
from .layer import Layer
from .inputlayer import InputLayer
from .denselayer import DenseLayer
from .activationlayer import ActivationLayer
from .softmaxlayer import SoftmaxLayer
from .losslayer import LossLayer

__all__ = [
    "Layer",
    "InputLayer",
    "DenseLayer",
    "ActivationLayer",
    "SoftmaxLayer",
    "LossLayer",
]
