"""vlearning: A Python package for building and training models and neural networks.

This package was created as part of the assignment for a bioinformatics course on
neural networks and deep learning. The way the neural networks are structured as layers
is done to follow the design specifications of the course, and the implementation is
based on the mathematical concepts taught in the course.

Functions:
    derivative: Create a numerical derivative function of a function.
    pseudo_log: Pseudo logarithmic function for values close to, or below, 0.

Classes:
    Perceptron: A single layer neural network.
    Neuron: A single neuron with a single input and single output.
    LinearRegression: A linear regression model.

Modules:
    activation_functions: Contains various activation functions.
    loss_functions: Contains various loss functions.

Subpackages:
    layers: Contains various layer types for building neural networks.
"""
# Make package version available
import importlib.metadata
__version__ = importlib.metadata.version(__package__ or __name__)

# Imports to make some functions and Classes, located within modules, directly available
from vlearning.epochtracker import EpochTracker
from vlearning.derivative import derivative
from vlearning.pseudo_log import pseudo_log
from vlearning.neuron import Neuron
from vlearning.perceptron import Perceptron
from vlearning.linear_regression import LinearRegression


__all__ = [
    "__version__",
    "data",
    # Functions
    "derivative",
    "pseudo_log",
    # Classes
    "Perceptron",
    "Neuron",
    "LinearRegression",
    "EpochTracker",
    # Modules
    "activation_functions",
    "loss_functions",
    # Subpackages
    "layers",
]
