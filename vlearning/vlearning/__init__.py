# Make package version available
import importlib.metadata
__version__ = importlib.metadata.version(__package__ or __name__)

# Imports to make some functions and Classes, located within modules, directly available
from vlearning.derivative import derivative
from vlearning.neuron import Neuron
from vlearning.perceptron import Perceptron
from vlearning.linear_regression import LinearRegression


__all__ = [
    "__version__",
    "data",
    # Functions
    "derivative",
    # Classes
    "Perceptron",
    "Neuron",
    "LinearRegression",
    # Modules
    "activation_functions",
    "loss_functions",
    # Subpackages
    "layers",
]
