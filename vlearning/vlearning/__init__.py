from vlearning.perceptron import Perceptron
from vlearning.linear_regression import LinearRegression

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)
__all__ = ["__version__", "data", "Perceptron", "LinearRegression"]
