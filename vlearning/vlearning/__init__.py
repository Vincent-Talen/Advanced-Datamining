import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)
__all__ = ["__version__", "data"]