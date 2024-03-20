"""Module defining various loss functions.

Available loss functions:
    - mean_squared_error
    - mean_absolute_error
    - hinge
"""

__all__ = ["mean_squared_error", "mean_absolute_error", "hinge"]


def mean_squared_error(yhat: float, y: float) -> float:
    """Mean Squared Error Loss-function

    Args:
        yhat: The predicted classification of the instance.
        y: The actual classification of the instance.

    Returns:
        The loss (error) of the instance.
    """
    return (yhat - y) ** 2


def mean_absolute_error(yhat: float, y: float) -> float:
    """Mean Absolute Error Loss-function

    Args:
        yhat: The predicted classification of the instance.
        y: The actual classification of the instance.

    Returns:
        The loss (error) of the instance.
    """
    return abs(yhat - y)


def hinge(yhat: float, y: float) -> float:
    """Hinge Loss-function

    Args:
        yhat: The predicted classification of the instance.
        y: The actual classification of the instance.

    Returns:
        The loss (error) of the instance.
    """
    return max(1 - yhat * y, 0)
