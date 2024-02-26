class LinearRegression:
    """
    Adaptation of the Perceptron where the signum activation function is replaced by
    the so-called 'identity function', where the post-activation is now just the pre-activation value.
    Because the error values will be larger, alpha, aka the 'Learning Rate', is introduced for fitting.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]

    def __repr__(self):
        return f"LinearRegression(dim={self.dim})"

    def predict(self, xs: list) -> list:
        """
        Calculates prediction value for each instance in the list of instances given.

        Args:
            xs: List of input data/instances

        Returns:
            List with predicted (yhat) values/classes
        """
        return [self.bias + sum(wi * xi for wi, xi in zip(self.weights, x)) for x in xs]

    def partial_fit(self, xs: list, ys: list, *, alpha: int = 0.01) -> None:
        """
        Update/fit the model with a single iteration over the given data.

        Args:
            xs: List of input data/instances
            ys: List of target values
            alpha: Learning rate
        """
        for x, y in zip(xs, ys):
            yhat = self.predict([x])[0]
            error = yhat - y
            self.bias -= alpha * error
            self.weights = [wi - alpha * error * xi for wi, xi in zip(self.weights, x)]

    def fit(self, xs: list, ys: list, *, alpha: int = 0.01, epochs: int = 1000) -> None:
        """
        Update/fit the model with {epochs} amount of iterations over the given data.

        Args:
            xs: List of input instances
            ys: List of target values
            alpha: Learning rate
            epochs: Maximum amount of iterations/epochs the model should perform
        """
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)
