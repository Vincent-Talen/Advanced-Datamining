class Perceptron:
    """
    This class implements Rosenblatt's Perceptron, it is a binary single neuron model.
    It can solve linear classification problems through simple learning algorithms
    and is considered the first generation of neural networks.

    Weights are used for inputs which are obtained during the training stage,
    after performing addition between the input values and weights
    it uses the signum activation function to return the predicted classification.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.fitted = False

    def __repr__(self):
        return f"Perceptron(dim={self.dim})"

    def predict(self, xs: list) -> list:
        """
        Calculates prediction value for each instance in the list of instances given.

        Args:
            xs: List of input data/instances

        Returns:
            List with predicted (yhat) values/classes
        """
        yhats = list()
        for x in xs:
            # Compute pre-activation value
            a = self.bias + sum([wi * xi for wi, xi in zip(self.weights, x)])
            # Save post-activation value
            yhats.append(1.0 if a > 0.0 else -1.0 if a < 0.0 else 0.0)
        return yhats

    def partial_fit(self, xs: list, ys: list) -> None:
        """
        Update/fit the model with a single iteration over the given data.

        Args:
            xs: List of input data/instances
            ys: List of target values
        """
        model_updated = False
        for x, y in zip(xs, ys):
            # get prediction yhat
            yhat = self.predict([x])[0]

            # update bias and weights if error is not 0
            if error := yhat - y:
                model_updated = True
                self.bias -= error
                self.weights = [wi - error * xi for wi, xi in zip(self.weights, x)]

        if not model_updated:
            self.fitted = True

    def fit(self, xs: list, ys: list, *, epochs: int = 0) -> None:
        """
        Update/fit the model with {epochs} amount of iterations over the given data.
        When given a value of 0 for epochs it will continue iterating until the model is fully fitted.

        Args:
            xs: List of input instances
            ys: List of target values
            epochs: Amount of iterations/epochs the model should update itself for
                    (default=0 -> unlimited until fully fitted)
        """
        epochs_completed = 0
        while not self.fitted and (epochs == 0 or epochs_completed < epochs):
            self.partial_fit(xs, ys)
            epochs_completed += 1
        if self.fitted:
            print(f"Model has been fully fitted after {epochs_completed} epochs")
        else:
            print(f"Model after fitting for {epochs_completed} epochs")


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
