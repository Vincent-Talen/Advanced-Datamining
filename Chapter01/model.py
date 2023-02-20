class Perceptron:
    def __repr__(self):
        return f"Perceptron(dim={self.dim})"

    def __init__(self, dim: int):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.fitted = False

    def predict(self, xs: list) -> list:
        yhats = list()
        for x in xs:
            # Compute pre-activation value
            a = sum([wi * xi for wi, xi in zip(self.weights, x)]) + self.bias
            # Save post-activation value
            yhats.append(1.0 if a > 0.0 else -1.0 if a < 0.0 else 0.0)
        return yhats

    def partial_fit(self, xs: list, ys: list) -> None:
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
        epochs_done = 0
        while not self.fitted and (epochs == 0 or epochs_done < epochs):
            self.partial_fit(xs, ys)
            epochs_done += 1
        print(f"Model fully fitted after {epochs_done} epochs")
