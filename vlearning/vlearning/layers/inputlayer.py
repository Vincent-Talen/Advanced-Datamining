"""This module contains the InputLayer class for a neural network.

An InputLayer is a special type of layer that is the first layer in a network and
the only layer instance you actually interact with when using the network. Meaning
it is the only layer that you can call the `predict` and `fit` methods on.
"""
from random import shuffle

from overrides import override
from tqdm import trange

from vlearning.layers import Layer


class InputLayer(Layer):
    """The first layer in a neural network, thus does not have any inputs.

    It is a special type of layer that is the first layer in a network and the only
    layer instance you actually interact with when using the network. Meaning
    it is the only layer that you can call the `predict` and `fit` methods on.

    Attributes:
        num_inputs (int): The number of inputs to the layer.
        num_outputs (int): The number of outputs from the layer.
        name (str): The name of the layer.
        next_layer (Layer | None): The next layer in the network.
        training_history (dict[str, list[float]]): Dictionary storing training history
    """
    @override
    def __init__(
        self,
        num_outputs: int,
        *,
        name: str | None = None,
        next_layer: Layer | None = None,
    ):
        """Overrides parent's method to add a training_history instance attribute.

        To keep track of the loss, accuracy, etc. of the network during training,
        the `training_history` attribute can be used and accessed by the fit methods.

        Args:
            num_outputs: The number of outputs from the layer.

        Keyword Args:
            name: The name of the layer.
            next_layer: The next layer in the network.
        """
        super().__init__(num_outputs, name=name, next_layer=next_layer)
        self.training_history = {"loss": []}

    @override
    def __call__(self, xs, labels=None, *, alpha=None):
        return self.next_layer(xs, labels, alpha=alpha)

    @override
    def _set_inputs(self, num_inputs: int) -> None:
        raise TypeError("An InputLayer does not have, nor accept, inputs.")

    def predict(self, xs: list[list[float]]) -> list[list[float]]:
        """Get the predicted values for the given dataset.

        This method is an accessibility/ease-of-use wrapper that simply calls
        the instance itself, returning the predicted values of the network.

        Args:
            xs: The instances the network should predict values for.

        Returns:
            The predicted values for the given dataset.
        """
        y_hats, _, _ = self(xs)
        return y_hats

    def evaluate(self, xs: list[list[float]], labels: list[list[float]]) -> float:
        """Get the mean loss of the network for the instances of the given dataset.

        Args:
            xs: The instances the network should predict values for.
            labels: A list containing the correct labels for all instances

        Returns:
            The mean loss of the network for the instances of the given dataset.
        """
        _, losses, _ = self(xs, labels)
        return sum(losses) / len(losses)

    def partial_fit(
        self,
        xs: list[list[float]],
        labels: list[list[float]],
        *,
        alpha: float = 0.001,
        validation_data: tuple[list[list[float]], list[list[float]]] = None,
        batch_size: int = None,
    ) -> None:
        """Fit/train the network to the given dataset for a single epoch.

        After training this one epoch the loss of the network is calculated and stored
        in the `training_history` instance attribute. If the `validation_data` argument
        is provided, the network will be evaluated on this data after each epoch and the
        validation loss will be stored in the `training_history` under the
        `validation_loss` key. If the `batch_size` argument is provided, the network
        will update weights and biases multiple times per epoch, thus converging faster.

        Args:
            xs:
                The instances the network should predict values for.
            labels:
                A list containing the correct label per feature of each instance if the
                loss should be returned, otherwise `None`.

        Keyword Args:
            alpha:
                The learning rate of the network.
            validation_data:
                A tuple containing two lists, the first being the validation data itself
                and the second the expected/actual labels for the validation instances.
            batch_size:
                The number of instances to train the network on before updating the
                weights. If not provided, the network will be trained on all instances.
        """
        num_instances = len(xs)
        batch_size = batch_size or num_instances

        epoch_losses = []
        for i in range(0, num_instances, batch_size):
            batch_instances = xs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            _, batch_losses, _ = self(batch_instances, batch_labels, alpha=alpha)
            epoch_losses.extend(batch_losses)

        epoch_mean_loss = sum(epoch_losses) / num_instances
        self.training_history["loss"].append(epoch_mean_loss)

        if validation_data:
            validation_loss = self.evaluate(*validation_data)
            self.training_history["validation_loss"].append(validation_loss)

    def fit(
        self,
        xs: list[list[float]],
        labels: list[list[float]] = None,
        *,
        alpha: float = 0.001,
        epochs: int = 100,
        validation_data: tuple[list[list[float]], list[list[float]]] = None,
        batch_size: int = None,
    ) -> dict[str, list[float]]:
        """Fit/train the network to the given dataset for a given number of epochs.

        If the `validation_data` argument is provided, the network will be evaluated
        on this data after each epoch and the validation loss will be stored in the
        `training_history` dictionary attribute under the `validation_loss` key. To
        improve the training process when batch learning, all instances' data and labels
        are shuffled before each epoch.

        Args:
            xs:
                The instances the network should predict values for.
            labels:
                A list containing the correct labels for all instances if the loss
                should be returned, otherwise `None`.

        Keyword Args:
            alpha:
                The learning rate of the network.
            epochs:
                The number of epochs to train the network for.
            validation_data:
                A tuple containing two lists, the first being the validation data itself
                and the second the expected/actual labels for the validation instances.
            batch_size:
                The number of instances to train the network on before updating the
                weights. If not provided, the network will be trained on all instances.

        Returns:
            A dictionary containing the training history of the network.
        """
        if validation_data:
            self.training_history.setdefault("validation_loss", [])

        for _ in trange(epochs, desc="Epochs trained", unit="epoch", ncols=128):
            # Shuffle the data and labels before each epoch by pairing them with zip
            paired_lists = list(zip(xs, labels))
            shuffle(paired_lists)
            xs_shuffled, labels_shuffled = zip(*paired_lists)

            # Train the network for a single epoch
            self.partial_fit(
                xs_shuffled,
                labels_shuffled,
                alpha=alpha,
                validation_data=validation_data,
                batch_size=batch_size
            )

        return self.training_history
