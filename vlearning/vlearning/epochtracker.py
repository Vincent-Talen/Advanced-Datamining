"""This module contains the EpochTracker class, which can be used to print a progress
bar and the training rate/time per epoch during fitting.

Examples:
    >>> for _ in EpochTracker(num_epochs=42):
    ...     do_stuff()

    >>> tracker = EpochTracker(num_epochs=42)
    >>> for epoch in range(1, 43):
    ...     do_stuff()
    ...     tracker.update(epoch)
"""
from time import time, sleep


class EpochTracker:
    """A class to track the progress per epoch of training a model.

    The EpochTracker class can be used as an iterable to loop over the epochs of a
    training process. It will print a progress bar showing the percentage and count of
    epochs completed of the total number of epochs and the training rate/time per epoch.

    It is still possible to instantiate the EpochTracker and call the update method
    manually to update the progress bar to a specific epoch.

    Attributes:
        num_epochs: The total number of epochs to train for.
        n_cols: The number of columns to use for the progress bar.
    """
    def __init__(self, num_epochs: int, *, n_cols: int = 128):
        """Initializes an EpochTracker instance with the given number of epochs.

        When the instance is created it immediately prints the initial progress bar,
        with the rate of epochs per second and estimated time left still unknown.

        Args:
            num_epochs: The amount of epochs the iterable should be.
            n_cols: The width (amount of cols/chars) of the complete epoch tracker.
        """
        self.num_epochs = num_epochs
        self.n_cols = n_cols
        self.start_time = time()
        self.update(0)

    def __iter__(self) -> int:
        """Makes it possible to use the EpochTracker as an iterable.

        It is a simple generator that yields the epoch number and updates the progress
        bar to that epoch. The generator will yield the epoch number from 1 to the
        number of epochs given when creating the EpochTracker instance.

        Yields:
            The number of the current epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            yield epoch
            self.update(epoch)

    @staticmethod
    def _create_progress_bar(bar_max_len: int, fraction: float) -> str:
        """Generates a progress bar string that is filled up to the given fraction.

        Args:
            bar_max_len: The maximum length of the progress bar.
            fraction: The fraction of the progress bar to fill.

        Returns:
            A string with UTF symbols representing the progress bar.
        """
        # Get characters with different thicknesses to use for the progress bar
        charset = u" " + u"".join(map(chr, range(0x258F, 0x2587, -1)))
        nsyms = len(charset) - 1
        # Get the amount of full and fractional symbols to use
        bar_len, frac_bar_len = divmod(int(fraction * bar_max_len * nsyms), nsyms)
        bar = charset[-1] * bar_len
        if bar_len < bar_max_len:  # whitespace padding
            bar += charset[frac_bar_len] + charset[0] * (bar_max_len - bar_len - 1)
        return bar

    def _refresh(self, n: int, elapsed_time: float) -> None:
        """Refreshes the progress bar to the given epoch and elapsed time.

        Args:
            n: The epoch to update the progress bar to.
            elapsed_time: The time in seconds since the start of the training.
        """
        # Calculate the rate of epochs per second and create a formatted string for it
        rate = n / elapsed_time if elapsed_time else None
        rate_str = (f"{rate:5.2f}" if rate else "?") + "epoch/s"
        inv_rate = 1 / rate if rate else None
        rate_inv_str = (f"{inv_rate:5.2f}" if rate else "?") + "s/epoch"
        rate_str = rate_inv_str if inv_rate and inv_rate > 1 else rate_str

        # Calculate the remaining time and the fraction of completed epochs
        fraction = n / self.num_epochs
        remaining_time = (self.num_epochs - n) / rate if rate and self.num_epochs else 0

        # Create formatted strings for the separate desired types of information
        n_elapsed_str = f"{n:>{len(str(self.num_epochs))}}/{self.num_epochs}"
        t_elapsed_str = self.format_interval(elapsed_time)
        t_remaining_str = self.format_interval(remaining_time) if rate else "?"

        # Create the full prefix- and postfix strings
        prefix = f"Epochs completed: {fraction * 100:3.0f}%|"
        postfix = f"| {n_elapsed_str} [{t_elapsed_str}<{t_remaining_str}, {rate_str}]"

        # Create the progress bar and display it including all other information
        bar_max_len = max(1, self.n_cols - len(prefix) - len(postfix))
        progress_bar = self._create_progress_bar(bar_max_len, fraction)
        print(f"\r{prefix}{progress_bar}{postfix}", end="", flush=True)

    def update(self, epoch: int) -> None:
        """Updates the progress bar to the given epoch.

        This method is called automatically when using the EpochTracker as an iterable,
        removing the need to call it manually, however this is still possible.

        Args:
            epoch: The epoch to update the progress bar to.
        """
        elapsed_time = time() - self.start_time
        self._refresh(epoch, elapsed_time)
        if epoch == self.num_epochs:
            print()

    @staticmethod
    def format_interval(seconds: int | float) -> str:
        """Formats a number of seconds as a clock time, [H:]MM:SS

        Args:
            seconds: Number of seconds.

        Returns:
            A string formatted as '[H:]MM:SS'.
        """
        minutes, s = divmod(int(seconds), 60)
        h, m = divmod(minutes, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    from random import uniform

    for _ in EpochTracker(num_epochs=69):
        sleep(uniform(0.7, 1.3))
