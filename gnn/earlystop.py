import logging
import numpy as np


class EarlyStop:
    """Early stops training if validation loss doesn't improve 'delta' after 'patience' epochs"""

    def __init__(
        self,
        patience: int = 10,
        loss_delta: float = 0.0,
        loss_relaxation: float = 0.0,
        descending: bool = True,
        best_val: float | None = None,
    ) -> None:
        """
        Initialize early stop class
        Args
            patience: How many epochs to wait after validation loss is improved.
            loss_delta: Minimum change of validation loss to regard as improved
            loss_relaxation: Relaxation of minimum validation loss.
            descending: If true, lower validation is better. Otherwise, higher is better
            best_val: initial best validation score
        """
        # Properties of early stop instance
        self._patience = patience
        self._loss_delta = loss_delta
        self._loss_relaxation = loss_relaxation
        self.logger = logging.getLogger("MOTTER")
        self.descending = 1 if descending else -1

        # Variables to store current state
        if best_val is not None:
            self.best_val = best_val
        else:
            self.best_val = np.inf if descending else -np.inf
        self.counter: int = 0  # Early stop counter
        self.abort: bool = False  # When true, training should be stop
        self.is_best: bool = False  # If current state is best

    def __call__(self, val: float) -> None:
        """
        Check input validation loss if it is the minimum.
        self.abort will be changed if early stop condition is fulfilled
        Args
            val: current validation
        """
        # Change validation in descending order
        descending_val = self.descending * val
        descending_best_val = self.descending * self.best_val

        if descending_val < descending_best_val - self._loss_delta:
            # Validation in descending order is smaller
            self._best(val)
        else:
            # Validation in descending order is bigger
            self._not_best()
            # Check abort (early stop)
            if self.counter >= self._patience:
                self.abort = True

    def resume(self, counter: int, best_val: float) -> None:
        """
        Resume from previous train
        Args
            counter: Last early stop counter
            best_val: best validation value
        """
        # Cutoff history and update best epoch, best_val
        self.best_val = best_val

        if counter == 0:
            # Experiment resume from best model
            self.is_best = True
            self.counter = 0
        else:
            # Experiment resume from last model
            self.is_best = False
            self.counter = counter

        # Reset aborting
        self.abort = False

    def relax(self, reset: bool = False) -> None:
        """
        Reset early stop insance
        Args
            reset: If true, reset counter to 0
        """
        # Relax best validation value according to order
        last_best_val = self.best_val
        self.best_val += self.descending * self._loss_relaxation

        # Reset counter if necessary
        if reset:
            self.counter = 0

        # Log the relaxed inforamtion
        self.logger.info(
            f"Best validation loss relaxes "
            f"{last_best_val:.4f} -> {self.best_val:.4f}"
        )

    def _best(self, val: float) -> None:
        """Input validation loss is best overall"""
        self.is_best = True

        # Reset early stop counter
        self.counter = 0

        # Log to file
        self.logger.info(f"Validation improved: {self.best_val:.4f} -> {val:.4f}")

        # Update best validation value
        self.best_val = val

    def _not_best(self) -> None:
        """Input validation loss is not best"""
        self.is_best = False

        # Update early stop counter
        self.counter += 1

        # Log to file
        self.logger.debug(f"Valdation doesn't improved, best: {self.best_val:.4f}")
        self.logger.info(f"Early stop counter {self.counter}/{self._patience}")


if __name__ == "__main__":
    print("This is module early stop")
