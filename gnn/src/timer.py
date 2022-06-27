import logging
import time
from contextlib import ContextDecorator
from typing import Union

import torch

from .stage import Stage


class Timer(ContextDecorator):
    """Track time and store the history"""

    def __init__(self, stage: Union[str, Stage], device: torch.device) -> None:
        if isinstance(stage, str):
            stage = Stage[stage.upper()]
        self._stage = stage

        # When level of logger is lower than debug, precisely track time
        self._logger = logging.getLogger("MOTTER")
        self._precise = (self._logger.level <= logging.DEBUG) and (
            device != torch.device("cpu")
        )
        self._device = device

        # Container of time
        self._start: float = 0.0  # start time
        self._history: list[float] = []  # List of time duration per epoch

    def start_timer(self) -> None:
        """Store current time to 'start'"""
        # Measure current time
        self._start = self._measure_current_time()

    def end_timer(self, exc_type, exc_value, traceback) -> None:
        """Store current time to 'end'"""
        # Measure duration
        duration = self._measure_current_time() - self._start

        # Save duration
        self._history.append(duration)

        # Log
        self._logger.debug(
            f"{self._stage.name:<10} finished with {duration:.4f} seconds"
        )

    def log_average(self) -> None:
        """Log average time"""
        self._logger.warning(
            f"Average {self._stage.name:<10} time: "
            f"{sum(self._history)/len(self._history):.4f}"
        )

    def _measure_current_time(self) -> float:
        """
        Measure current time.
        If _precise is true, synchronize cuda device and measure time
        """
        if self._precise:
            torch.cuda.synchronize(device=self._device)
        return time.perf_counter()

    # Set alias for context manager dunder methods
    __enter__ = start_timer
    __exit__ = end_timer


if __name__ == "__main__":
    print("This is module timetracker")
