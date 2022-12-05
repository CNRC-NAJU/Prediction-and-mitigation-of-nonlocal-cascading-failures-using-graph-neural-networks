from __future__ import annotations

from typing import Protocol

import torch.optim as optim
from ..hyperparameter import HyperParameter


class BaseSchedular(Protocol):
    @classmethod
    def from_hp(
        cls, hp: HyperParameter, optimizer: optim.Optimizer | None = None
    ) -> BaseSchedular:
        ...

    def resume(self, epoch: int) -> None:
        ...

    def step(self, epoch_iteration: float | None = None):
        ...
