from __future__ import annotations

from typing import Optional, Protocol

import torch.optim as optim
from src.hyperparameter import HyperParameter


class BaseSchedular(Protocol):
    @classmethod
    def from_hp(
        cls, hp: HyperParameter, optimizer: Optional[optim.Optimizer] = None
    ) -> BaseSchedular:
        ...

    def resume(self, epoch: int) -> None:
        ...

    def step(self, epoch_iteration: Optional[float] = None):
        ...
