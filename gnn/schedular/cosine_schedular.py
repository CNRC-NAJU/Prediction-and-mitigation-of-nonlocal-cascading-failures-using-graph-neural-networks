import logging
import math

import torch.optim as optim
from ..hyperparameter import HyperParameter
from torch.optim.lr_scheduler import _LRScheduler


class CosineSchedular(_LRScheduler):
    r"""
    Cosine annealing schedular with warm up restart
    \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When T_{cur}=T_{i}`, set `\eta_t = \eta_{min}`.
    When T_{cur}=0` after restart, set `\eta_t=\eta_{max}`.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_period: int,
        period_mult: float,
        warmup: int,
        base_eta_max: float,
        eta_max_mult: float,
        last_epoch: int = -1,
    ) -> None:
        r"""
        Initialize shedular
        Note that \eta_{min} at equation is learning rate at input optimizer
        Args
            optimizer: Optimizer to control lerning rate
            period: Base number of epochs for learning rate refresh. Exclude length of warmup
            period_mult: multiplier of base period for learning rate refresh. i-th period = warmup + base_period * (period_mult ** i)
            warmup: Number of epochs for warm-up. always constant over cycles
            base_eta_max: Base \eta_{max} at equation
            eta_max_mult: multiplier of base eta max. i-th \eta_{max} = base_eta_max * (eta_max_mult ** i)
            last_epoch: The index of last epoch
        """
        # Check input
        if base_period <= 0 or not isinstance(base_period, int):
            raise ValueError(
                f"Expected positive integer base_period, but got {base_period}"
            )
        if period_mult < 1.0 or not isinstance(period_mult, float):
            raise ValueError(f"Expected float period_mult >= 1, but got {period_mult}")
        if warmup < 0 or not isinstance(warmup, int):
            raise ValueError(f"Expected positive integer warmup, but got {warmup}")
        if base_eta_max < 0 or not isinstance(base_eta_max, float):
            raise ValueError(
                f"Expected positive float base_eta_max, but got {base_eta_max}"
            )
        if eta_max_mult <= 0 or eta_max_mult > 1 or not isinstance(eta_max_mult, float):
            raise ValueError(
                f"Expected [0,1) float eta_max_mult, but got {eta_max_mult}"
            )

        # Store properties of schedular
        self._base_period = base_period
        self._period_mult = period_mult
        self._warmup = warmup
        self._base_eta_max = base_eta_max
        self._eta_max_mult = eta_max_mult

        # Variables to store current state of schedular
        self._cycle = 0  # Index of cosine cycle
        self._start_of_cycle = 0  # Epoch when current cycle is started
        self._period_of_cycle = (
            warmup + base_period
        )  # Period(number of epochs) of current cycle
        self._eta_max = base_eta_max  # eta max used for current cycle
        self._location_at_cycle = 0  # location of schedular w.r.t start_of_cycle

        # Variables defined at super()
        self.base_lrs: list[float] = []
        self._last_lr: list[float] = []
        self.optimizer: optim.Optimizer
        self.last_epoch: int
        super().__init__(optimizer, last_epoch)

        # Logger
        self.logger = logging.getLogger("MOTTER")

    def get_lr(self) -> list[float]:
        """Get learnining rate of schedular"""
        # Warm up stage: linearly increase learning rate to eta_max
        if self._location_at_cycle < self._warmup:
            return [
                base_lr
                + (self._eta_max - base_lr) / self._warmup * self._location_at_cycle
                for base_lr in self.base_lrs
            ]
        # Cosine annealing learning rate
        else:
            return [
                base_lr
                + 0.5
                * (self._eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self._location_at_cycle - self._warmup)
                        / (self._period_of_cycle - self._warmup)
                    )
                )
                for base_lr in self.base_lrs
            ]

    def step(self, epoch_iteration: float | None = None) -> None:
        """
        Update state of schedular and set learning rate of optimizer
        Args
            epoch_iteration: epoch + iteration/(tot iteration per epoch)
        """
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        # Epoch iteration is not given: step up 1 epoch
        if epoch_iteration is None:
            self._location_at_cycle += 1
            self.last_epoch += 1

        # Epoch_iteration is given
        else:
            self._location_at_cycle = epoch_iteration - self._start_of_cycle
            self.last_epoch = math.floor(epoch_iteration)

        # If current location is bigger than period of cycle, update cycle
        if self._location_at_cycle >= self._period_of_cycle:
            self._update_cycle()

        # Set learning rate of optimizer
        self._set_lr()

    def _update_cycle(self) -> None:
        """Update current cycle"""
        # Location at cycle is larger than period of cycle: reduce it
        self._location_at_cycle -= self._period_of_cycle

        # Index of cycle should be increased by 1
        self._cycle += 1

        # Start of current cycle is now increased by period of cycle
        self._start_of_cycle += self._period_of_cycle

        # Period of cycle is updated by multiplier of period mult
        self._period_of_cycle = int(
            self._warmup + (self._period_of_cycle - self._warmup) * self._period_mult
        )

        # Eta max of current cycls is updated by multiplier of eta max mult
        self._eta_max *= self._eta_max_mult

        # Logging
        self.logger.debug(f"New cycle of schedular started at epoch {self.last_epoch}")

    def _set_lr(self) -> None:
        """Set learning rate to optimizer"""
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    @classmethod
    def from_hp(cls, hp: HyperParameter, optimizer: optim.Optimizer | None = None):
        """Alternative initializer using hyper parameter instance"""
        if optimizer is None:
            # dummy optimizer
            import torch.nn as nn

            optimizer = optim.Adam(nn.Linear(1, 1).parameters())
        return cls(
            optimizer,
            hp.schedular_period,
            hp.schedular_period_mult,
            hp.warmup,
            hp.schedular_eta_max,
            hp.schedular_eta_max_mult,
        )

    def resume(self, epoch: int) -> None:
        """load schedular based on input epoch"""
        # Step for epoch times: If input epoch is 0, single epoch is done
        for _ in range(epoch + 1):
            self.step()

        # Set learning rate of optimizer
        self._set_lr()

    def get_updated_epoch(self, max_epoch: int) -> list[int]:
        updated_epoch = [0]
        idx = 0
        while updated_epoch[-1] <= max_epoch:
            updated_epoch.append(
                int(
                    updated_epoch[-1]
                    + self._base_period * self._period_mult**idx
                    + self._warmup
                )
            )
            idx += 1
        return updated_epoch[1:-1]
