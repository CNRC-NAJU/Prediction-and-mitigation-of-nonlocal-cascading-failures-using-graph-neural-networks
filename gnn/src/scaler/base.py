from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import numpy.typing as npt
import torch

arr = Union[torch.Tensor, npt.NDArray[np.float32]]


class BaseScaler(ABC):
    """Base scaler class for both torch.Tensor and np.array"""

    @abstractmethod
    def fit(self, data: arr) -> None:
        """Fit scaler to raw data"""
        pass

    @abstractmethod
    def transform(self, data: arr) -> torch.Tensor:
        """Transform data based on stored fit informations"""
        pass

    @abstractmethod
    def inverse_transform(self, data: arr) -> torch.Tensor:
        """Inverse transform based on stored fit informations"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation of scaler"""
        pass

    def fit_transform(self, data: arr) -> torch.Tensor:
        """Do both fit and transform"""
        self.fit(data)
        return self.transform(data)
