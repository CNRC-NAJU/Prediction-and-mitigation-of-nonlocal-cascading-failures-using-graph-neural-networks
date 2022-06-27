import numpy as np
import torch

from .base import BaseScaler, arr


class IdentityScaler(BaseScaler):
    def fit(self, data: arr) -> None:
        return

    def transform(self, data: arr) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data

    def inverse_transform(self, data: arr) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data

    def __str__(self) -> str:
        return f"Identity Scaler"
