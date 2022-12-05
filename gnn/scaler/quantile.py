import torch

from .base import BaseScaler, arr


class QuantileScaler(BaseScaler):
    """
    QuatileTransformer + MinMaxScaler
    WARNING: This scaler doesn't preserve gradient during inverse transform
    """

    def __init__(
        self, num_data: int, n_quantiles: int = 500, dist: str = "uniform"
    ) -> None:
        from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

        self.quantile_transformer = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution=dist, subsample=num_data
        )
        self.minmax_scaler = MinMaxScaler()

    def fit(self, data: arr) -> None:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        tmp = self.quantile_transformer.fit_transform(data)
        self.minmax_scaler.fit(tmp)

    def transform(self, data: arr) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            device = data.device
            data = data.detach().cpu().numpy()
        else:
            device = torch.device("cpu")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        qtransformed = self.quantile_transformer.transform(data)
        minmax_qtransformed = self.minmax_scaler.transform(qtransformed)

        return torch.from_numpy(minmax_qtransformed).squeeze().to(device)

    def inverse_transform(self, data: arr) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            device = data.device
            data = data.detach().cpu().numpy()
        else:
            device = torch.device("cpu")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        qtransformed = self.minmax_scaler.inverse_transform(data)
        inversed = self.quantile_transformer.inverse_transform(qtransformed)

        return torch.from_numpy(inversed).squeeze().to(device)

    def __str__(self) -> str:
        return f"Quntile transformer with min-max scaling"
