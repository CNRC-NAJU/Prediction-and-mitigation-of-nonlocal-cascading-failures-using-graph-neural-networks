import contextlib
from enum import Enum
from typing import Callable, ContextManager

import torch

from gnn.metric_fn import (get_kt, get_mae, get_mcf, get_mse,
                           get_networkwise_kt, get_networkwise_mae,
                           get_networkwise_mcf, get_networkwise_mse,
                           get_networkwise_r2, get_r2)
from gnn.stage import Stage


class MetricName(Enum):
    MCF = "Mean Cumulative Fraction"  # Bigger is better, No gradient, [0,1]
    KT = "Kendall Tau Score"  # Bigger is better, No gradient, [0,1]
    MSE = "Mean Squared Error"  # Lower is better
    MAE = "Mean Absolute Error"  # Lower is better
    R2 = "R2 score"  # Bigger is better


class Metric:
    """Container for various metrics for experiment"""

    def __init__(
        self,
        stage: str | Stage,
        loss: str | MetricName = MetricName.MAE,
        networkwise: bool = True,
    ) -> None:
        if isinstance(stage, str):
            stage = Stage[stage.upper()]
        if isinstance(loss, str):
            loss = MetricName[loss.upper()]
        self.stage = stage
        self.loss = loss
        self.networkwise = networkwise

        # Dictionary of functions which calculate each metrics
        self.networkwise = networkwise
        self.metric_fn_dict: dict[MetricName, Callable[[], torch.Tensor]] = {
            MetricName.MCF: self.get_mcf,
            MetricName.MSE: self.get_mse,
            MetricName.MAE: self.get_mae,
            MetricName.KT: self.get_kt,
            MetricName.R2: self.get_r2,
        }

        # Calculating gradient is only necessary for loss
        self.calculate_grad: dict[MetricName, ContextManager] = dict.fromkeys(
            MetricName, torch.no_grad()
        )
        self.calculate_grad[self.loss] = contextlib.nullcontext()

        # Dictionary of metric tensors with gradient
        self.tensor_dict = dict.fromkeys(MetricName, torch.tensor([]))

        # Stored variables to calculate every metrics per iteration
        self.pred = torch.tensor([])  # Prediction value output from nn
        self.true = torch.tensor([])  # True value from data
        self.network_idx = torch.tensor([])  # Index of network of each nodes
        self.network_size_list = torch.tensor([])  # List of network size

        # Stored variables to average metrics per epoch
        self.iteration_count: int = 0
        self.avg_dict = dict.fromkeys(MetricName, 0.0)

    # ======================== Methods called per iterations =========================
    def set(
        self,
        pred_value: torch.Tensor,
        true_value: torch.Tensor,
        batch: torch.Tensor,
        network_size: torch.Tensor,
    ) -> None:
        """
        Set predict value and true value
        Since the pred/true values are changed, all metrics should be reset
        Args
            pred_value: prediction(output) of model
            true_value: true label(target)
            batch: Index of network where each node are located
            network_size: List of network size
        """
        self.pred = pred_value.squeeze()
        self.true = true_value.squeeze()
        self.network_idx = batch
        self.network_size_list = network_size

        # Reset metric values
        self.tensor_dict = dict.fromkeys(MetricName, torch.tensor([]))

        # Some metrics(mse, mae) doesn't need to do networkwise-calculation
        # if given network sizes of input data are all same
        self.equal_network_size = torch.all(network_size == network_size[0])

    def step(self) -> None:
        """Calculate all metric values and sum to average"""
        self.iteration_count += 1

        for metric_name, metric_fn in self.metric_fn_dict.items():
            # Calculate each metric values
            self.tensor_dict[metric_name] = metric_fn()
            self.avg_dict[metric_name] += self.tensor_dict[metric_name].item()

    def get_loss(self) -> torch.Tensor:
        """Return proper 'loss' tensor with gradient flow"""
        return self.metric_fn_dict[self.loss]()

    # ========================== Methods called per epochs ===========================
    def average(self) -> dict[MetricName, float]:
        """Average metric values stored in average_dict during single epoch"""
        self.avg_dict = {
            metric_name: value / self.iteration_count
            for metric_name, value in self.avg_dict.items()
        }

        return self.avg_dict

    def reset_average(self) -> None:
        """Clear average dictionary ready for new epoch"""
        self.iteration_count = 0
        self.avg_dict = dict.fromkeys(MetricName, 0.0)

    def __str__(self) -> str:
        info = f"{self.stage.name:<10}"
        for metric_name, value in self.avg_dict.items():
            if metric_name in [MetricName.MSE, MetricName.MAE]:
                info += f" {metric_name.name}={value:.2e}"
            else:
                info += f" {metric_name.name}={value:.4f}"
        return info

    # ============================== Calculate Metrics ===============================
    def get_mcf(self) -> torch.Tensor:
        if self.tensor_dict[MetricName.MCF].nelement() != 0:
            return self.tensor_dict[MetricName.MCF]

        with self.calculate_grad[MetricName.MCF]:
            if self.networkwise:
                mcf_tensor = get_networkwise_mcf(
                    self.pred, self.true, self.network_size_list
                )
            else:
                mcf_tensor = get_mcf(self.pred, self.true)
        return mcf_tensor

    def get_mse(self) -> torch.Tensor:
        if self.tensor_dict[MetricName.MSE].nelement() != 0:
            return self.tensor_dict[MetricName.MSE]

        with self.calculate_grad[MetricName.MSE]:
            if self.networkwise and (not self.equal_network_size):
                mse_tensor = get_networkwise_mse(self.pred, self.true, self.network_idx)
            else:
                mse_tensor = get_mse(self.pred, self.true)
        return mse_tensor

    def get_mae(self) -> torch.Tensor:
        if self.tensor_dict[MetricName.MAE].nelement() != 0:
            return self.tensor_dict[MetricName.MAE]

        with self.calculate_grad[MetricName.MAE]:
            if self.networkwise and (not self.equal_network_size):
                mae_tensor = get_networkwise_mae(self.pred, self.true, self.network_idx)
            else:
                mae_tensor = get_mae(self.pred, self.true)
        return mae_tensor

    def get_kt(self) -> torch.Tensor:
        if self.tensor_dict[MetricName.KT].nelement() != 0:
            return self.tensor_dict[MetricName.KT]

        with self.calculate_grad[MetricName.KT]:
            if self.networkwise:
                kt_tensor = get_networkwise_kt(
                    self.pred, self.true, self.network_size_list
                )
            else:
                kt_tensor = get_kt(self.pred, self.true)
        return kt_tensor

    def get_r2(self) -> torch.Tensor:
        if self.tensor_dict[MetricName.R2].nelement() != 0:
            return self.tensor_dict[MetricName.R2]

        with self.calculate_grad[MetricName.R2]:
            if self.networkwise:
                r2_tensor = get_networkwise_r2(self.pred, self.true, self.network_idx)
            else:
                r2_tensor = get_r2(self.pred, self.true)
        return r2_tensor


if __name__ == "__main__":
    print("This is module metric")
