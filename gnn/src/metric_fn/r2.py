import torch
import torch.nn.functional as F
from torch_scatter import gather_coo, segment_coo

"""
Calculate R2 score between two input tensors
R2 = 1 - mse(pred, true) / mse(true, mean_true)
   = 1 - mse(pred, true) / var(true)

Params:
pred_tensor: 1D tensor with size (N_1 + N_2 + ... + N_M)
true_tensor: 1D tensor with shape (N_1 + N_2 + ... + N_M)
network_idx: 1D tensor with shape (N_1 + N_2 + ... + N_M)
             index of network that each nodes are located
             ex) [1,1,...,1,2,...2, ..., M,...,M]
where
Total number of networks: M
N_1: Network size (number of nodes) of 1-st network
...
N_M: Network size (number of nodes) of M-th network

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network-wise calculation return average of R2 per each networks.
"""


def get_r2(pred_tensor: torch.Tensor, true_tensor: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.div(
        F.mse_loss(pred_tensor, true_tensor), true_tensor.var(unbiased=False)
    )


def get_networkwise_r2(
    pred_tensor: torch.Tensor, true_tensor: torch.Tensor, network_idx: torch.Tensor
) -> torch.Tensor:
    true_mean_per_network = segment_coo(true_tensor, network_idx, reduce="mean")
    networkwise_true_err = true_tensor - gather_coo(true_mean_per_network, network_idx)
    true_var_per_network = segment_coo(
        src=networkwise_true_err**2.0, index=network_idx, reduce="mean"
    )

    mse_per_network = segment_coo(
        src=(pred_tensor - true_tensor) ** 2.0, index=network_idx, reduce="mean"
    )

    return 1.0 - torch.mean(mse_per_network / true_var_per_network)
