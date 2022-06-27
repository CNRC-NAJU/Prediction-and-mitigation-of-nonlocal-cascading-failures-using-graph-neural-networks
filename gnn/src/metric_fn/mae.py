import torch
import torch.nn.functional as F
from torch_scatter import segment_coo

"""
Calculate Mean Absolute Error between two input tensors

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
Network-wise calculation return average of MAE per each networks.
When network size of all networks are equal(i.e. N_1=N_2=...=N_M),
network-wise calculation gives same result as normal MAE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Weighted calculation applies weight of (true/true_max) to absolute error before doing average
When true <= eps, eps becomes the weight
"""


def get_mae(pred_tensor: torch.Tensor, true_tensor: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_tensor, true_tensor)


def get_networkwise_mae(
    pred_tensor: torch.Tensor, true_tensor: torch.Tensor, network_idx: torch.Tensor
) -> torch.Tensor:
    mae_per_network = segment_coo(
        src=torch.abs(pred_tensor - true_tensor), index=network_idx, reduce="mean"
    )
    return mae_per_network.mean()
