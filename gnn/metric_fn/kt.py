import itertools

import torch
from scipy.stats import kendalltau

"""
CAUTION!! This metric does not provide back-propagation: No gradient flow

Calculate kendall tau correlation between two input tensors

Params:
pred_tensor: 1D tensor with size (N_1 + N_2 + ... + N_M)
true_tensor: 1D tensor with shape (N_1 + N_2 + ... + N_M)
network_size_tensor: 1D tensor with shape (M,)
                     list of network sizes
                     ex) [N_1, N_2, ..., N_M]
top_ratio: Fraction of largest values to be computed.
           If 1.0, same as normal kendall tau correlation
where
Total number of networks: M
N_1: Network size (number of nodes) of 1-st network
...
N_M: Network size (number of nodes) of M-th network

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network-wise calculation returns average of kendall tau correlation per each networks
"""


def get_kt(pred_tensor: torch.Tensor, true_tensor: torch.Tensor) -> torch.Tensor:
    tau, _ = kendalltau(
        pred_tensor.detach().cpu().numpy(), true_tensor.detach().cpu().numpy()
    )
    return torch.tensor(tau, dtype=torch.float32, device=true_tensor.device)


def get_networkwise_kt(
    pred_tensor: torch.Tensor,
    true_tensor: torch.Tensor,
    network_size_tensor: torch.Tensor,
) -> torch.Tensor:
    tau_tensor = torch.zeros_like(
        network_size_tensor, dtype=torch.float32, device=true_tensor.device
    )

    network_boundary = list(itertools.accumulate(network_size_tensor))
    for idx in range(len(network_size_tensor)):
        start = network_boundary[idx - 1] if idx else 0
        end = network_boundary[idx]
        tau_tensor[idx] = get_kt(pred_tensor[start:end], true_tensor[start:end])

    return tau_tensor.mean()
