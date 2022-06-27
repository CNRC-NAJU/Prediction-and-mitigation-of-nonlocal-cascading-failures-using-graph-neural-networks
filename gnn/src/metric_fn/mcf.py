import itertools

import torch

"""
CAUTION!! This metric does not provide back-propagation: No gradient flow

Calculate Mean Cumulative Fraction (MCF) score between two input tensors
MCF = Area under cumulative sum of true label in descending order of prediction
    = 1/N * sum_{n=1}^{N} * 1/N * sum_{i=0}^{n} x_{sigma(i)}
    = 1/N^2 * sum_{i=1}^{N} (N-i) x_{sigma(i)}
where sigma(i): ranking of i-th node in descending order

Params:
pred_tensor: 1D tensor with size (N_1 + N_2 + ... + N_M)
true_tensor: 1D tensor with shape (N_1 + N_2 + ... + N_M)
network_size_tensor: 1D tensor with shape (M,)
                     list of network sizes
                     ex) [N_1, N_2, ..., N_M]
where
Total number of networks: M
N_1: Network size (number of nodes) of 1-st network
...
N_M: Network size (number of nodes) of M-th network

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network-wise calculation return average of R2 per each networks.
"""


def get_mcf(pred_tensor: torch.Tensor, true_tensor: torch.Tensor) -> torch.Tensor:
    # Sort true tensor in descending order
    true_descending_tensor, _ = true_tensor.sort(descending=True)

    # mcf of true tensor: both descending and ascending
    weight = torch.arange(
        start=len(true_tensor),
        end=0,
        step=-1,
        dtype=torch.float32,
        device=true_tensor.device,
    )
    true_descending_mcf = torch.sum(true_descending_tensor * weight)
    true_ascending_mcf = torch.sum(reversed(true_descending_tensor) * weight)

    # mcf value of prediction index
    _, pred_descendig_idx = pred_tensor.sort(descending=True, stable=True)
    pred_descending_mcf = torch.sum(true_tensor[pred_descendig_idx] * weight)

    return torch.div(
        pred_descending_mcf - true_ascending_mcf,
        true_descending_mcf - true_ascending_mcf,
    )


def get_networkwise_mcf(
    pred_tensor: torch.Tensor,
    true_tensor: torch.Tensor,
    network_size_tensor: torch.Tensor,
) -> torch.Tensor:
    mcf_tensor = torch.zeros_like(
        network_size_tensor, dtype=torch.float32, device=true_tensor.device
    )
    network_boundary = list(itertools.accumulate(network_size_tensor))
    for idx in range(len(network_size_tensor)):
        start = network_boundary[idx - 1] if idx else 0
        end = network_boundary[idx]
        mcf_tensor[idx] = get_mcf(pred_tensor[start:end], true_tensor[start:end])

    return mcf_tensor.mean()
