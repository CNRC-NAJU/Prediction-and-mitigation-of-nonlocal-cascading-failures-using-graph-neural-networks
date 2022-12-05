import torch.nn as nn
from ..hyperparameter import HyperParameter

from .GINNet import SkipGINNet


def count_trainable_param(model: nn.Module) -> int:
    """Return number of trainable parameters of model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gnn(hp: HyperParameter) -> nn.Module:
    in_feature = 1 + len(hp.add_degree) + len(hp.add_network_size)

    if hp.model == "SkipGINNet":
        assert isinstance(hp.num_parallel, int)
        return SkipGINNet(
            hp.num_layer,
            in_feature,
            hp.hidden_feature,
            hp.num_parallel,
            hp.dropout,
            hp.bn_momentum,
            hp.num_out,
            hp.out_act,
        )

    raise NotImplementedError(f"No such model {hp.model}")
