from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        num_layer: int,
        in_feature: int,
        hidden_feature: int,
        out_feature: int,
        final_act: Optional[str],
        bn_momentum: Optional[float],
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layer == 1:
            self.mlp = nn.Linear(in_feature, out_feature)

        else:
            mlp: list[nn.Module] = []

            # First layer
            mlp.append(
                nn.Linear(in_feature, hidden_feature, bias=(bn_momentum is None))
            )
            if bn_momentum is not None:
                mlp.append(nn.BatchNorm1d(hidden_feature, momentum=bn_momentum))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(p=dropout))

            # Second ~ last-1 layer
            for _ in range(num_layer - 2):
                mlp.append(
                    nn.Linear(
                        hidden_feature, hidden_feature, bias=(bn_momentum is None)
                    )
                )
                if bn_momentum is not None:
                    mlp.append(nn.BatchNorm1d(hidden_feature, momentum=bn_momentum))
                mlp.append(nn.ReLU())

            # Last layer
            mlp.append(nn.Linear(hidden_feature, out_feature))
            self.mlp = nn.Sequential(*mlp)

        # Final activation
        if final_act is None:
            self.act = lambda x: x
        elif final_act == "relu":
            self.act = torch.relu
        elif final_act == "sigmoid":
            self.act = torch.sigmoid
        elif final_act == "hardsigmoid":
            self.act = F.hardsigmoid
        elif final_act == "softsign":
            self.act = lambda x: (F.softsign(x) + 1.0) / 2.0
        elif final_act == "modified_sigmoid":

            def modified_sigmoid(x: torch.Tensor) -> torch.Tensor:
                return torch.min(
                    torch.sigmoid(x) * 1.01, torch.tensor([1.0], device=x.device)
                )

            self.act = modified_sigmoid
        else:
            raise NotImplementedError(f"No such act: {final_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.mlp(x))


class Regressor(nn.Module):
    def __init__(
        self,
        num_out: int,
        hidden_feature: int,
        bn_momentum: Optional[float],
        act: Optional[str],
        dropout: float,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            num_layer=num_out,
            in_feature=hidden_feature,
            hidden_feature=hidden_feature,
            out_feature=1,
            final_act=act,
            bn_momentum=bn_momentum,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
