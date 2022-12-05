import torch
import torch.nn as nn
import torch_geometric.data as gData
import torch_geometric.nn as gnn

from .regressor import MLP, Regressor


class SkipGINNet(nn.Module):
    def __init__(
        self,
        num_layer: int,
        in_feature: int,
        hidden_feature: int,
        mlp_num_layer: int,
        dropout: float,
        bn_momentum: float | None,
        num_out: int,
        activation: str | None,
    ) -> None:
        super().__init__()
        assert num_layer == 8, "Number of layer should be 8"
        assert isinstance(bn_momentum, float), "Batch Norm should be performed"

        self.conv1 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                in_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn1 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        self.conv2 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                hidden_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn2 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        self.conv3 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                hidden_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn3 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        self.conv4 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                hidden_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn4 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        self.conv5 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                hidden_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn5 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        self.conv6 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                hidden_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn6 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        self.conv7 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                hidden_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn7 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        self.conv8 = gnn.GINConv(
            nn=MLP(
                mlp_num_layer,
                hidden_feature,
                hidden_feature,
                hidden_feature,
                "relu",
                bn_momentum,
                dropout,
            ),
            train_eps=True,
        )
        self.bn8 = gnn.BatchNorm(in_channels=hidden_feature, momentum=bn_momentum)

        # Out layer
        self.regressor = Regressor(
            num_out=num_out,
            hidden_feature=hidden_feature,
            bn_momentum=bn_momentum,
            act=activation,
            dropout=dropout,
        )

    def forward(self, data: gData.Data) -> torch.Tensor:
        x = data.x
        x = torch.relu(self.bn1(self.conv1(x, data.edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, data.edge_index)))
        skip1 = x

        x = torch.relu(self.bn3(self.conv3(x, data.edge_index)))
        x = torch.relu(self.bn4(self.conv4(x, data.edge_index)))
        skip2 = x

        x = torch.relu(self.bn5(self.conv5(x, data.edge_index)))
        x = torch.relu(self.bn6(self.conv6(x, data.edge_index)))
        skip3 = x

        x = torch.relu(self.bn7(self.conv7(x, data.edge_index)))
        x = torch.relu(self.bn8(self.conv8(x, data.edge_index)))
        skip4 = x

        x = (skip1 + skip2 + skip3 + skip4) / 4.0

        return self.regressor(x)
