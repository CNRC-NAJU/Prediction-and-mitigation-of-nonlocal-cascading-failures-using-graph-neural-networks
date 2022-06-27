from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import torch
import torch_geometric.data as gData
import torch_geometric.loader as gLoader
import torch_geometric.utils as gUtils

from . import scaler as sc
from .hyperparameter import HyperParameter
from .stage import Stage


class MotterLaiData:
    def __init__(
        self,
        stage: Union[str, Stage],
        data_dir: Path,
        network_list: list[str],
        index_list: list[Any],
        add_degree: list[str],
        add_network_size: list[str],
        label: str,
        scaler: Union[None, str, sc.BaseScaler],
        batch_size: int,
    ) -> None:
        assert len(network_list) == len(index_list)
        if isinstance(stage, str):
            stage = Stage[stage.upper()]

        self.add_degree = add_degree
        self.add_network_size = add_network_size
        self.label = label
        self.batch_size = batch_size
        self.stage = stage

        # Read data
        in_feature_list: list[torch.Tensor] = []
        edge_index_list: list[torch.Tensor] = []
        out_list: list[torch.Tensor] = []
        for network, index in zip(network_list, index_list):
            in_feature, edge_index, out = self.read_df_with_index(
                data_dir / f"{network}.pkl", index
            )
            in_feature_list.extend(in_feature)
            edge_index_list.extend(edge_index)
            out_list.extend(out)
        out_tensor = torch.cat(out_list, dim=0)
        self.out_max = out_tensor.max().item()

        # Scale the label
        if isinstance(scaler, sc.BaseScaler):
            # If scaler is given, use the scaler
            self.scaler = scaler
        else:
            # Otherwise, define new scaler and fit to out_tensor
            if scaler is None:
                self.scaler = sc.IdentityScaler()
            elif scaler == "QuantileScaler":
                self.scaler = sc.QuantileScaler(num_data=len(out_tensor))
            else:
                self.scaler = getattr(sc, scaler)()
            self.scaler.fit(out_tensor)
        out_list = [self.scaler.transform(out) for out in out_list]

        # Save to list of gData.Data format
        self.data_list: list[gData.Data] = [
            gData.Data(x=in_feature, edge_index=edge_index, y=out, n=len(out))
            for in_feature, edge_index, out in zip(
                in_feature_list, edge_index_list, out_list
            )
        ]

    @classmethod
    def from_hp(
        cls,
        stage: Union[str, Stage],
        hp: HyperParameter,
        network_list: Union[str, list[str], None] = None,
        index_list: Optional[list[Any]] = None,
        scaler: Optional[sc.BaseScaler] = None,
    ):
        """Alternative initializer using hyper parameter instance"""
        if isinstance(stage, str):
            stage = Stage[stage.upper()]

        if isinstance(network_list, str):
            network_list = [network_list]

        if network_list is None:
            if stage is Stage.TRAIN:
                network_list = hp.train
            elif stage is Stage.VAL:
                network_list = hp.val
            else:
                raise RuntimeError(f"You should specify network_list for TEST dataset")

        if index_list is None:
            index_list = [None] * len(network_list)

        if scaler is None:
            return cls(
                stage,
                hp.data_dir,
                network_list,
                index_list,
                hp.add_degree,
                hp.add_network_size,
                hp.label,
                hp.scaler,
                hp.batch_size,
            )
        else:
            return cls(
                stage,
                hp.data_dir,
                network_list,
                index_list,
                hp.add_degree,
                hp.add_network_size,
                hp.label,
                scaler,
                hp.batch_size,
            )

    def get_loader(
        self, batch_size: Optional[int] = None, pin_memory: bool = True
    ) -> gLoader.DataLoader:
        if batch_size is None:
            batch_size = self.batch_size
        return gLoader.DataLoader(
            self.data_list,
            batch_size=batch_size,
            shuffle=(self.stage is Stage.TRAIN),
            pin_memory=pin_memory,
        )

    def to(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, torch.device):
            device = str(device)
        if len(device) == 1:
            device = f"cuda:{device}"

        self.data_list = [data.to(device) for data in self.data_list]

    def __str__(self) -> str:
        return f"Number of {self.stage.name} data: {len(self.data_list)}"

    def read_df_with_index(
        self, file: Path, index: Union[None, float, tuple, pd.Index]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        df: pd.DataFrame = pd.read_pickle(file)
        num_data = len(df)
        if index is None:
            pass
        elif isinstance(index, float):
            df = df.loc[: int(index * num_data)]
        elif isinstance(index, tuple):
            df = df.loc[int(index[0] * num_data) : int(index[1] * num_data)]
        elif isinstance(index, pd.Index):
            df = df.loc[index]
        else:
            raise ValueError("index list is invalid")

        # Extract information from dataframe
        alpha_list: list[torch.Tensor] = df["alpha"].tolist()
        edge_index_list: list[torch.Tensor] = df["edge_index"].tolist()
        label_list: list[torch.Tensor] = df[self.label].tolist()

        # Infeatures. Default: constant alpha
        in_feature_list = [alpha.unsqueeze(1) for alpha in alpha_list]
        if self.add_degree:
            degree_list: list[torch.Tensor] = [
                gUtils.degree(edge_index[0, :]).unsqueeze(1).type(torch.float32)
                for edge_index in edge_index_list
            ]
            if "normalize" in self.add_degree:
                # degree / max_degree
                max_degree_list = [torch.max(degree) for degree in degree_list]
                in_feature_list = [
                    torch.cat([in_feature, degree / max_degree], dim=1)
                    for in_feature, degree, max_degree in zip(
                        in_feature_list, degree_list, max_degree_list
                    )
                ]
            if "log" in self.add_degree:
                # log10(degree)
                in_feature_list = [
                    torch.cat([in_feature, torch.log10(degree)], dim=1)
                    for in_feature, degree in zip(in_feature_list, degree_list)
                ]
            if "inverse" in self.add_degree:
                # 1/degree
                in_feature_list = [
                    torch.cat([in_feature, 1 / degree], dim=1)
                    for in_feature, degree in zip(in_feature_list, degree_list)
                ]
            if "sqrt" in self.add_degree:
                # 1/sqrt(degree)
                in_feature_list = [
                    torch.cat([in_feature, 1 / torch.sqrt(degree)], dim=1)
                    for in_feature, degree in zip(in_feature_list, degree_list)
                ]

        if self.add_network_size:
            network_size_list: list[torch.Tensor] = [
                len(alpha) * torch.ones(size=(len(alpha), 1), dtype=torch.float32)
                for alpha in alpha_list
            ]
            if "inverse" in self.add_network_size:
                # 1/N
                in_feature_list = [
                    torch.cat([in_feature, 1.0 / n], dim=1)
                    for in_feature, n in zip(in_feature_list, network_size_list)
                ]
            if "sqrt" in self.add_network_size:
                # 1/sqrt(N)
                in_feature_list = [
                    torch.cat([in_feature, 1.0 / torch.sqrt(n)], dim=1)
                    for in_feature, n in zip(in_feature_list, network_size_list)
                ]
            if "log" in self.add_network_size:
                # 1/log10(N)
                in_feature_list = [
                    torch.cat([in_feature, 1.0 / torch.log10(n)], dim=1)
                    for in_feature, n in zip(in_feature_list, network_size_list)
                ]

        return in_feature_list, edge_index_list, label_list
