from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch


def directed2undirected(edge_list: npt.NDArray[np.int64]) -> torch.LongTensor:
    """Get directed edge list of shape (E, 2),
    Return undirected edge index of shape (2, E), for torch_geometric"""
    return torch.from_numpy(np.concatenate([edge_list, edge_list[:, [1, 0]]])).T  # type: ignore


BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"


def read_data(network_name: str, alpha: float) -> dict[str, torch.Tensor]:
    edge_list = pd.read_csv(
        DATA_DIR / f"edge_list/{network_name}.txt",
        delimiter="\t",
        header=None,
        dtype=np.int64,
    ).values
    avalanche_fraction = pd.read_csv(
        DATA_DIR / f"avalanche_fraction/{network_name}_{alpha}.txt",
        header=None,
        dtype=np.float32,
    ).values
    failure_fraction = pd.read_csv(
        DATA_DIR / f"failure_fraction/{network_name}_{alpha}.txt",
        header=None,
        dtype=np.float32,
    ).values
    termination_step = pd.read_csv(
        DATA_DIR / f"time/{network_name}_{alpha}.txt", header=None, dtype=np.int64
    ).values
    avalanche_centrality = avalanche_fraction * failure_fraction

    return {
        "edge_index": directed2undirected(edge_list),
        "alpha": torch.from_numpy(alpha * np.ones_like(avalanche_centrality)),
        "avalanche_fraction": torch.from_numpy(avalanche_fraction),
        "failure_fraction": torch.from_numpy(failure_fraction),
        "termination_step": torch.from_numpy(termination_step),
        "avalanche_centrality": torch.from_numpy(avalanche_centrality),
    }


def main():
    network_names = [f"test_{idx}" for idx in range(2)]
    dataset_name, alpha = "test", 0.25

    data = []
    for network_name in network_names:
        data.append(read_data(network_name, alpha))

    df = pd.DataFrame.from_records(data)
    df.to_pickle(DATA_DIR / f"{dataset_name}_{alpha}.pkl")


if __name__ == "__main__":
    main()
