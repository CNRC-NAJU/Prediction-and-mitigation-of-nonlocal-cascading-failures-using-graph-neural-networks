import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch_geometric.data as gData

from gnn import Experiment

BASE_DIR = Path(__file__).parent.resolve()
SHK_NETWORK = [
    "shk_rand_0.25",
    "shk_1000_0.25",
    "shk_2000_0.25",
    "shk_3000_0.25",
    "shk_4000_0.25",
    "shk_5000_0.25",
    "shk_6000_0.25",
    "shk_7000_0.25",
    "shk_8000_0.25",
]


def directed2undirected(edge_list: npt.NDArray[np.int64]) -> torch.LongTensor:
    """Get directed edge list of shape (E, 2),
    Return undirected edge index of shape (2, E), for torch_geometric"""
    return torch.from_numpy(np.concatenate([edge_list, edge_list[:, [1, 0]]])).T  # type: ignore


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    """
    Save argument values to hyper parameter instance
    Args
        options: arguments from jupyter kernel
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data_dir", default=f"{BASE_DIR}/data", help="data path")
    parser.add_argument(
        "--root_dir",
        default=f"{BASE_DIR}/experiment",
        help="Root directory for experiments.",
    )
    parser.add_argument(
        "--network_name", default="test_0", help="Which network to be predicted"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.25, help="alpha used for Motter-Lai model"
    )
    parser.add_argument(
        "--exp_ids",
        nargs="+",
        default=[1, 2, 3, 4],
        help="experiment id of GNNs used for prediction",
    )

    # Parse the arguments and return
    if options is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args=options)


@torch.no_grad()
def main() -> None:
    args = get_args()

    # Directories
    data_dir = Path(args.data_dir)
    prediction_dir = data_dir / f"avalanche_centrality_gnn"
    prediction_dir.mkdir(exist_ok=True)

    # Read network
    edge_list = pd.read_csv(
        data_dir / f"edge_list/{args.network_name}.txt",
        delimiter="\t",
        header=None,
        dtype=np.int64,
    ).values
    graph: nx.Graph = nx.from_edgelist(edge_list)
    alpha: float = args.alpha
    num_nodes = graph.number_of_nodes()

    # Define data for input
    data = gData.Data(
        x=alpha * torch.ones(num_nodes, 1),
        edge_index=directed2undirected(edge_list),
        y=torch.ones(num_nodes, 1),  # dummy label
        n=num_nodes,
    )

    # Prediction of GNNs
    predictions: list[npt.NDArray[np.float32]] = []
    for exp_id in args.exp_ids:
        experiment = Experiment.from_id(exp_id, args.root_dir)
        device = experiment.device
        prediction: torch.Tensor = experiment.model(data.to(str(device)))
        predictions.append(prediction.cpu().numpy())

    avg_prediction = np.mean(predictions, axis=0)
    np.savetxt(prediction_dir / f"{args.network_name}_{alpha}.txt", avg_prediction)


if __name__ == "__main__":
    main()
