import argparse
import itertools
from pathlib import Path
from typing import Generator, Optional, Union

import matplotlib.figure as fig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

"""
    Generate power grid network introduced at
    'A random growth model for power grids and other spatially embedded infrastructure networks'
    by P. Schultz, J. Heitzig, J. Kurths (2014)
"""

BASE_DIR = Path(__file__).parents[1].resolve()
NETWORK_DIR = BASE_DIR / "data/edge_list"

NETWORK_DIR.mkdir(parents=True, exist_ok=True)


def get_args(options: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_num_nodes", type=int, default=1, help="Number of initial nodes"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=100, help="Final number of nodes"
    )
    parser.add_argument("-p", type=float, default=0.2, help="Model parameter p")
    parser.add_argument("-q", type=float, default=0.3, help="Model parameter q")
    parser.add_argument("-r", type=float, default=0.33333, help="Model parameter r")
    parser.add_argument("-s", type=float, default=0.1, help="Model parameter s")
    parser.add_argument(
        "--ensemble_name", default="test", help="Name of ensemble of networks"
    )
    parser.add_argument("--ensemble_idx", type=int, default=0, help="Ensemble index")

    # Parse the arguments and return
    if options is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args=options)


class SHKNetwork:
    def __init__(
        self,
        initial_num_nodes: int,
        final_num_nodes: int,
        p: float,
        q: float,
        r: float,
        s: float,
    ) -> None:
        """
        initial_num_nodes: Number of nodes to be initialized
        final_num_nodes: Number of nodes at final stage of network growth
        p,q,r,s: parameters for generating network
        """
        self.initial_num_nodes = initial_num_nodes
        self.final_num_nodes = final_num_nodes
        self.p = p
        self.q = q
        self.r = r
        self.s = s

        # Randomly choose position of all nodes
        self.pos = np.random.rand(self.final_num_nodes, 2).astype(np.float32)

        # Create initial network
        self.graph = self._MST(
            self.initial_num_nodes
        )  # Generate minimum spanning tree using euclidean distance
        self._add_initial_edge()  # Add initial edges based on fr

        # Grow the network
        for new_node in range(self.initial_num_nodes, self.final_num_nodes):
            self._grow(new_node)

        # save position information to graph
        nx.set_node_attributes(
            G=self.graph,
            values={node: self.pos[node] for node in range(self.final_num_nodes)},
            name="pos",
        )

    def plot(
        self, fig_ax: Optional[tuple[fig.Figure, plt.Axes]] = None, save: bool = False
    ) -> tuple[fig.Figure, plt.Axes]:
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = fig_ax
        ax.axis("off")

        nx.draw_networkx(
            self.graph,
            pos=nx.get_node_attributes(self.graph, "pos"),
            ax=ax,
            node_size=50,
            node_color="k",
            with_labels=False,
        )
        if save:
            fig.savefig("powergrid.png")

        return fig, ax

    def to_csv(self, file_path: Union[str, Path], delimiter: str = "\t") -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        nx.write_edgelist(self.graph, file_path, delimiter=delimiter, data=False)

    def to_pickle(self, file_path: Union[str, Path]) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix != ".pkl":
            file_path = file_path.with_suffix(".pkl")

        import pickle

        with open(file_path, mode="wb") as f:
            pickle.dump(self.graph, f)

    def _MST(self, N0: int) -> nx.Graph:
        """Minimum Spanning Tree using euclidean distance between node positions"""
        graph = nx.Graph()

        # Store euclidean distance as edge weight
        for i, j in itertools.combinations(range(N0), 2):
            dist = self.get_euclidean_dist(self.pos[i], self.pos[j])
            graph.add_edge(i, j, weight=dist)

        # Generate minimum spanning tree using euclidean distance
        graph = nx.minimum_spanning_tree(graph)

        return graph

    def _add_initial_edge(self) -> None:
        """Add Initial edges"""
        additional_num_edge = int(
            self.initial_num_nodes * (1.0 - self.s) * (self.p + self.q)
        )
        node_combination = list(
            itertools.combinations(range(self.initial_num_nodes), 2)
        )

        for _ in range(additional_num_edge):
            # Calculate hop distance of current graph
            hop_dist_gen = nx.shortest_path_length(self.graph)
            assert isinstance(hop_dist_gen, Generator)  # Drop type error

            # hop distance generator to ndarray
            hop_dist = np.zeros(
                (self.initial_num_nodes, self.initial_num_nodes), dtype=np.int32
            )
            for node, hop_dist_dict in hop_dist_gen:
                hop_dist_dict = dict(
                    sorted(hop_dist_dict.items())
                )  # Sort by node index
                hop_dist[node] = np.fromiter(hop_dist_dict.values(), dtype=np.int32)

            # Calculate fr values for each combinations of node i,j
            fr_list = [
                float(
                    self.get_fr(
                        self.r,
                        hop_dist[i, j],
                        self.get_euclidean_dist(self.pos[i], self.pos[j]),
                    )
                )
                for i, j in node_combination
            ]

            # Candidates of new edge is sorted from fr_list: max -> min
            for idx in np.argsort(fr_list)[::-1]:
                if not self.graph.has_edge(*node_combination[idx]):
                    node1, node2 = node_combination[idx]
                    break
            else:
                # All node combinations has edges: do nothing
                continue

            # Add edge to chosen node1, node2
            self.graph.add_edge(node1, node2)

    def _grow(self, new_node: int) -> None:
        """Grow network"""
        if np.random.rand() < self.s and self.graph.number_of_edges():
            self._bridge(new_node)
        else:
            self._steady(new_node)

    def _bridge(self, new_node: int) -> None:
        """Remove random link and add new node to bridge the removed link"""
        while True:
            # Randomly select existing node
            target_edge = list(self.graph.edges())[
                np.random.randint(self.graph.number_of_edges())
            ]
            new_position = self.pos[target_edge, :].mean(axis=0)

            # If new position is not occupied in self.pos, save it
            if new_position not in self.pos:
                self.pos[new_node] = new_position
                break

        # Remove target edge and create new edges
        self.graph.remove_edge(*target_edge[:2])
        self.graph.add_edge(target_edge[0], new_node)
        self.graph.add_edge(target_edge[1], new_node)

    def _steady(self, new_node: int) -> None:
        """Add new node and connect new edges to give more redundancy to network"""
        # Find minimal euclidean distance node w.r.t new_node and add edge
        euclidean_dist_to_new_node = self.get_euclidean_dist(
            self.pos[:new_node], self.pos[new_node]
        )
        node = np.argmin(euclidean_dist_to_new_node).item()
        self.graph.add_edge(node, new_node)

        if np.random.rand() < self.p:
            # Add edge with node having maximum fr value
            node = self._find_max_fr_node(new_node, network_size=new_node)
            self.graph.add_edge(node, new_node)

        if np.random.rand() < self.q and self.graph.number_of_nodes() > 2:
            # Choose random node and add edge with another node having maximum fr value
            node1 = np.random.randint(new_node)
            node2 = self._find_max_fr_node(node1, network_size=new_node)
            self.graph.add_edge(
                *sorted([node1, node2])
            )  # Sort node1 and node2, before add edge

    def _find_max_fr_node(self, target: int, network_size: int) -> int:
        """
        Find node with maximum fr value w.r.t target node
        Args
            target: target node to calculate fr
            network_size: Network size before adding new node
        Return
            index of node that has maximum fr with target in range [0, network_size]
        """
        # Calculate euclidean distance
        euclidean_dist = self.get_euclidean_dist(
            self.pos[: network_size + 1], self.pos[target]
        )
        euclidean_dist[target] = np.inf

        # Calculate hop distance
        hop_dist_dict = nx.shortest_path_length(self.graph, source=target)
        assert isinstance(hop_dist_dict, dict)  # Drop type error
        hop_dist_dict = dict(sorted(hop_dist_dict.items()))  # Sort by node index
        hop_dist = np.fromiter(hop_dist_dict.values(), dtype=np.int32)

        # Calculate fr: target itself has value of 0
        fr = self.get_fr(self.r, hop_dist[:-1], euclidean_dist[:-1])
        assert isinstance(fr, np.ndarray)
        for nb in self.graph.neighbors(target):
            if nb == network_size:
                continue
            fr[nb] = 0

        # Return index of node with maximum fr
        return np.argmax(fr).item()

    @staticmethod
    def get_fr(
        r: float,
        hop_dist: Union[int, npt.NDArray[np.int32]],
        euclidean_dist: Union[float, npt.NDArray[np.float32]],
    ) -> Union[float, npt.NDArray[np.float32]]:
        """Calculate fr defined at eq 13"""
        return np.power(1.0 + hop_dist, r) / euclidean_dist

    @staticmethod
    def get_euclidean_dist(
        pos1: npt.NDArray[np.float32], pos2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Calculate euclidean distance between two positions
        Last dimension of pos1,pos2 should be x,y(,z) dimension
        """
        return np.sqrt(np.sum(np.power(pos1 - pos2, 2.0), axis=-1))


def main() -> None:
    args = get_args()

    # Create power grid
    grid = SHKNetwork(
        initial_num_nodes=args.initial_num_nodes,
        final_num_nodes=args.num_nodes,
        p=args.p,
        q=args.q,
        r=args.r,
        s=args.s,
    )

    # Save in csv format
    grid.to_csv(NETWORK_DIR / f"{args.ensemble_name}_{args.ensemble_idx}.csv")


if __name__ == "__main__":
    main()
