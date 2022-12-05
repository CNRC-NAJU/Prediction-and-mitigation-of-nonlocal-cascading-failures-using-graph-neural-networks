from typing import Any

import matplotlib.pyplot as plt
import torch

from gnn import Experiment, HyperParameter, get_args, get_mcf_dict


def test(
    experiment: Experiment,
    network_list: list[str],
    name: str,
    plot: bool = True,
) -> dict[str, float]:
    mcf_dict = get_mcf_dict(experiment, network_list)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(list(mcf_dict.keys()), list(mcf_dict.values()))
        ax.set_xticks(list(mcf_dict.keys()))
        ax.set_xticklabels("_".join(network.split("_")[:2]) for network in network_list)
        fig.savefig(str(experiment.exp_dir / f"{name}.png"), facecolor="w")

    return mcf_dict


def main():
    # Get arguments and create hyper parameter instance accordingly
    hp = HyperParameter(**vars(get_args()))
    hp.save()

    # Generate experiment
    experiment = Experiment(hp, log=True)

    # Run experiment
    experiment.run()

    # Test bigger network
    scalable_list = [
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
    scalable_mcf = test(experiment, scalable_list, "scalable", plot=True)
    print(scalable_mcf)

    # Test real network
    real_list = ["es_98_0.25", "fr_146_0.25", "gb_120_0.25"]
    real_mcf = test(experiment, real_list, "real", plot=True)
    print(real_mcf)


if __name__ == "__main__":
    main()
