from typing import Any

import matplotlib.pyplot as plt
import torch
from src import Experiment, HyperParameter, get_args, get_mcf_dict


def test(
    experiment: Experiment,
    network_list: list[str],
    check_point: dict[str, Any],
    name: str,
    plot: bool = True,
) -> dict[str, Any]:
    mcf_dict = get_mcf_dict(experiment, network_list)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(mcf_dict.keys(), mcf_dict.values())
        ax.set_xticks(list(mcf_dict.keys()))
        if name == "scalable":
            ax.set_xticklabels(network.split("_")[2] for network in network_list)
        elif name == "real":
            ax.set_xticklabels(
                "_".join(network.split("_")[:2]) for network in network_list
            )
        ax.set_ylim(0.8, 1.0)
        fig.savefig(experiment.exp_dir / f"{name}.png", facecolor="w")

    check_point[name] = mcf_dict
    return check_point


def main():
    # Get arguments and create hyper parameter instance accordingly
    hp = HyperParameter(**vars(get_args()))
    hp.save()

    # Generate experiment
    experiment = Experiment(hp, log=True)

    # Run experiment
    experiment.run()

    # Test bigger network
    check_point: dict[str, Any] = torch.load(experiment.best_state_file_path)
    alpha = hp.train[0].split("_")[-1]
    scalable_list = [
        hp.train[0],
        f"shk_1000_{alpha}",
        f"shk_2000_{alpha}",
        f"shk_3000_{alpha}",
        f"shk_4000_{alpha}",
        f"shk_5000_{alpha}",
        f"shk_6000_{alpha}",
        f"shk_7000_{alpha}",
        f"shk_8000_{alpha}",
    ]
    check_point = test(experiment, scalable_list, check_point, "scalable", plot=False)

    # Test real network
    real_list = [
        f"es_98_{alpha}",
        f"fr_146_{alpha}",
        f"gb_120_{alpha}",
    ]
    check_point = test(experiment, real_list, check_point, "real", plot=False)

    torch.save(check_point, experiment.best_state_file_path)


if __name__ == "__main__":
    main()
