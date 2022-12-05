from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gnn.data import MotterLaiData
from gnn.metric import Metric, MetricName

if TYPE_CHECKING:
    from gnn.experiment import Experiment


def configure_logger(exp_dir: str | Path, level: str) -> None:
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    # Define logger instance
    logger = logging.getLogger("MOTTER")
    if logger.handlers:
        logger.handlers.clear()

    # Define format of logging
    formatter = logging.Formatter(fmt="{message}", style="{")

    # Define handler of logger
    handler = logging.FileHandler(filename=exp_dir / f"{level.lower()}.log")
    handler.setFormatter(formatter)

    # Configure logger
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))


def get_mcf_dict(
    experiment: Experiment, network_list: str | list[str]
) -> dict[str, float]:
    if isinstance(network_list, str):
        network_list = [network_list]
    mcf_dict: dict[str, float] = dict.fromkeys(network_list, 0.0)

    for network in network_list:
        data = MotterLaiData.from_hp(
            stage="TEST",
            hp=experiment.hp,
            network_list=network,
            scaler=experiment.scaler,
        )
        metric = Metric(stage="TEST")

        experiment.evaluate(data, metric)
        experiment.logger.warning(f"\n{network}\n{metric}")
        mcf_dict[network] = metric.avg_dict[MetricName.MCF]

    return mcf_dict
