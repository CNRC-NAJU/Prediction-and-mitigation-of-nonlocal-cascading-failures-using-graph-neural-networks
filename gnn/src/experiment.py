import logging
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import matplotlib.figure as fig
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.utils as utils
import torch.optim as optim
from tqdm.auto import tqdm

from . import schedular
from .data import MotterLaiData
from .earlystop import EarlyStop
from .hyperparameter import HyperParameter
from .metric import Metric, MetricName
from .models import count_trainable_param, get_gnn
from .timer import Timer
from .utils import configure_logger


class Experiment:
    """Create NN model, Motter lai data using input hyper parameter and do training"""

    def __init__(self, hp: HyperParameter, log: bool = False):
        # ============================ Device for experiment ============================
        try:
            self.device = torch.device(hp.device)
        except RuntimeError:
            # When hp.device: 0,1,2,3
            self.device = torch.device(f"cuda:{hp.device}")
        self.hp = hp

        # ================================= Print, log ==================================
        self.use_tqdm = not hp.no_tqdm
        self.exp_id = hp.exp_id
        self.exp_dir = Path(hp.root_dir) / str(hp.exp_id)
        self.best_state_file_path: Path = self.exp_dir / "best.pth"
        self.last_state_file_path: Path = self.exp_dir / "last.pth"
        self.hist_file_path: Path = self.exp_dir / "history.dat"
        self.hist_file = open(self.hist_file_path, mode="a", buffering=1)

        configure_logger(self.exp_dir, hp.log_level)
        if log:
            self.logger = logging.getLogger("MOTTER")
        else:
            self.logger = logging.getLogger()

        # ================================= Random seed ==================================
        if hp.seed is not None:
            # Fix random seed and remove randomness
            torch.manual_seed(hp.seed)
            if "cuda" in str(self.device):
                # When using gpu
                torch.cuda.manual_seed(hp.seed)
                cudnn.deterministic = True
                cudnn.benchmark = False

        # =============================== Motter-Lai Data ================================
        disk_read_start = time.perf_counter()
        train_index: list[Any] = [None] * len(hp.train)
        for i, network in enumerate(hp.train):
            # Validation should also be picked at this directory
            if network in hp.val:
                train_index[i] = 0.8
        self.train_data = MotterLaiData.from_hp("train", hp, index_list=train_index)
        self.scaler = self.train_data.scaler

        val_index: list[Any] = [None] * len(hp.val)
        for i, network in enumerate(hp.val):
            # Train is also picked from this directory
            if network in hp.train:
                val_index[i] = (0.8, 1.0)
        self.val_data = MotterLaiData.from_hp(
            "val", hp, index_list=val_index, scaler=self.scaler
        )
        disk_read_end = time.perf_counter()

        self.train_loader = self.train_data.get_loader()
        self.val_loader = self.val_data.get_loader()

        # ====================== Model, Optimizer, Schedular, Clip =======================
        self.model = get_gnn(hp).to(self.device)
        self.optimizer: optim.Optimizer = getattr(optim, hp.optimizer)(
            self.model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
        )
        schedular_type: schedular.BaseSchedular = getattr(
            schedular, f"{hp.schedular}Schedular"
        )
        self.schedular = schedular_type.from_hp(hp, self.optimizer)
        self.clip = hp.clip

        # ============================ Max Epoch, Early stop =============================
        self.max_epoch = hp.epochs
        if "inv_" in hp.objective:
            self.inverse = True
            self.early_stop_metric = MetricName[hp.objective.removeprefix("inv_")]
        else:
            self.inverse = False
            self.early_stop_metric = MetricName[hp.objective]
        self.early_stop = EarlyStop(
            patience=hp.patience,
            loss_delta=hp.early_stop_delta,
            descending=self.early_stop_metric in [MetricName.MSE, MetricName.MAE],
        )

        # ============================= Loss, Metric, Timer ==============================
        self.loss_name = MetricName[hp.loss]
        self.train_metric = Metric("TRAIN", self.loss_name, hp.networkwise)
        # Validation loss should be calculated network-wise
        self.val_metric = Metric("VAL", self.loss_name, True)

        self.train_timer = Timer("TRAIN", self.device)
        self.val_timer = Timer("VAL", self.device)

        # ========================== Container for best epoch ===========================
        self.last_epoch: int = -1  # Last epoch of previous experiment
        self.last_counter: int = 0  # Last early stop counter
        self.best_epoch: int = -1  # Best epoch with minimum validation loss
        self.best_val_metric = dict.fromkeys(MetricName, 0.0)  # Best validation metrics

        # ====================== Log the newly created experiment =======================
        self.logger.info(self.train_data)
        self.logger.info(self.val_data)
        self.logger.info(
            "Reading data takes "
            f"{round(disk_read_end - disk_read_start, 2):.2f} seconds"
        )
        self.logger.info(
            "Number of trainable parameters: " f"{count_trainable_param(self.model):,}"
        )

    @classmethod
    def from_id(
        cls,
        exp_id: int,
        root_dir: Union[str, Path],
        best: bool = True,
        log: bool = False,
    ):
        hp = HyperParameter.load(exp_id, root_dir)
        assert (hp.root_dir / str(exp_id)).is_dir(), f"No such experiment: {exp_id}"

        experiment = cls(hp, log)
        experiment.load(best)
        return experiment

    def load(self, best: bool = True) -> None:
        """
        Load previous experiment
        Args
            best: If true, load best state. Otherwise, load last state
            device: target device to load model
        """
        # Load model
        self._load_state(best)
        self.logger.warning(f"Load experiment from epoch {self.last_epoch}")

    def run(self, max_epoch: Optional[int] = None) -> None:
        """Run train/validation for max_epoch times"""
        if max_epoch is None:
            max_epoch = self.max_epoch
        self._start_run()

        # Only for possibly unbounded variables
        epoch = 0  # Epoch index
        train_metric_dict = dict.fromkeys(
            MetricName, 0.0
        )  # Average train metric values per each epoch
        val_metric_dict = dict.fromkeys(
            MetricName, 0.0
        )  # Average validation metric values per each epoch

        # Start running
        epoch_range = range(self.last_epoch + 1, self.last_epoch + max_epoch + 1)
        for epoch in self._get_range(epoch_range, desc="Epoch"):
            self.logger.info("")
            self.logger.warning(f"Epoch {epoch}/{epoch_range[-1]}")

            # Do train and save metric of current epoch
            with self.train_timer:
                train_metric_dict = self.train(epoch)
                self.logger.debug(self.train_metric)

            # Do validation and save metric of current epoch
            with self.val_timer:
                val_metric_dict = self.validate()
                self.logger.debug(self.val_metric)

            # Update early stop
            self.early_stop(val_metric_dict[self.early_stop_metric])
            if self.early_stop.is_best:
                # Save model, optimizer state if this is best epoch
                self._save_state(epoch, val_metric_dict)

            self._write_history(train_metric_dict, val_metric_dict)

            # Early stop
            if self.early_stop.abort:
                self.logger.warning(f"Early stopping!!")
                break

        # When early stop is not triggered, save last state
        else:
            self._save_state(epoch, val_metric_dict, best=False)

        self._finish_run()

    # ============================== Entire Run Helper ===============================
    def _start_run(self) -> None:
        """When resume training, all history should be cut off from previous best epoch"""
        self.model.to(self.device)

        # When not resume training, do nothing
        if self.last_epoch == -1:
            return

        # Resume runner by previous epoch
        self.schedular.resume(self.last_epoch)
        self.early_stop.resume(
            self.last_counter, self.best_val_metric[self.early_stop_metric]
        )

        # Cut history file
        with open(self.hist_file_path, "r+") as hist_file:
            history = hist_file.readlines()
            hist_file.seek(0)
            for _, line in zip(range(self.last_epoch + 1), history):
                hist_file.write(line)
            hist_file.truncate()

    def _finish_run(self) -> None:
        """Log the overall results and load best state"""

        # Report overall data
        self.logger.info("")
        self.logger.warning(
            f"Best model: Epoch {self.best_epoch}, "
            f"{self.early_stop_metric.name}="
            f"{self.best_val_metric[self.early_stop_metric]:.4f}\n"
        )
        self.train_timer.log_average()
        self.val_timer.log_average()

        # Close history file
        self.hist_file.close()

        # Load best model
        self._load_state(best=True)

    # ============================= Single Epoch Helper ==============================
    def train(self, epoch: int) -> dict[MetricName, float]:
        self.train_metric.reset_average()
        self.model.train()

        for iteration, batch_data in enumerate(self.train_loader):
            batch_data.to(self.device)
            self.optimizer.zero_grad()

            # Feed forward
            prediction = self.model(batch_data)
            self.train_metric.set(
                prediction, batch_data.y, batch_data.batch, batch_data.n
            )

            # Calculate metrics
            self.train_metric.step()
            loss = self.train_metric.get_loss()

            # Back propagation
            loss.backward()

            # Update model parameters and learning rate
            if self.clip != np.inf:
                try:
                    utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as e:
                    self.logger.error(
                        "Error occurs due to non-finite gradient at clipping"
                    )
                    raise RuntimeError(e)
            self.optimizer.step()
            self.schedular.step(epoch + (iteration + 1) / len(self.train_loader))

        # Return Average metrics
        return self.train_metric.average()

    @torch.no_grad()
    def validate(self) -> dict[MetricName, float]:
        self.val_metric.reset_average()
        self.model.eval()

        for batch_data in self.val_loader:
            batch_data.to(self.device)

            # Feed forward
            prediction = self.model(batch_data)
            if self.inverse:
                self.val_metric.set(
                    self.scaler.inverse_transform(prediction),
                    self.scaler.inverse_transform(batch_data.y),
                    batch_data.batch,
                    batch_data.n,
                )
            else:
                self.val_metric.set(
                    prediction, batch_data.y, batch_data.batch, batch_data.n
                )

            # Calculate metrics
            self.val_metric.step()

        # Return Average metrics
        return self.val_metric.average()

    @torch.no_grad()
    def evaluate(
        self,
        data: MotterLaiData,
        metric: Optional[Metric] = None,
        inverse_transform: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Evaluate data and save metric values to input metric
        Returns pred_list, true_list
        """
        if metric is None:
            metric = Metric("TEST")
        metric.reset_average()
        self.model.eval()

        prediction_list: list[torch.Tensor] = []
        true_label_list: list[torch.Tensor] = []

        for batch_data in data.get_loader(batch_size=1):
            batch_data.to(self.device)

            # Feed forward
            prediction = self.model(batch_data)
            if inverse_transform:
                prediction = data.scaler.inverse_transform(prediction)
                true_label = data.scaler.inverse_transform(batch_data.y)
            else:
                true_label = batch_data.y

            # Store to return values
            prediction_list.append(prediction)
            true_label_list.append(true_label)

            # Save metric
            metric.set(prediction, true_label, batch_data.batch, batch_data.n)
            metric.step()
        metric.average()

        return prediction_list, true_label_list

    def _write_history(
        self,
        train_metric_dict: dict[MetricName, float],
        val_metric_dict: dict[MetricName, float],
    ) -> None:
        """Save current metrics to history file"""
        self.hist_file.write(
            ",".join(
                str(val)
                for val in [
                    *train_metric_dict.values(),
                    *val_metric_dict.values(),
                    self.early_stop.counter,
                ]
            )
            + "\n"
        )

    # =================================== Utility ====================================
    def _get_range(
        self, iterable: Iterable[Any], desc: Optional[str] = None
    ) -> Iterable[Any]:
        """
        Return iterable with or without tqdm according to self.use_tqdm
        Args
            iterable: input for tqdm
            desc: description of tqdm bar
        """
        return tqdm(iterable, desc=desc) if self.use_tqdm else iterable

    def _save_state(
        self, epoch: int, val_metric: dict[MetricName, float], best: bool = True
    ) -> None:
        """
        Save current state: epoch, val_metric, model state, optimizer state, grad scaler state
        Args
            epoch: Number of epoch to be saved
            val_metric: dictionary of metric values for every metrics available
            best: If true, this is best state
        """
        if best:
            # Store best epoch and validation metric
            self.best_epoch = epoch
            self.best_val_metric = val_metric
        else:
            # Store last epoch. best_epoch, best_val_metric: not changing
            self.last_epoch = epoch

        # Create dictionary object to be saved by torch save
        obj: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
        }

        if best:
            torch.save(obj, self.best_state_file_path)
        else:
            # When saving last state, also need to save last epoch, early stop counter
            obj["last_epoch"] = self.last_epoch
            obj["early_stop_counter"] = self.early_stop.counter
            torch.save(obj, self.last_state_file_path)

    def _load_state(self, best: bool) -> None:
        """
        Load state files: best.pth or last.pth
        Args
            best: If true, load best state. Otherwise, load last state
            device: destination device to load neural network
        """
        # Read state file
        if best:
            file_path = self.best_state_file_path
        else:
            file_path = self.last_state_file_path
        try:
            check_point = torch.load(file_path, map_location=self.device)
        except FileNotFoundError:
            raise RuntimeError(f"No such file {file_path.resolve()}")

        # Load state to model and optimizer
        self.model.load_state_dict(check_point["model_state_dict"])
        self.optimizer.load_state_dict(check_point["optimizer_state_dict"])

        # Load best epoch, validation metrics
        self.best_epoch = check_point["best_epoch"]
        self.best_val_metric = check_point["best_val_metric"]

        if best:
            # When loading best state, last epoch will be same as best epoch
            self.last_epoch = self.best_epoch
        else:
            self.last_epoch = check_point["last_epoch"]
            self.last_counter = check_point["early_stop_counter"]

    @classmethod
    def plot_metric(
        cls,
        exp_dir: Union[str, Path],
        metric_name: Union[str, MetricName],
        fig_ax: Optional[tuple[fig.Figure, plt.Axes]] = None,
        schedular_list: Optional[list[int]] = None,
        save: bool = False,
    ) -> tuple[fig.Figure, plt.Axes]:
        """
        Plot metric history
        Args
            exp_dir: Path of experimant directory
            metric_name: target metric to plot
            fig_ax: When given, use the figure, axes object
            schedular_list: list of epoch when schedular is updated
            save: If true, save figure
        """
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if isinstance(exp_dir, str):
            exp_dir = Path(exp_dir)
        if isinstance(metric_name, str):
            metric_name = MetricName[metric_name]
        train_loss_index = list(MetricName).index(metric_name)
        val_loss_index = len(MetricName) + train_loss_index

        # Read history file
        history = np.loadtxt(exp_dir / "history.dat", delimiter=",")
        best_epoch = np.argwhere(history[:, -1].astype(np.int64) == 0)[-1].item()

        # Schedular
        if schedular_list is not None:
            for s in schedular_list:
                ax.axvline(s, color="green")

        # Plot val loss
        ax.plot(history[:, val_loss_index], label="validation", color="b")
        ax.plot(
            best_epoch,
            history[best_epoch, val_loss_index],
            "rX",
            markersize=15,
            label="Best",
        )

        # Plot train loss
        ax1 = ax.twinx()
        if metric_name in [MetricName.MSE, MetricName.MAE, MetricName.R2]:
            ax1.plot(history[:, train_loss_index], label="train", color="g")
            ax1.set_ylabel("train")
            ax1.legend()

        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("val")
        ax.set_xlim(0, len(history))
        ax.set_title(metric_name.value)

        # ylim
        if metric_name is MetricName.MCF:
            ax.set_ylim(0.8, 1.0)
        elif metric_name is MetricName.R2:
            ax.set_ylim(0.0, 1.0)
            ax1.set_ylim(0.0, 1.0)

        # Save plot
        if save:
            fig.savefig(exp_dir / f"{metric_name.value}.png", facecolor="w")
        return fig, ax
