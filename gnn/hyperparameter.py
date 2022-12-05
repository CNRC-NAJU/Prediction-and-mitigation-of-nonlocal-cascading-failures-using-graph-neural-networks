import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

BASE_DIR = Path(__file__).parents[1].resolve()


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    """
    Save argument values to hyper parameter instance
    Args
        options: arguments from jupyter kernel
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--data_dir", default=f"{BASE_DIR}/data", help="data path")
    parser.add_argument(
        "--train",
        nargs="+",
        default=["shk_rand_0.25"],
        help="Network type for train dataset",
    )
    parser.add_argument(
        "--val",
        nargs="+",
        default=["shk_rand_0.25"],
        help="Network type for validation dataset",
    )
    parser.add_argument(
        "--add_degree",
        nargs="+",
        choices=["normalize", "log", "inverse", "sqrt"],
        help="(deprecated)\n"
        "normalize: [degree/max_degree]\n"
        "log: [log10(degree)]\n"
        "inverse: [1/degree]\n"
        "sqrt: [1/sqrt(degree)] as additional input feature",
    )
    parser.add_argument(
        "--add_network_size",
        nargs="+",
        choices=["inverse", "sqrt", "log"],
        help="(deprecated)\n"
        "inverse: [1/N]\n"
        "sqrt: [1/sqrt(N)]\n"
        "log: [1/log10(N)] as additional input feature",
    )
    parser.add_argument(
        "--label",
        default="avalanche_centrality",
        choices=["avalanche_fraction", "failure_fraction", "avalanche_centrality"],
        help="Which quantity to predict",
    )
    parser.add_argument(
        "--scaler",
        default=None,
        choices=[None, "QuantileScaler"],
        help="Which scaler to use scaling label.\n"
        "If none, predict value, if quantile scaler, predict rank.",
    )

    # GNN
    parser.add_argument(
        "--model", default="SkipGINNet", choices=["SkipGINNet"], help="GNN model name"
    )
    parser.add_argument("--num_layer", type=int, default=8, help="Number of GNN layer")
    parser.add_argument(
        "--hidden_feature", type=int, default=128, help="Dimension for hidden feature"
    )
    parser.add_argument(
        "--num_parallel", type=int, default=2, help="Number of mlp layers used in GIN"
    )
    parser.add_argument(
        "--hop_length", type=int, default=4, help="How many neighbors to count at once"
    )
    parser.add_argument(
        "--num_out", type=int, default=1, help="Number of layers at regressor"
    )
    parser.add_argument(
        "--out_act",
        default="sigmoid",
        choices=["sigmoid", "hardsigmoid", "softsign", "modified_sigmoid"],
        help="Activation function for out layer",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--bn_momentum", type=float, default=0.1, help="Momentum of batch norm"
    )

    # Metrics
    parser.add_argument(
        "--networkwise",
        action="store_false",
        help="If this flag is not set, metrics are calculated networkwise and then averaged",
    )
    parser.add_argument(
        "--loss",
        default="MAE",
        choices=["MSE", "MAE", "R2"],
        help="Metric name to use as loss: back-propagation",
    )
    parser.add_argument(
        "--objective",
        default="inv_MCF",
        help="Metric name for early stopping. Include inv_ in front of metric name if you want to measure after inverse scaling.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        default="RMSprop",
        choices=["Adagrad", "Adam", "RMSprop", "SGD"],
        help="torch optimizer names",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--clip", type=float, default=float("inf"), help="Maximum norm of gradients"
    )
    # Schedular
    parser.add_argument(
        "--schedular",
        default="Cosine",
        choices=["Cosine"],
        help="Which schedular to control learning rate",
    )
    parser.add_argument(
        "--schedular_period",
        type=int,
        default=20,
        help="Period of cosine annealing shedular. When set to 0, schedular will not be used",
    )
    parser.add_argument(
        "--schedular_period_mult",
        type=float,
        default=2.0,
        help="Multiplier of period of schedular.",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="Number of epochs for warm up stage"
    )
    parser.add_argument(
        "--schedular_eta_max",
        type=float,
        default=1e-3,
        help="Base eta max used in consine annealing schedular. See detailed description at class CosineAnnealingWarmRestart",
    )
    parser.add_argument(
        "--schedular_eta_max_mult",
        type=float,
        default=1.0,
        help="Multiplier of base eta max. See detailed description at class CosineAnnealingWarmRestart",
    )

    # Early Stop
    parser.add_argument(
        "--patience",
        type=int,
        default=300,
        help="How many epochs to wait after validation loss is improved",
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        default=0.0,
        help="Minimum change of validation loss to regard as improved",
    )

    # Learning
    parser.add_argument(
        "--device",
        default="0",
        choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "0", "1", "2", "3"],
        help="device to use",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for torch random")
    parser.add_argument(
        "--epochs", type=int, default=600, help="Maximum number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")

    # Log, Save
    parser.add_argument(
        "--log_level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Level of logging",
    )
    parser.add_argument(
        "--no_tqdm", action="store_true", help="When this flag is on, do not use tqdm"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=f"{BASE_DIR}/experiment",
        help="Root directory for experiments.",
    )
    parser.add_argument(
        "--exp_id",
        type=int,
        default=None,
        help="Experiment index. When specified, load the experiemt",
    )

    # Parse the arguments and return
    if options is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args=options)


@dataclass(unsafe_hash=True)
class HyperParameter:
    """Data class to save hyper parameters"""

    # Data
    data_dir: Path = field(compare=False)
    train: list[str]
    val: list[str]
    add_degree: list[str]
    add_network_size: list[str]
    label: str
    scaler: str | None

    # GNN
    model: str
    num_layer: int
    hidden_feature: int
    num_parallel: int | None
    hop_length: int | None
    num_out: int
    out_act: str | None
    dropout: float
    bn_momentum: float | None

    # Metrics
    networkwise: bool
    loss: str
    objective: str

    # Optimizer
    optimizer: str
    learning_rate: float
    weight_decay: float
    clip: float

    # Schedular
    schedular: str
    schedular_period: int
    schedular_period_mult: float
    warmup: int
    schedular_eta_max: float
    schedular_eta_max_mult: float

    # Early Stop
    patience: int
    early_stop_delta: float

    # Learning
    device: str = field(compare=False)
    seed: int | None = field(compare=False)
    epochs: int = field(compare=False)
    batch_size: int

    # Log,save
    log_level: str = field(compare=False)
    no_tqdm: bool = field(compare=False)
    root_dir: Path = field(compare=False)
    exp_id: int | None = field(compare=False)

    def __post_init__(self) -> None:
        # When only single parameter is passed to train/val, change them to list
        self.train = [self.train] if isinstance(self.train, str) else sorted(self.train)
        self.val = [self.val] if isinstance(self.val, str) else sorted(self.val)
        self.add_degree = [] if self.add_degree is None else sorted(self.add_degree)
        self.add_network_size = (
            [] if self.add_network_size is None else sorted(self.add_network_size)
        )

        # Change string type path to Path type
        self.data_dir = Path(self.data_dir).resolve()
        self.root_dir = Path(self.root_dir).resolve()
        if self.data_dir.is_relative_to(BASE_DIR):
            self.data_dir = self.data_dir.relative_to(BASE_DIR)
        if self.root_dir.is_relative_to(BASE_DIR):
            self.root_dir = self.root_dir.relative_to(BASE_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if self.exp_id is None:
            # Create experiment index and it's corresponding directory
            self.exp_id = self._get_experiment_idx()
        else:
            try:
                # When able to load json configuration, load hyper parameters
                self._load_json()
            except FileNotFoundError:
                # When json configuration is not present, don't load hyper parameters
                pass
        # Create directory
        self.exp_dir = self.root_dir / str(self.exp_id)
        self.exp_dir.mkdir(exist_ok=True)
        self.drop_unused_param()

    def drop_unused_param(self) -> None:
        """Change value of unused hyper parameter to None"""
        if "GINNet" in self.model:
            self.hop_length = None

    def save(self) -> None:
        """Save data class to json format"""
        attributes_dict = {
            attribute: str(getattr(self, attribute))
            for attribute in self.__dataclass_fields__
        }
        with open(self.exp_dir / "hyper_parameter.json", "w") as f:
            json.dump(attributes_dict, f, indent=4)

    @classmethod
    def load(cls, exp_id: int, root_dir: str | Path = BASE_DIR / "experiment"):
        return cls(**vars(get_args([f"--root_dir={root_dir}", f"--exp_id={exp_id}"])))

    def _get_experiment_idx(self) -> int:
        """
        Scan root directory and get the current experiment index
        Return
            current experiment index
        """
        # Scan root directory and find latest index
        try:
            latest_experiment_idx = max(
                int(dir.name) for dir in self.root_dir.iterdir() if dir.is_dir()
            )
        # When no experiment is done before.
        except ValueError:
            latest_experiment_idx = 0
        return latest_experiment_idx + 1

    def _load_json(self) -> None:
        """
        Load data class from json format
        In json foramt, repr=False attributes are not present
        """

        def str_to_boolean(string: str) -> bool:
            if string == "True":
                return True
            elif string == "False":
                return False
            else:
                raise RuntimeError(f"{string} cannot be converted to boolean")

        def str_to_list_str(string: str) -> list[str]:
            replace_table = dict.fromkeys(map(ord, "[]'"), None)
            string_list = string.translate(replace_table).split(",")
            if string_list == [""]:
                return []
            else:
                return [s.strip() for s in string_list]

        # Read json file
        exp_dir = self.root_dir / str(self.exp_id)
        with open(exp_dir / "hyper_parameter.json", "r") as f:
            attributes_dict = json.load(f)

        # Save json to values of instance
        for key, value in attributes_dict.items():
            attribute_type = self._get_type(key)
            if value == "None":
                setattr(self, key, None)
            elif key in ["train", "val", "add_degree", "add_network_size"]:
                setattr(self, key, str_to_list_str(value))
            elif attribute_type.__name__ == "bool":
                setattr(self, key, str_to_boolean(value))
            elif value == "inf":
                setattr(self, key, float("inf"))
            else:
                setattr(self, key, attribute_type(value))

    def _get_type(self, key: str) -> Type:
        if key in ["scaler", "out_act"]:
            return str
        elif key in ["bn_momentum"]:
            return float
        elif key in ["seed", "exp_id"]:
            return int
        return type(getattr(self, key))

    def __str__(self) -> str:
        """Print attributes of hyper parameter in json format"""
        str_list: list[str] = []

        # Data
        str_list.append(self._attribute_group_to_string("Data"))
        str_list.extend(
            self._attribute_list_to_string(
                [
                    "data_dir",
                    "train",
                    "val",
                    "add_degree",
                    "add_network_size",
                    "label",
                    "scaler",
                ]
            )
        )

        # GNN
        str_list.append(self._attribute_group_to_string("GNN"))
        str_list.extend(
            self._attribute_list_to_string(
                [
                    "model",
                    "num_layer",
                    "hidden_feature",
                    "num_parallel",
                    "hop_length",
                    "num_out",
                    "out_act",
                    "dropout",
                    "bn_momentum",
                ]
            )
        )

        # Metrics
        str_list.append(self._attribute_group_to_string("Metric"))
        str_list.extend(
            self._attribute_list_to_string(["networkwise", "loss", "objective"])
        )

        # Optimizer
        str_list.append(self._attribute_group_to_string("Optimizer"))
        str_list.extend(
            self._attribute_list_to_string(
                ["optimizer", "learning_rate", "weight_decay", "clip"]
            )
        )

        # Schedular
        str_list.append(self._attribute_group_to_string("Schedular"))
        str_list.extend(
            self._attribute_list_to_string(
                [
                    "schedular",
                    "schedular_period",
                    "schedular_period_mult",
                    "warmup",
                    "schedular_eta_max",
                    "schedular_eta_max_mult",
                ]
            )
        )

        # Early Stop
        str_list.append(self._attribute_group_to_string("Early Stop"))
        str_list.extend(
            self._attribute_list_to_string(["patience", "early_stop_delta"])
        )

        # Learning
        str_list.append(self._attribute_group_to_string("Overall learning parameters"))
        str_list.extend(
            self._attribute_list_to_string(["device", "seed", "epochs", "batch_size"])
        )

        # Log,save related
        str_list.append(self._attribute_group_to_string("Log, Save"))
        str_list.extend(
            self._attribute_list_to_string(
                ["log_level", "no_tqdm", "root_dir", "exp_id"]
            )
        )

        return "\n".join(str_list)

    @staticmethod
    def _attribute_group_to_string(group_name: str, width: int = 40) -> str:
        group_name = f" {group_name} "
        return group_name.center(width - 2, "-")

    def _attribute_list_to_string(self, attr_list: list[str]) -> list[str]:
        return [f"{attr}: {getattr(self, attr)}" for attr in attr_list]


if __name__ == "__main__":
    print(BASE_DIR / "data")
    print(Path("..").resolve().relative_to(BASE_DIR))
    # print("This is module hyper parameter")

    # hp = HyperParameter(**vars(get_args([])))
    # print(hp)
