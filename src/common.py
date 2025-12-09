import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
import uproot
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset

# Default column names expected in the ROOT trees
DEFAULT_FEATURES = [
    "t0_theta",
    "t0_phi",
    "t0_dedx",
    "t1_theta",
    "t1_phi",
    "t1_dedx",
    "t2_theta",
    "t2_phi",
    "t2_dedx",
]
DEFAULT_TREE_NAME = "train_data"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_RANDOM_STATE = 42
DEFAULT_INPUT_DIR = Path("../../data/input")
DEFAULT_OUTPUT_DIR = Path("../../data/output")
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"
DEFAULT_TUNE_DIR = Path("../tune")
DEFAULT_PTH_DIR = Path("../pth")

DEFAULT_TUNED_PARAMS_NAME = "tuned_params.json"
DEFAULT_TUNING_SUMMARY_NAME = "tuning_trials.csv"
DEFAULT_MODEL_NAME = "model.pth"
DEFAULT_METRICS_NAME = "metrics.json"
DEFAULT_PREDICTIONS_NAME = "predictions.csv"
DEFAULT_REACTION_LABELS = {"SigmaNCusp": 1, "QFLambda": 2, "QFSigmaZ": 3}
DEFAULT_LABEL_MAPPING = {
    "signal_labels": [DEFAULT_REACTION_LABELS["SigmaNCusp"]],
    "background_labels": [
        DEFAULT_REACTION_LABELS["QFLambda"],
        DEFAULT_REACTION_LABELS["QFSigmaZ"],
    ],
}


def _ensure_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_paths(values: Iterable[str], base_dir: Path) -> list:
    paths = []
    for path in _ensure_list(values):
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        paths.append(str(p))
    return paths


def _resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def resolve_data_files(data_cfg: dict, base_dir: Path) -> list:
    """
    Resolve input file paths using a shared input directory by default.
    """
    files_cfg = data_cfg.get("files", [])
    input_dir_cfg = data_cfg.get("input_dir", DEFAULT_INPUT_DIR)
    input_dir = _resolve_path(input_dir_cfg, base_dir)

    if isinstance(files_cfg, dict):
        file_names = list(files_cfg.values())
    else:
        file_names = _ensure_list(files_cfg)

    if not file_names:
        return []

    resolved = _resolve_paths(file_names, input_dir)
    missing = [p for p in resolved if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Data file(s) not found: {missing}")
    return resolved


def resolve_named_path(
    value: Optional[str],
    default_dir: Path,
    default_name: str,
    base_dir: Path,
) -> Path:
    """
    Resolve a path where the config can pass just a filename (no slashes)
    and we place it under a default directory.
    """
    default_dir = _resolve_path(default_dir, base_dir)
    if not value:
        return (default_dir / default_name).resolve()

    value_path = Path(value)
    if value_path.name == value and value_path.parent == Path("."):
        return (default_dir / value_path.name).resolve()

    return _resolve_path(value, base_dir)


def resolve_named_dir(
    value: Optional[str],
    default_dir: Path,
    base_dir: Path,
) -> Path:
    """
    Resolve a directory where the config can pass just a directory name.
    """
    default_dir = _resolve_path(default_dir, base_dir)
    if not value:
        return default_dir

    value_path = Path(value)
    if value_path.name == value and value_path.parent == Path("."):
        return (default_dir / value_path.name).resolve()

    return _resolve_path(value, base_dir)


def load_config(config_path: str):
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        if path.suffix.lower() in {".yml", ".yaml"}:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    return config or {}, path.parent.resolve()


def resolve_device(device_pref=None) -> torch.device:
    device_str: Optional[str] = None
    if isinstance(device_pref, dict):
        device_str = device_pref.get("device")
    elif isinstance(device_pref, str):
        device_str = device_pref

    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class E90Dataset(TorchDataset):
    def __init__(
        self,
        files,
        tree_name: str = DEFAULT_TREE_NAME,
        features=None,
        label_column: str = DEFAULT_LABEL_COLUMN,
        label_mapping: Optional[dict] = None,
        fraction: float = 1.0,
        scaler: Optional[StandardScaler] = None,
        is_train: bool = True,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        """
        label_mapping: optional dict with keys:
            signal_labels: list of labels to map to 1
            background_labels: list of labels to map to 0
        """
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be between 0 and 1.")

        feature_cols = features or DEFAULT_FEATURES
        dfs = []

        for fpath in _ensure_list(files):
            with uproot.open(fpath) as file:
                df = file[tree_name].arrays(library="pd")
                if label_column not in df.columns:
                    raise ValueError(f"Label column '{label_column}' not found in {fpath}.")
                missing = [col for col in feature_cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing feature columns in {fpath}: {missing}")
                dfs.append(df[feature_cols + [label_column]])

        self.data = pd.concat(dfs, ignore_index=True)

        if fraction < 1.0:
            self.data = (
                self.data.sample(frac=fraction, random_state=random_state)
                .reset_index(drop=True)
            )
        else:
            self.data = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Optional label remapping for binary classification
        self.label_mapping = label_mapping
        if label_mapping:
            sig_labels = set(label_mapping.get("signal_labels", []))
            bg_labels = set(label_mapping.get("background_labels", []))
            if not sig_labels or not bg_labels:
                raise ValueError("label_mapping must define non-empty signal_labels and background_labels.")
            def map_label(x):
                if x in sig_labels:
                    return 1
                if x in bg_labels:
                    return 0
                raise ValueError(f"Label {x} not in signal_labels or background_labels.")
            self.data[label_column] = self.data[label_column].apply(map_label)

        self.X = self.data[feature_cols].values.astype(np.float32)
        self.y = self.data[label_column].values.astype(np.int64)
        self.num_classes = 2 if label_mapping else int(np.unique(self.y).size)

        if is_train:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            if self.scaler:
                self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def create_model_from_params(params: dict, input_dim: int, num_classes: int) -> torch.nn.Sequential:
    import torch.nn as nn

    n_layers = int(params["n_layers"])
    dropout_rate = float(params["dropout_rate"])
    hidden_units = int(params["hidden_units"])

    layers = []
    in_features = input_dim

    for _ in range(n_layers):
        layers.append(nn.Linear(in_features, hidden_units))
        layers.append(nn.BatchNorm1d(hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = hidden_units

    out_features = 1 if num_classes == 2 else num_classes
    layers.append(nn.Linear(in_features, out_features))
    return nn.Sequential(*layers)
