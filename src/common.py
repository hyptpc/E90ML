import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import uproot
import gc
from sklearn.metrics import f1_score
from torch.utils.data import Dataset as TorchDataset

# Standard output directories
DEFAULT_TUNE_DIR = Path("../tune")
DEFAULT_PTH_DIR = Path("../pth")
DEFAULT_INPUT_DIR = Path("../../data/input")
DEFAULT_OUTPUT_DIR = Path("../../data/output")
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"
TUNE_DIR = DEFAULT_TUNE_DIR
PTH_DIR = DEFAULT_PTH_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PLOTS_DIR = DEFAULT_PLOTS_DIR

DEFAULT_PLOT_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "figure.subplot.left": 0.12,
    "figure.subplot.right": 0.8,
    "figure.subplot.top": 0.88,
    "figure.subplot.bottom": 0.12,
}

# Default label mapping (SigmaNCusp=1, QFLambda=2, QFSigmaZ=3)
DEFAULT_REACTION_LABELS = {"SigmaNCusp": 1, "QFLambda": 2, "QFSigmaZ": 3}
DEFAULT_LABEL_MAPPING = {
    "signal_labels": [DEFAULT_REACTION_LABELS["SigmaNCusp"]],
    "background_labels": [
        DEFAULT_REACTION_LABELS["QFLambda"],
        DEFAULT_REACTION_LABELS["QFSigmaZ"],
    ],
}
LABEL_MAPPING = DEFAULT_LABEL_MAPPING


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


def resolve_dir(value: str, default_dir: Path, base_dir: Path) -> Path:
    """
    If value is a bare filename, place it under default_dir. Otherwise resolve relative to config.
    """
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return _resolve_path(value, base_dir)
    return _resolve_path(default_dir / candidate.name, base_dir)


def resolve_data_files(data_cfg: dict, base_dir: Path) -> list:
    """Resolve the list of ROOT files to load, preferring the shared input directory."""
    files_cfg = data_cfg.get("files")
    if not files_cfg:
        return []

    input_dir = data_cfg.get("input_dir", DEFAULT_INPUT_DIR)
    input_dir = _resolve_path(str(input_dir), base_dir)

    if isinstance(files_cfg, dict):
        file_paths = list(files_cfg.values())
    else:
        file_paths = _ensure_list(files_cfg)

    resolved = _resolve_paths(file_paths, input_dir)
    missing = [p for p in resolved if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Data file(s) not found: {missing}")
    return resolved


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


def get_config_value(cfg: dict, *keys: str) -> Optional[str]:
    """Return the first non-empty config value from the provided keys."""
    for key in keys:
        value = cfg.get(key)
        if value not in (None, ""):
            return value
    return None


def apply_plot_style(overrides: Optional[dict] = None):
    """Apply shared matplotlib rcParams with optional overrides."""
    import matplotlib as mpl

    params = dict(DEFAULT_PLOT_STYLE)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    mpl.rcParams.update(params)


def compute_f1(y_true, y_pred, num_classes: int):
    """
    Compute F1-score for binary (binary average) or multiclass (macro average) cases.
    """
    average = "binary" if num_classes == 2 else "macro"
    return f1_score(y_true, y_pred, average=average, zero_division=0)


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


def _resolve_seed(local_seed, global_seed):
    """
    Returns int seed if provided, otherwise None to let ops be random.
    Accepts empty string as 'no seed'.
    """
    seed = local_seed if local_seed not in (None, "") else global_seed
    if seed in (None, ""):
        return None
    return int(seed)


def load_data(
    files: list,
    tree_name: str,
    features: list,
    label_column: str,
    label_mapping: Optional[dict],
    fraction: float,
    random_state: Optional[int],
) -> Tuple[pd.DataFrame, int]:
    """
    Load ROOT files into a single DataFrame, optionally downsample, and remap labels.
    Returns a tuple of (dataframe, num_classes).
    """
    feature_cols = features
    dfs = []

    for fpath in files:
        with uproot.open(fpath) as file:
            df = file[tree_name].arrays(library="pd")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in {fpath}.")

            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns in {fpath}: {missing}")

            dfs.append(df[feature_cols + [label_column]])


    data = pd.concat(dfs, ignore_index=True)

    if fraction < 1.0:
        data = data.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
    else:
        data = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if label_mapping:
        sig_labels = set(label_mapping.get("signal_labels", []))
        bg_labels = set(label_mapping.get("background_labels", []))

        def map_label(x):
            if x in sig_labels:
                return 1
            if x in bg_labels:
                return 0
            raise ValueError(f"Label {x} not in signal_labels or background_labels.")

        data[label_column] = data[label_column].apply(map_label)
        num_classes = 2
    else:
        num_classes = int(np.unique(data[label_column]).size)

    return data, num_classes


class E90Dataset(TorchDataset):
    """
    A simple Dataset wrapper for Tensor data.
    Does NOT handle file loading or scaling logic to avoid leakage.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
