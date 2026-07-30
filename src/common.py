import json
import os
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


def resolve_data_dirs(base_dir: Path) -> Tuple[Path, Path]:
    """Resolve and create environment-specific input and output directories."""
    data_root_raw = os.environ.get("E90ML_DATA_DIR")
    if data_root_raw:
        data_root = _resolve_path(str(data_root_raw), base_dir)
        input_dir = data_root / "input"
        output_dir = data_root / "output"
    else:
        input_dir = _resolve_path(str(DEFAULT_INPUT_DIR), base_dir)
        output_dir = _resolve_path(str(DEFAULT_OUTPUT_DIR), base_dir)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def resolve_data_files(data_cfg: dict, base_dir: Path) -> list:
    """Resolve configured ROOT files below the environment-specific input directory."""
    files_cfg = data_cfg.get("files")
    if not files_cfg:
        return []

    input_dir, _ = resolve_data_dirs(base_dir)

    if isinstance(files_cfg, dict):
        file_paths = list(files_cfg.values())
    else:
        file_paths = _ensure_list(files_cfg)

    resolved = _resolve_paths(file_paths, input_dir)
    missing = [p for p in resolved if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Data file(s) not found: {missing}")
    return resolved


def _merge_config(base: dict, override: dict) -> dict:
    """Recursively merge a child config into its base config."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_path(path: Path, seen: set) -> dict:
    path = path.expanduser().resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"Circular config inheritance detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        if path.suffix.lower() in {".yml", ".yaml"}:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    config = config or {}

    parent_raw = config.pop("extends", None)
    if not parent_raw:
        return config
    parent_path = Path(parent_raw).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent_config = _load_config_path(parent_path, seen | {path})
    return _merge_config(parent_config, config)


def load_config(config_path: str):
    path = Path(config_path).expanduser().resolve()
    return _load_config_path(path, set()), path.parent


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
    if device_str == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_seed(local_seed, global_seed):
    """
    Returns int seed if provided, otherwise None to let ops be random.
    Accepts empty string as 'no seed'.
    """
    seed = local_seed if local_seed not in (None, "") else global_seed
    if seed in (None, ""):
        return None
    return int(seed)


def validate_fraction(fraction: float, name: str = "fraction") -> float:
    """Validate a positive data fraction no greater than one."""
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1], got {fraction}.")
    return fraction


def validate_stratified_split(labels: np.ndarray, val_split: float, context: str) -> float:
    """Validate that a stratified train/validation split can contain every class."""
    val_split = float(val_split)
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"{context}.val_split must be in the interval (0, 1), got {val_split}.")

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"{context} requires at least two classes, found {classes.tolist()}.")
    too_small = {int(label): int(count) for label, count in zip(classes, counts) if count < 2}
    if too_small:
        raise ValueError(
            f"{context} stratified split requires at least two samples per class; "
            f"insufficient classes: {too_small}."
        )

    train_size = int(np.floor(len(labels) * (1.0 - val_split)))
    val_size = len(labels) - train_size
    if train_size < len(classes) or val_size < len(classes):
        raise ValueError(
            f"{context}.val_split={val_split} produces train/validation sets too small "
            f"for {len(classes)} classes."
        )
    return val_split


def batchnorm_safe_drop_last(dataset_size: int, batch_size: int) -> bool:
    """Avoid a one-sample training batch, which BatchNorm1d cannot process."""
    if dataset_size < 2:
        raise ValueError("Training requires at least two samples when the model uses BatchNorm1d.")
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2 when the model uses BatchNorm1d.")
    return dataset_size > batch_size and dataset_size % batch_size == 1


def make_event_groups(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Hash model inputs and labels so identical rows remain in the same split."""
    feature_values = np.ascontiguousarray(features, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.int64)
    if feature_values.ndim != 2 or len(feature_values) != len(label_values):
        raise ValueError("Features and labels must have compatible row counts.")

    groups = np.full(len(label_values), np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    feature_bits = feature_values.view(np.uint32).reshape(feature_values.shape)
    for column in range(feature_bits.shape[1]):
        groups = (groups ^ feature_bits[:, column].astype(np.uint64)) * prime
    groups = (groups ^ label_values.view(np.uint64)) * prime
    return groups


def stratified_group_split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    val_split: float,
    random_state: Optional[int],
    context: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a reproducible stratified split without sharing identical rows."""
    val_split = validate_stratified_split(labels, val_split, context)
    reciprocal = 1.0 / val_split
    n_splits = int(round(reciprocal))
    if n_splits < 2 or not np.isclose(reciprocal, n_splits):
        raise ValueError(
            f"{context}.val_split must be the reciprocal of an integer for grouped "
            f"stratification, got {val_split}."
        )

    for label in np.unique(labels):
        group_count = np.unique(groups[np.asarray(labels) == label]).size
        if group_count < n_splits:
            raise ValueError(
                f"{context} requires at least {n_splits} distinct groups for class "
                f"{int(label)}, found {group_count}."
            )

    # Assign every identical-event group to one deterministic pseudo-random
    # fold. This is linear in the number of rows and remains practical for
    # multi-million-event datasets.
    seed = np.uint64(0 if random_state is None else int(random_state))
    mixed = groups + seed + np.uint64(0x9E3779B97F4A7C15)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    mixed = mixed ^ (mixed >> np.uint64(31))
    fold_ids = mixed % np.uint64(n_splits)

    labels = np.asarray(labels)
    val_mask = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        class_mask = labels == label
        fold_counts = np.bincount(
            fold_ids[class_mask].astype(np.int64),
            minlength=n_splits,
        )
        target_count = int(round(class_mask.sum() * val_split))
        selected_fold = int(np.argmin(np.abs(fold_counts - target_count)))
        val_mask |= class_mask & (fold_ids == selected_fold)

    val_indices = np.flatnonzero(val_mask)
    train_indices = np.flatnonzero(~val_mask)
    overlap = np.intersect1d(
        np.unique(groups[train_indices]),
        np.unique(groups[val_indices]),
        assume_unique=True,
    )
    if overlap.size:
        raise RuntimeError(f"{context} produced {overlap.size} overlapping event groups.")
    return train_indices, val_indices


def sample_indices(size: int, fraction: float, random_state: Optional[int]) -> np.ndarray:
    """Sample row indices without replacement while preserving the original order."""
    fraction = validate_fraction(fraction)
    if fraction == 1.0:
        return np.arange(size)
    n_keep = max(1, min(size, int(round(size * fraction))))
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(size, size=n_keep, replace=False))


def summarize_labels(labels: np.ndarray) -> dict:
    """Return JSON-serializable total and per-class event counts."""
    classes, counts = np.unique(np.asarray(labels), return_counts=True)
    return {
        "total": int(len(labels)),
        "class_counts": {str(label): int(count) for label, count in zip(classes, counts)},
    }


def save_data_summary(summary: dict, output_path: Path) -> None:
    """Save and print the event counts used by a pipeline stage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Data summary: {json.dumps(summary, sort_keys=True)}")
    print(f"Saved data summary to '{output_path}'.")


def load_data(
    files: list,
    tree_name: str,
    features: list,
    label_column: str,
    label_mapping: Optional[dict],
    fraction: float,
    random_state: Optional[int],
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Load ROOT files into a single DataFrame, optionally downsample, and remap labels.
    Returns a tuple of (dataframe, num_classes).
    """
    if not files:
        raise ValueError("At least one input file is required.")
    fraction = validate_fraction(fraction)
    feature_cols = list(features)
    if not feature_cols:
        raise ValueError("At least one feature column is required.")
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Feature columns must not contain duplicates.")
    if label_column in feature_cols:
        raise ValueError(
            f"Label column '{label_column}' must not be included in feature columns (target leakage)."
        )
    requested_columns = list(dict.fromkeys(feature_cols + [label_column]))
    dfs = []

    for fpath in files:
        with uproot.open(fpath) as file:
            if tree_name not in file:
                raise KeyError(f"Tree '{tree_name}' not found in {fpath}.")
            tree = file[tree_name]
            available = {str(key).split(";")[0] for key in tree.keys()}
            missing = [col for col in requested_columns if col not in available]
            if missing:
                raise ValueError(f"Missing columns in {fpath}: {missing}")
            df = tree.arrays(requested_columns, library="pd")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in {fpath}.")

            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns in {fpath}: {missing}")

            dfs.append(df[requested_columns])

    data = pd.concat(dfs, ignore_index=True)
    if data.empty:
        raise ValueError("No events were loaded from the configured input files.")

    if shuffle:
        data = data.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
    else:
        if fraction < 1.0:
            n_keep = max(1, int(len(data) * fraction))
            data = data.iloc[:n_keep].reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)

    if label_mapping:
        sig_labels = set(label_mapping.get("signal_labels", []))
        bg_labels = set(label_mapping.get("background_labels", []))
        overlap = sig_labels & bg_labels
        if overlap:
            raise ValueError(f"Labels cannot be both signal and background: {sorted(overlap)}.")
        known_labels = sig_labels | bg_labels
        observed_labels = set(data[label_column].unique().tolist())
        unknown_labels = observed_labels - known_labels
        if unknown_labels:
            raise ValueError(
                f"Labels not present in signal_labels or background_labels: {sorted(unknown_labels)}."
            )
        data[label_column] = np.where(data[label_column].isin(sig_labels), 1, 0).astype(np.int64)
        num_classes = 2
    else:
        num_classes = int(np.unique(data[label_column]).size)

    try:
        feature_values = data[feature_cols].to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("All configured feature columns must be numeric.") from exc
    if not np.isfinite(feature_values).all():
        invalid_counts = {}
        for index, column in enumerate(feature_cols):
            invalid_count = int((~np.isfinite(feature_values[:, index])).sum())
            if invalid_count:
                invalid_counts[column] = invalid_count
        raise ValueError(f"Feature columns contain NaN or infinite values: {invalid_counts}.")

    return data, num_classes


class E90Dataset(TorchDataset):
    """
    A simple Dataset wrapper for Tensor data.
    Does NOT handle file loading or scaling logic to avoid leakage.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_model_from_params(params: dict, input_dim: int, num_classes: int) -> torch.nn.Sequential:
    import torch.nn as nn

    n_layers = int(params["n_layers"])
    dropout_rate = float(params["dropout_rate"])
    hidden_units = int(params["hidden_units"])
    if input_dim < 1:
        raise ValueError("input_dim must be at least 1.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1.")
    if hidden_units < 1:
        raise ValueError("hidden_units must be at least 1.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the interval [0, 1).")

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
