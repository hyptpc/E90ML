import json
from pathlib import Path
from typing import Iterable, Optional, Tuple
import itertools
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


def calculate_physics_features(df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, list]:
    """
    Add opening-angle features (cos theta) assuming ux, uy, uz are already unit vectors.
    Expects feature_cols ordered as [t0_ux, t0_uy, t0_uz, t0_dedx, t1_ux, ...].
    """
    vars_per_track = 4  # ux, uy, uz, dedx
    track_vectors = []
    for idx in range(0, len(feature_cols), vars_per_track):
        u_cols = feature_cols[idx : idx + 3]
        if len(u_cols) < 3:
            break
        track_vectors.append((idx // vars_per_track, u_cols))

    if len(track_vectors) < 2:
        print("Warning: Not enough track vectors to calc angles. Skipping.")
        return df, feature_cols

    new_features = []
    for (i, u_cols_i), (j, u_cols_j) in itertools.combinations(track_vectors, 2):
        # Unit vectors => dot product equals cos theta
        dot_product = (
            df[u_cols_i[0]] * df[u_cols_j[0]]
            + df[u_cols_i[1]] * df[u_cols_j[1]]
            + df[u_cols_i[2]] * df[u_cols_j[2]]
        )
        col_name = f"ang_cos_t{i+1}t{j+1}"
        # Clip to [-1, 1] and cast to float32 immediately to save memory
        df[col_name] = dot_product.clip(-1.0, 1.0).astype(np.float32)
        new_features.append(col_name)

    if new_features:
        print(f"Added physics features: {new_features}")
    return df, feature_cols + new_features


def get_augmented_feature_columns(feature_cols: list) -> list:
    """Return feature list including physics angle features."""
    vars_per_track = 4
    track_vectors = []
    for idx in range(0, len(feature_cols), vars_per_track):
        u_cols = feature_cols[idx : idx + 3]
        if len(u_cols) < 3:
            break
        track_vectors.append((idx // vars_per_track, u_cols))

    if len(track_vectors) < 2:
        return list(feature_cols)

    new_features = []
    for (i, _), (j, _) in itertools.combinations(track_vectors, 2):
        new_features.append(f"ang_cos_t{i+1}t{j+1}")
    return list(feature_cols) + new_features


def load_data(
    files: list,
    tree_name: str,
    features: list,
    label_column: str,
    label_mapping: Optional[dict],
    fraction: float,
    random_state: Optional[int],
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load ROOT files iteratively to save memory.
    Returns (X, y, num_classes) as numpy arrays, not a DataFrame.
    """
    feature_cols = list(features)
    X_list = []
    y_list = []
    unique_labels = set()
    final_feature_cols: list = []

    for idx_file, fpath in enumerate(files):
        print(f"Loading file {idx_file + 1}/{len(files)}: {Path(fpath).name} ...")
        try:
            with uproot.open(fpath) as file:
                df = file[tree_name].arrays(feature_cols + [label_column], library="pd")
        except Exception as exc:
            print(f"Skipping {fpath} due to error: {exc}")
            continue

        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {fpath}.")

        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in {fpath}: {missing}")

        df, current_cols = calculate_physics_features(df, feature_cols)
        if not final_feature_cols:
            final_feature_cols = current_cols

        X_part = df[final_feature_cols].values.astype(np.float32)
        y_part = df[label_column].values.astype(np.int64)
        del df

        if fraction < 1.0:
            n_samples = len(y_part)
            n_keep = max(1, int(n_samples * fraction))
            if shuffle and random_state is not None:
                rng = np.random.default_rng(random_state + idx_file)
                indices = rng.choice(n_samples, n_keep, replace=False)
                X_part = X_part[indices]
                y_part = y_part[indices]
            else:
                X_part = X_part[:n_keep]
                y_part = y_part[:n_keep]

        X_list.append(X_part)
        y_list.append(y_part)
        unique_labels.update(np.unique(y_part))
        gc.collect()

    if not X_list:
        raise ValueError("No data loaded.")

    print("Concatenating arrays...")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    del X_list, y_list
    gc.collect()

    if shuffle:
        print("Shuffling final dataset...")
        rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()
        indices = rng.permutation(len(y))
        X = X[indices]
        y = y[indices]

    if label_mapping:
        sig_labels = set(label_mapping.get("signal_labels", []))
        bg_labels = set(label_mapping.get("background_labels", []))

        new_y = np.full_like(y, -1)
        mask_sig = np.isin(y, list(sig_labels))
        mask_bg = np.isin(y, list(bg_labels))
        new_y[mask_sig] = 1
        new_y[mask_bg] = 0
        if np.any(new_y == -1):
            raise ValueError("Found labels not in signal_labels or background_labels.")
        y = new_y
        num_classes = 2
    else:
        num_classes = len(unique_labels)

    print(f"Data loaded: Shape {X.shape}, Mem usage approx {X.nbytes / 1024**2:.1f} MB")
    return X, y, num_classes


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
