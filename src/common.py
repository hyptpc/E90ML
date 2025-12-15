import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import uproot
import gc
from sklearn.metrics import f1_score

from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import GCNConv, global_mean_pool

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
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return _resolve_path(value, base_dir)
    return _resolve_path(default_dir / candidate.name, base_dir)


def resolve_data_files(data_cfg: dict, base_dir: Path) -> list:
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
    for key in keys:
        value = cfg.get(key)
        if value not in (None, ""):
            return value
    return None


def split_track_features(features: list) -> dict:
    """
    Split a flat feature list into track-specific slices for t0, t1, and t2.
    Expects at least 12 entries (4 features per track).
    """
    if features is None or len(features) < 12:
        raise ValueError("feature_columns must contain at least 12 entries (t0, t1, t2 each with 4 features).")
    return {
        "t0": features[0:4],
        "t1": features[4:8],
        "t2": features[8:12],
    }


def resolve_best_params_path(training_cfg: dict, tuning_cfg: dict, base_dir: Path, tune_dir: Path = TUNE_DIR) -> Path:
    """
    Resolve the path to the tuned hyperparameters file and ensure it exists.
    """
    best_params_raw = get_config_value(training_cfg, "best_params_file", "best_params_path") or get_config_value(
        tuning_cfg, "tune_params_file", "best_params_file", "best_params_path"
    )
    if not best_params_raw:
        raise ValueError("Config must set training.best_params_file or tuning.tune_params_file.")

    best_params_path = resolve_dir(best_params_raw, tune_dir, base_dir)
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best parameter file not found at {best_params_path}.")
    return best_params_path


def load_best_params(training_cfg: dict, tuning_cfg: dict, base_dir: Path, tune_dir: Path = TUNE_DIR) -> Tuple[dict, Path]:
    """
    Load tuned hyperparameters and return both the dict and resolved path.
    """
    best_params_path = resolve_best_params_path(training_cfg, tuning_cfg, base_dir, tune_dir)
    with best_params_path.open() as f:
        params = json.load(f)
    return params, best_params_path


def resolve_model_output_path(training_cfg: dict, base_dir: Path, pth_dir: Path = PTH_DIR, must_exist: bool = False) -> Path:
    """
    Resolve the model output path under pth_dir (or absolute path).
    Set must_exist=True to require a saved model to be present.
    """
    model_output_raw = get_config_value(training_cfg, "model_output_file", "model_output_path")
    if not model_output_raw:
        raise ValueError("Config must set training.model_output_file.")
    model_output_path = resolve_dir(model_output_raw, pth_dir, base_dir)
    if must_exist and not model_output_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_output_path}.")
    return model_output_path


def apply_plot_style(overrides: Optional[dict] = None):
    import matplotlib as mpl
    params = dict(DEFAULT_PLOT_STYLE)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    mpl.rcParams.update(params)


def compute_f1(y_true, y_pred, num_classes: int):
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
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, int]:
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

    if shuffle:
        if fraction < 1.0:
            data = data.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
        else:
            data = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        if fraction < 1.0:
            n_keep = max(1, int(len(data) * fraction))
            data = data.iloc[:n_keep].reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)

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


# --- GNN Classes ---

class E90GraphDataset(PyGDataset):
    """
    Dataset that converts a Pandas DataFrame into a PyG graph format.
    Each event becomes a fully connected 3-node graph (t0=ScatPi, t1, t2).
    """
    def __init__(self, df: pd.DataFrame, feature_cols_dict: dict, label_col: str):
        super().__init__()
        self.data_list = []
        self._process_dataframe(df, feature_cols_dict, label_col)

    def _process_dataframe(self, df, cols, label_col):
        # Edge index for a fully connected 3-node graph
        # Node IDs: 0=t0, 1=t1, 2=t2
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1]
        ], dtype=torch.long)

        # Pre-extract as NumPy arrays for speed
        t0_ux = df[cols['t0'][0]].values
        t0_uy = df[cols['t0'][1]].values
        t0_uz = df[cols['t0'][2]].values
        t0_dedx = df[cols['t0'][3]].values
        
        t1_ux = df[cols['t1'][0]].values
        t1_uy = df[cols['t1'][1]].values
        t1_uz = df[cols['t1'][2]].values
        t1_dedx = df[cols['t1'][3]].values
        
        t2_ux = df[cols['t2'][0]].values
        t2_uy = df[cols['t2'][1]].values
        t2_uz = df[cols['t2'][2]].values
        t2_dedx = df[cols['t2'][3]].values
        
        labels = df[label_col].values

        for i in range(len(df)):
            # Node features: [ux, uy, uz, dedx, is_scat_pi]
            # t0 (Scat Pi): is_scat_pi = 1
            node_t0 = [t0_ux[i], t0_uy[i], t0_uz[i], t0_dedx[i], 1.0]
            # t1 (Track): is_scat_pi = 0
            node_t1 = [t1_ux[i], t1_uy[i], t1_uz[i], t1_dedx[i], 0.0]
            # t2 (Track): is_scat_pi = 0
            node_t2 = [t2_ux[i], t2_uy[i], t2_uz[i], t2_dedx[i], 0.0]

            x = torch.tensor([node_t0, node_t1, node_t2], dtype=torch.float)
            y = torch.tensor([int(labels[i])], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class E90GNN(nn.Module):
    """
    Simple GCN (Graph Convolutional Network) model.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout_rate, num_classes):
        super(E90GNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout_rate = dropout_rate
        self.lin = nn.Linear(hidden_channels, 1 if num_classes == 2 else num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Message Passing (Node Embedding)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 2. Readout (Global Pooling)
        # Average node features in the graph to summarize each event
        x = global_mean_pool(x, batch)

        # 3. Classifier
        x = self.lin(x)
        return x


def create_gnn_model_from_params(params: dict, input_dim: int, num_classes: int) -> nn.Module:
    """
    Factory to build a GNN model from a hyperparameter dictionary.
    """
    return E90GNN(
        in_channels=input_dim,
        hidden_channels=int(params.get("hidden_units", 64)),
        num_layers=int(params.get("n_layers", 3)),
        dropout_rate=float(params.get("dropout_rate", 0.2)),
        num_classes=num_classes
    )
