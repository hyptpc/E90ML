"""Plot input track vectors for each ROOT sample as 3D scatter plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ===== Editable defaults =====
CONFIG_PATH = PROJECT_ROOT / "param" / "usr" / "demo.yaml"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_PATH = PROJECT_ROOT / "ana" / "fig" / "input_vectors_t.png"
OUTPUT_PID_PATH = PROJECT_ROOT / "ana" / "fig" / "input_vectors_p.png"  # Set to None to skip
TREE_NAME_OVERRIDE = None  # Set a string to override config tree_name; keep None to use config
MAX_EVENTS = 300
# =============================

TRACKS = ("t0", "t1", "t2")
TRACK_COLORS = {"t0": "tab:red", "t1": "tab:green", "t2": "tab:blue"}
# PDG labels from TPCMlFeature.cc: 0=scat pi (primary), 1=decay proton, 2=decay pi.
PID_COLORS = {0: "tab:purple", 1: "tab:pink", 2: "tab:orange"}
PID_LABELS = {0: "scatter pion", 1: "decay proton", 2: "decay pion"}
PLOT_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.subplot.left": 0.12,
    "figure.subplot.right": 0.8,
    "figure.subplot.top": 0.88,
    "figure.subplot.bottom": 0.12,
}


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_input_files(config: dict, input_dir: Path) -> List[Tuple[str, Path]]:
    files: Dict[str, str] = config["data"]["files"]
    ordered: List[Tuple[str, Path]] = []
    for sample_name, filename in files.items():
        candidate = (input_dir / filename).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Missing input file: {candidate}")
        ordered.append((sample_name, candidate))
    return ordered


def load_vectors(
    root_path: Path, tree_name: str, tracks: Iterable[str], max_events: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    columns = [
        f"{track}_{suffix}" for track in tracks for suffix in ("ux", "uy", "uz", "dedx", "pdg")
    ]
    with uproot.open(root_path) as root_file:
        tree = root_file[tree_name]
        arrays = tree.arrays(columns, library="np", entry_stop=max_events)

    vectors: Dict[str, np.ndarray] = {}
    pids: Dict[str, np.ndarray] = {}
    for track in tracks:
        ux = arrays[f"{track}_ux"]
        uy = arrays[f"{track}_uy"]
        uz = arrays[f"{track}_uz"]
        dedx = arrays[f"{track}_dedx"]
        vectors[track] = np.column_stack((ux * dedx, uy * dedx, uz * dedx))
        pids[track] = arrays[f"{track}_pdg"]
    return vectors, pids


def summarize_tracks(track_vectors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for track, vectors in track_vectors.items():
        lengths = np.linalg.norm(vectors, axis=1) if len(vectors) else np.array([])
        summary[track] = {
            "total": len(vectors),
            "nonzero": int(np.count_nonzero(lengths)),
        }
    return summary


def _apply_axes_style(ax, sample_name: str, max_length: float) -> None:
    def _nice_limit(val: float) -> float:
        """Round up to a 1-2-5 step for cleaner tick labels."""
        if val <= 0:
            return 1.0
        exponent = np.floor(np.log10(val))
        fraction = val / (10.0 ** exponent)
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
        return nice_fraction * (10.0 ** exponent)

    tick_max = _nice_limit(max_length)
    ax.set_title(f"{sample_name}", fontsize=14, pad=10)
    ax.set_xlim(-tick_max, tick_max)
    ax.set_ylim(-tick_max, tick_max)
    ax.set_zlim(-tick_max, tick_max)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=32)

    tick_positions = np.linspace(-tick_max, tick_max, 5)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_zticks(tick_positions)
    ax.set_xlabel("ux * dE/dx")
    ax.set_ylabel("uy * dE/dx")
    ax.set_zlabel("uz * dE/dx")
    ax.grid(True, color="0.7", linewidth=0.7, linestyle="--", alpha=0.6)
    # Keep a subtle gray background while showing ticks/titles.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.92, 0.92, 0.92, 1.0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))


def plot_tracks(
    ax, sample_name: str, track_vectors: Dict[str, np.ndarray], stats: Dict[str, Dict[str, int]]
) -> None:
    lengths: List[np.ndarray] = []
    for vectors in track_vectors.values():
        if len(vectors) == 0:
            continue
        lengths.append(np.linalg.norm(vectors, axis=1))
    max_length = float(np.percentile(np.concatenate(lengths), 99)) if lengths else 1.0
    max_length = max(max_length, 1e-6)

    for track, vectors in track_vectors.items():
        if len(vectors) == 0:
            continue
        color = TRACK_COLORS.get(track, "gray")
        ax.scatter(
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            color=color,
            alpha=0.6,
            s=8,
            depthshade=False,
        )

    _apply_axes_style(ax, sample_name, max_length)

    handles = [
        plt.Line2D([0], [0], color=TRACK_COLORS[track], lw=2, label=track) for track in TRACKS
    ]
    ax.legend(handles=handles, loc="upper left", fontsize="small", framealpha=0.75)


def plot_tracks_by_pid(
    ax, sample_name: str, track_vectors: Dict[str, np.ndarray], track_pids: Dict[str, np.ndarray]
) -> None:
    lengths: List[np.ndarray] = []
    for vectors in track_vectors.values():
        if len(vectors) == 0:
            continue
        lengths.append(np.linalg.norm(vectors, axis=1))
    max_length = float(np.percentile(np.concatenate(lengths), 99)) if lengths else 1.0
    max_length = max(max_length, 1e-6)

    for track, vectors in track_vectors.items():
        pid_vals = track_pids.get(track)
        if len(vectors) == 0 or pid_vals is None:
            continue
        for pid_code in np.unique(pid_vals):
            color = PID_COLORS.get(int(pid_code), "k")
            mask = pid_vals == pid_code
            ax.scatter(
                vectors[mask, 0],
                vectors[mask, 1],
                vectors[mask, 2],
                color=color,
                alpha=0.6,
                s=8,
                depthshade=False,
            )

    _apply_axes_style(ax, f"{sample_name}", max_length)

    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=PID_LABELS.get(code, str(code)))
        for code, color in PID_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize="small", framealpha=0.75)


def main() -> None:
    config = load_config(CONFIG_PATH)
    tree_name = TREE_NAME_OVERRIDE or config["data"].get("tree_name", "g4s2s")
    samples = resolve_input_files(config, INPUT_DIR)

    datasets = []
    for sample_name, path in samples:
        vectors, pids = load_vectors(path, tree_name, TRACKS, MAX_EVENTS)
        stats = summarize_tracks(vectors)
        datasets.append((sample_name, vectors, pids, stats))

    plt.rcParams.update(PLOT_STYLE)

    fig, axes = plt.subplots(
        1, len(datasets), subplot_kw={"projection": "3d"}, figsize=(5 * len(datasets), 5)
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, (sample_name, vectors, pids, stats) in zip(axes, datasets):
        plot_tracks(ax, sample_name, vectors, stats)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)

    if OUTPUT_PID_PATH:
        fig_pid, axes_pid = plt.subplots(
            1, len(datasets), subplot_kw={"projection": "3d"}, figsize=(5 * len(datasets), 5)
        )
        if len(datasets) == 1:
            axes_pid = [axes_pid]

        for ax, (sample_name, vectors, pids, _) in zip(axes_pid, datasets):
            plot_tracks_by_pid(ax, sample_name, vectors, pids)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        OUTPUT_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig_pid.savefig(OUTPUT_PID_PATH, dpi=200)


if __name__ == "__main__":
    main()
