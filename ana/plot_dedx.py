"""Plot raw vs truncated dE/dx distributions (using per-hit arrays) for each reaction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import uproot
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "param" / "usr" / "demo.yaml"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_TRACK_PATH = PROJECT_ROOT / "ana" / "fig" / "dedx_tracks.png"
OUTPUT_PDG_PATH = PROJECT_ROOT / "ana" / "fig" / "dedx_pdg.png"
TREE_NAME_OVERRIDE: str | None = None
MAX_EVENTS: int | None = None  # Set to None to read all entries

TRACKS: Sequence[str] = ("t0", "t1", "t2")
REACTIONS: Sequence[str] = ("SigmaNCusp", "QFLambda", "QFSigmaZ")
PDG_CODES: Sequence[int] = (0, 1, 2)
PDG_LABELS: Mapping[int, str] = {
    0: "scatter pion",
    1: "decay proton",
    2: "decay pion",
}
DEDX_KINDS = (
    ("dedx_raw", "Raw", "tab:blue", "-"),
    ("dedx_trunc", "Truncated", "tab:red", "-"),
)

# Keep the same style as plot_inputs.py so figures match.
PLOT_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "figure.subplot.hspace": 0.4,
    "figure.subplot.wspace": 0.25,
}

# Optional manual x-tick locations; set per-track (track0/track1/track2)
# and per-PDG (pdg0/pdg1/pdg2). When provided, ticks and x-limits follow
# these values.
CUSTOM_XTICKS: Mapping[str, Sequence[float]] = {
    "track0": [0.0, 0.025, 0.050, 0.075],
    "track1": [0.0, 0.025, 0.050, 0.075],
    "track2": [0.0, 0.025, 0.050, 0.075],
    "pdg0":   [0.0, 0.025, 0.050, 0.075],
    "pdg1":   [0.0, 0.025, 0.050, 0.075],
    "pdg2":   [0.0, 0.025, 0.050, 0.075],
}


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_file_pairs(input_dir: Path, reactions: Iterable[str]) -> Dict[str, Dict[str, Path]]:
    """Return available <reaction>_dedx.root files for each reaction."""
    pairs: Dict[str, Dict[str, Path]] = {}
    for reaction in reactions:
        path = input_dir / f"{reaction}_dedx.root"
        if path.exists():
            pairs[reaction] = {"truncated": path}
        else:
            print(f"[warn] No ROOT file found for {reaction}: expected {path}")
    return pairs


def load_sample(path: Path, tree_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    all_columns = []
    for t in TRACKS:
        all_columns.extend(
            [
                f"{t}_pdg",
                f"{t}_dedx_raw",
                f"{t}_dedx_trunc",
                f"{t}_dedx",  # fallback for older files
            ]
        )
    with uproot.open(path) as root_file:
        tree = root_file[tree_name]
        # uproot returns keys with and without trailing ";*" so normalize to bare names.
        available = {k.split(";")[0] for k in tree.keys()}
        columns = [c for c in all_columns if c in available]
        arrays = tree.arrays(columns, library="ak", how=dict, entry_stop=MAX_EVENTS)

    pdg = {t: arrays.get(f"{t}_pdg", ak.Array([])) for t in TRACKS}
    dedx_raw = {t: arrays.get(f"{t}_dedx_raw", ak.Array([])) for t in TRACKS}
    dedx_trunc = {t: arrays.get(f"{t}_dedx_trunc", ak.Array([])) for t in TRACKS}
    # Fallback: if per-hit truncated array is missing, use scalar t*_dedx when available.
    for t in TRACKS:
        if len(dedx_trunc[t]) == 0:
            dedx_trunc[t] = arrays.get(f"{t}_dedx", dedx_trunc[t])
    return {"pdg": pdg, "dedx": {"dedx_raw": dedx_raw, "dedx_trunc": dedx_trunc}}


def load_samples(
    file_pairs: Mapping[str, Mapping[str, Path]], tree_name: str
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    samples: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
    for reaction, versions in file_pairs.items():
        samples[reaction] = {}
        for version, path in versions.items():
            samples[reaction][version] = load_sample(path, tree_name)
    return samples


def _flatten(arr: ak.Array | np.ndarray | None) -> np.ndarray:
    if arr is None:
        return np.array([])
    if isinstance(arr, np.ndarray):
        return arr.ravel()
    if len(arr) == 0:
        return np.array([])
    return ak.to_numpy(ak.flatten(arr, axis=None))


def compute_bin_edges(values: List[np.ndarray], bins: int = 60) -> np.ndarray:
    combined = (
        np.concatenate([arr for arr in values if arr is not None and arr.size])
        if values
        else np.array([])
    )
    if combined.size == 0:
        return np.linspace(0, 1, 2)
    lo, hi = np.percentile(combined, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins)


def dedx_for_pdg(
    sample: Mapping[str, Dict[str, Dict[str, Mapping[str, ak.Array]]]],
    pdg_code: int,
    kind: str,
) -> np.ndarray:
    dedx_values: List[np.ndarray] = []
    pdg_map = sample.get("pdg", {})
    dedx_map = sample.get("dedx", {}).get(kind, {})
    for track in TRACKS:
        pdg_arr = pdg_map.get(track)
        dedx_arr = dedx_map.get(track)
        if pdg_arr is None or dedx_arr is None or len(pdg_arr) == 0:
            continue
        masked = dedx_arr[pdg_arr == pdg_code]
        flattened = _flatten(masked)
        if flattened.size:
            dedx_values.append(flattened)
    return np.concatenate(dedx_values) if dedx_values else np.array([])


def plot_track_histograms(samples: Mapping[str, Mapping[str, Dict[str, Dict[str, np.ndarray]]]]) -> None:
    bins_by_track = {}
    for track in TRACKS:
        all_values: List[np.ndarray] = []
        for reaction_data in samples.values():
            for sample_data in reaction_data.values():
                for kind, _, _, _ in DEDX_KINDS:
                    arr = sample_data.get("dedx", {}).get(kind, {}).get(track)
                    all_values.append(_flatten(arr))
        bins_by_track[track] = compute_bin_edges(all_values)

    fig, axes = plt.subplots(len(TRACKS), len(REACTIONS), figsize=(14, 10), sharex="row")
    axes = np.atleast_2d(axes)
    y_max: Dict[str, float] = {t: 0.0 for t in TRACKS}

    for row, track in enumerate(TRACKS):
        for col, reaction in enumerate(REACTIONS):
            ax = axes[row, col]
            reaction_data = samples.get(reaction)
            if reaction_data is None:
                ax.set_axis_off()
                continue

            sample = next(iter(reaction_data.values()), None)
            if not sample:
                ax.set_axis_off()
                continue

            for kind, kind_label, color, linestyle in DEDX_KINDS:
                values = _flatten(sample.get("dedx", {}).get(kind, {}).get(track))
                if values.size == 0:
                    continue
                counts, _, _ = ax.hist(
                    values,
                    bins=bins_by_track[track],
                    histtype="step",
                    density=False,
                    color=color,
                    linewidth=1.3,
                    linestyle=linestyle,
                    label=kind_label,
                )
                if counts.size:
                    y_max[track] = max(y_max[track], float(np.max(counts)))

            track_label = track.replace("t", "track", 1)
            ax.set_title(f"{reaction} / {track_label}")
            ax.set_xlabel("dE/dx [MeV/cm]")
            if col == 0:
                ax.set_ylabel("Counts")
            custom_ticks = CUSTOM_XTICKS.get(track_label)
            if custom_ticks:
                ax.set_xticks(custom_ticks)
                if len(custom_ticks) >= 2:
                    ax.set_xlim(min(custom_ticks), max(custom_ticks))
            # Compact tick labels for small ranges (track1/2) unless custom ticks provided.
            if row >= 1 and not custom_ticks:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)  # force plain decimals
            formatter.set_useOffset(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.legend(fontsize="small")

    # Align y-limits per track row for easier comparison.
    for row, track in enumerate(TRACKS):
        if y_max[track] > 0:
            ylim = y_max[track] * 1.05
            for col in range(len(REACTIONS)):
                axes[row, col].set_ylim(0, ylim)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTPUT_TRACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_TRACK_PATH, dpi=200)


def plot_pdg_histograms(samples: Mapping[str, Mapping[str, Dict[str, Dict[str, np.ndarray]]]]) -> None:
    bins_by_pdg: Dict[int, np.ndarray] = {}
    for pdg in PDG_CODES:
        all_values: List[np.ndarray] = []
        for reaction_data in samples.values():
            for sample_data in reaction_data.values():
                for kind, _, _, _ in DEDX_KINDS:
                    all_values.append(dedx_for_pdg(sample_data, pdg, kind))
        bins_by_pdg[pdg] = compute_bin_edges(all_values)

    fig, axes = plt.subplots(len(PDG_CODES), len(REACTIONS), figsize=(14, 10), sharex="row")
    axes = np.atleast_2d(axes)
    y_max: Dict[int, float] = {pdg: 0.0 for pdg in PDG_CODES}

    for row, pdg in enumerate(PDG_CODES):
        for col, reaction in enumerate(REACTIONS):
            ax = axes[row, col]
            reaction_data = samples.get(reaction)
            if reaction_data is None:
                ax.set_axis_off()
                continue

            sample = next(iter(reaction_data.values()), None)
            if not sample:
                ax.set_axis_off()
                continue

            for kind, kind_label, color, linestyle in DEDX_KINDS:
                values = dedx_for_pdg(sample, pdg, kind)
                if values.size == 0:
                    continue
                counts, _, _ = ax.hist(
                    values,
                    bins=bins_by_pdg[pdg],
                    histtype="step",
                    density=False,
                    color=color,
                    linewidth=1.3,
                    linestyle=linestyle,
                    label=kind_label,
                )
                if counts.size:
                    y_max[pdg] = max(y_max[pdg], float(np.max(counts)))

            pdg_label = PDG_LABELS.get(pdg, f"PDG {pdg}")
            ax.set_title(f"{reaction} / {pdg_label}")
            ax.set_xlabel("dE/dx [MeV/cm]")
            if col == 0:
                ax.set_ylabel("Counts")
            if pdg == 0:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            custom_ticks = CUSTOM_XTICKS.get(f"pdg{pdg}")
            if custom_ticks:
                ax.set_xticks(custom_ticks)
                if len(custom_ticks) >= 2:
                    ax.set_xlim(min(custom_ticks), max(custom_ticks))
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.legend(fontsize="small")

    # Align y-limits per PDG row for easier comparison.
    for row, pdg in enumerate(PDG_CODES):
        if y_max[pdg] > 0:
            ylim = y_max[pdg] * 1.05
            for col in range(len(REACTIONS)):
                axes[row, col].set_ylim(0, ylim)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTPUT_PDG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDG_PATH, dpi=200)


def main() -> None:
    config = load_config(CONFIG_PATH)
    tree_name = TREE_NAME_OVERRIDE or config["data"].get("tree_name", "g4s2s")

    file_pairs = resolve_file_pairs(INPUT_DIR, REACTIONS)
    if not file_pairs:
        raise SystemExit("No input ROOT files found.")

    samples = load_samples(file_pairs, tree_name)
    plt.rcParams.update(PLOT_STYLE)

    plot_track_histograms(samples)
    plot_pdg_histograms(samples)


if __name__ == "__main__":
    main()
