import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_feature_map(base_dir: str) -> Dict:
    """Load the feature_map.json located in the given base_dir."""
    path = os.path.join(base_dir, "feature_map.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature_map.json not found in {base_dir}")
    with open(path) as f:
        return json.load(f)


def load_process_samples(process_dir: str, max_files: int = 5) -> np.ndarray:
    """Load and concatenate a few *_x.npy files from a given process folder."""
    files = [os.path.join(process_dir, f) for f in os.listdir(process_dir) if f.endswith("_x.npy")]
    if not files:
        return np.empty((0,))
    files = sorted(files)[:max_files]
    arrays = [np.load(f) for f in files]
    return np.concatenate(arrays, axis=0)


def extract_leading_objects(
    data: np.ndarray, start: int, end: int, topk: int, n_cols: int, include_count: bool
) -> np.ndarray:
    """Extract only the leading object features (excluding count if present)."""
    # If count=True, remove the last entry from slice
    true_end = end - 1 if include_count else end
    group_slice = data[:, start:true_end]
    if topk and topk > 1:
        group_slice = group_slice.reshape(group_slice.shape[0], topk, n_cols)
        group_slice = group_slice[:, 0, :]  # take leading object only
    return group_slice


def plot_process_features(
    process_name: str,
    data: np.ndarray,
    fmap: Dict[str, Dict],
    output_dir: str,
    bins: int = 100,
):
    """Plot all features for a single process into its own folder structure."""
    proc_out = os.path.join(output_dir, process_name)
    os.makedirs(proc_out, exist_ok=True)

    for group_name, info in fmap.items():
        start, end = info["start"], info["end"]
        columns = info["columns"]
        topk = info.get("topk", None)
        n_cols = len(columns)
        include_count = info.get("count", False)

        group_out = os.path.join(proc_out, group_name)
        os.makedirs(group_out, exist_ok=True)

        # ---- Plot per-feature histograms ----
        group_slice = extract_leading_objects(data, start, end, topk, n_cols, include_count)
        if group_slice.size == 0:
            print(f"⚠️ No data for group {group_name} in {process_name}")
            continue

        for i, col in enumerate(columns):
            values = group_slice[:, i]
            plt.figure(figsize=(6, 4))
            plt.hist(values, bins=bins, histtype="stepfilled", linewidth=1.0, alpha=0.8)
            plt.title(f"{process_name} — {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(group_out, f"{col}.png"))
            plt.close()

        # ---- Plot count histogram if applicable ----
        if include_count:
            count_idx = end - 1
            if count_idx >= data.shape[1]:
                print(
                    f"⚠️ Skipping count for {group_name}: index {count_idx} out of range for {process_name}"
                )
                continue
            count_values = data[:, count_idx]
            plt.figure(figsize=(6, 4))
            plt.hist(count_values, bins=bins, histtype="stepfilled", linewidth=1.0, alpha=0.8)
            plt.title(f"{process_name} — {group_name}_count")
            plt.xlabel("count")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(group_out, f"{group_name}_count.png"))
            plt.close()


def plot_all_processes(
    base_dir: str, output_dir: str = "plots/raw_features", split: str = "train", max_files: int = 5
):
    """Load all process subfolders under the specified split and plot features."""
    fmap = load_feature_map(base_dir)
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split folder '{split_dir}' not found under {base_dir}")

    process_dirs = [
        os.path.join(split_dir, d)
        for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ]

    print(f"🟢 Found {len(process_dirs)} process folders under {split_dir}")

    for proc_dir in process_dirs:
        proc_name = os.path.basename(proc_dir)
        print(f"🔹 Processing {proc_name}...")
        data = load_process_samples(proc_dir, max_files=max_files)
        if data.size == 0:
            print(f"⚠️ No data found for {proc_name}")
            continue
        print(f"  → Loaded {data.shape[0]} events, {data.shape[1]} features")
        plot_process_features(proc_name, data, fmap, output_dir)

    print(f"✅ All processes plotted and saved to {output_dir}")
