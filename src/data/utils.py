import json
import math
import os
import random
import shutil

from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from .datasets import LocalVectorDataset

# ============================================================
# CONFIG DERIVED UTILITIES
# ============================================================


def get_all_cols(config: dict):
    """Return a flat list of all columns used across all dataset groups."""
    cols = []
    for spec in config.values():
        cols.extend(spec["cols"])
    return cols


def compute_vlen(config: dict) -> int:
    """Compute the total flattened vector length given a dataset config."""
    total = 0
    for spec in config.values():
        ncols = len(spec["cols"])
        topk = spec["topk"]
        if topk is None:
            total += ncols  # scalars
        else:
            total += ncols * topk
        if spec.get("count", False):
            total += 1
    return total


# ============================================================
# FEATURE PACKING HELPERS
# ============================================================


def _pack_topk_batch(pt, *others, k: int, fill: float):
    """Picks top-k objects by first feature (e.g. pt).

    Returns: np.ndarray of shape (n_events, k, n_features)
    """
    arrays = (pt,) + others
    expanded = [arr[:, :, None] for arr in arrays]
    stacked = ak.concatenate(expanded, axis=2)

    # convert and sort
    stacked = ak.values_astype(stacked, np.float32)
    stacked = stacked[ak.argsort(stacked[:, :, 0], axis=1, ascending=False)]

    # take top-k
    topk = stacked[:, :k, :]
    n_features = len(arrays)

    # pad / fill
    topk = ak.pad_none(topk, k, axis=1)
    fill_list = [fill] * n_features
    topk = ak.fill_none(topk, fill_list, axis=1)

    return ak.to_numpy(topk)


def _pack_leading_batch(pt, *others, fill: float):
    """Picks top-1 object (highest pt).

    Returns: np.ndarray of shape (n_events, 1, n_features)
    """
    arrays = (pt,) + others
    expanded = [arr[:, :, None] for arr in arrays]
    stacked = ak.concatenate(expanded, axis=2)

    stacked = ak.values_astype(stacked, np.float32)
    leading = stacked[ak.argmax(stacked[:, :, 0], axis=1, keepdims=True)]

    n_features = len(arrays)
    leading = ak.pad_none(leading, 1, axis=1)
    fill_list = [fill] * n_features
    leading = ak.fill_none(leading, fill_list, axis=1)

    return ak.to_numpy(leading)


# ============================================================
# BATCH VECTOR CONSTRUCTION
# ============================================================


def build_vectors_batch(batch: dict, config: dict, fill: float = 0.0) -> np.ndarray:
    """
    batch: dict mapping column name → awkward.Array
    config: dataset config (e.g. cfg.data.datasets_config)
    fill: fill value for missing entries

    Returns: np.ndarray of shape (n_events, VLEN)
    """
    features = []

    for name, spec in config.items():
        cols = spec["cols"]
        topk = spec["topk"]

        if topk is None:
            # scalars (e.g. MET)
            vals = [ak.to_numpy(ak.fill_none(batch[c], fill)).reshape(-1, 1) for c in cols]
            group = np.concatenate(vals, axis=1)

        elif topk == 1:
            arrays = [batch[c] for c in cols]
            group = _pack_leading_batch(*arrays, fill=fill).reshape(len(arrays[0]), -1)

        else:
            arrays = [batch[c] for c in cols]
            group = _pack_topk_batch(*arrays, k=topk, fill=fill).reshape(len(arrays[0]), -1)

        features.append(group)

        # optional count feature
        if spec.get("count", False):
            nobj = ak.num(batch[cols[0]], axis=1).to_numpy().reshape(-1, 1)
            features.append(nobj)

    # concatenate everything into flat vector
    return np.concatenate(features, axis=1)


# ============================================================
# FEATURE MAP SAVING
# ============================================================


def save_feature_map(config, out_dir: str, vlen: int):
    """Save a feature_map.json describing flattened layout."""
    feature_map = {}
    offset = 0

    config_dict = OmegaConf.to_container(config, resolve=True)

    for group_name, cfg in config_dict.items():
        cols = cfg["cols"]
        topk = cfg["topk"]
        count = cfg.get("count", False)

        if topk is None:
            size = len(cols)
        else:
            size = len(cols) * topk
        if count:
            size += 1

        feature_map[group_name] = {
            "start": int(offset),
            "end": int(offset + size),
            "columns": cols,
            "topk": topk,
            "count": count,
        }
        offset += size

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "feature_map.json"), "w") as f:
        json.dump(feature_map, f, indent=2)

    print(f"✓ feature_map.json saved → {out_dir}")


# ============================================================
# FILE + STORAGE UTILITIES
# ============================================================


def move_to_eos(local_dir: str, eos_dir: str):
    """Move a directory or file from a local tmpdir to EOS target."""
    os.makedirs(os.path.dirname(eos_dir), exist_ok=True)
    try:
        shutil.move(local_dir, eos_dir)
    except Exception as e:
        print(f"⚠️ Move failed: {e}")


# ============================================================
# VECTORIZE AND SAVE LOCALLY
# ============================================================


def vectorize_to_local(
    base_dir: str,
    config: dict,
    class_names: list,
    folder_map: dict,
    labels_map: dict,
    all_cols: list,
    vlen: int,
    tmp_vec_dir: str,
    eos_vec_dir: str,
    split_counts: list,  # [train, val, test]
    read_batch_size: int = 512,  # this is only for how many are read from Parquet at once
):
    """Stream Parquet shards from EOS, vectorize each batch, and save .npy shards under tmp_vec_dir
    (train/val/test split).

    Then move to eos_vec_dir.
    """
    os.makedirs(tmp_vec_dir, exist_ok=True)
    os.makedirs(eos_vec_dir, exist_ok=True)
    save_feature_map(config, eos_vec_dir, vlen)

    split_names = ["train", "val", "test"]
    assert len(split_counts) == 3, "split_counts must have 3 elements"

    for cname in class_names:
        folder = os.path.join(base_dir, folder_map[cname])
        files = sorted(
            f for f in os.listdir(folder) if f.endswith(".parquet") and not f.startswith(".")
        )

        glob_split_idx = 0


        for split_idx, split in enumerate(split_names):
            existing_files = []
            split_folder = os.path.join(eos_vec_dir, split, folder_map[cname])
            if os.path.exists(split_folder):
                existing_files.extend(f for f in os.listdir(split_folder) if f.endswith("_x.npy"))
            if len(existing_files) * 10000 >= split_counts[split_idx]:
                print(f"🟡 Skipping {cname} {split} split (already have enough files)")
                skip = len(existing_files)
                if skip >= len(files):
                    files = []
                else:
                    files = files[skip:]
                glob_split_idx += 1
                continue

        if glob_split_idx >= len(split_names):
            continue

        split_limit = split_counts[glob_split_idx]
        counters = 0

        print(f"🟡 Building {split_names[glob_split_idx]} split for class {cname}...")

        for f in files:
            path = os.path.join(folder, f)
            print(f"→ Processing {path}")

            all_feats, all_labels = [], []
            with pq.ParquetFile(path) as pqf:
                for batch in pqf.iter_batches(columns=all_cols, batch_size=read_batch_size):
                    tbl = pa.Table.from_batches([batch])
                    arrays = {col: ak.from_arrow(tbl[col]) for col in all_cols}
                    feats = build_vectors_batch(arrays, config, fill=0.0)

                    remain = split_limit - counters
                    feats = feats[:remain]
                    n = feats.shape[0]

                    all_feats.append(feats)
                    all_labels.append(np.full((n,), labels_map[cname], dtype=np.int64))
                    counters += n

                    if counters >= split_limit:
                        break

            if not all_feats:
                continue

            feats_cat = np.concatenate(all_feats, axis=0)
            labels_cat = np.concatenate(all_labels, axis=0)



            split = split_names[glob_split_idx]



            tmp_split_dir = os.path.join(tmp_vec_dir, split, folder_map[cname])
            eos_split_dir = os.path.join(eos_vec_dir, split, folder_map[cname])
            os.makedirs(tmp_split_dir, exist_ok=True)
            os.makedirs(eos_split_dir, exist_ok=True)

            base = os.path.splitext(f)[0]
            local_x = os.path.join(tmp_split_dir, f"{base}_x.npy")
            local_y = os.path.join(tmp_split_dir, f"{base}_y.npy")

            np.save(local_x, feats_cat)
            np.save(local_y, labels_cat)

            move_to_eos(local_x, os.path.join(eos_split_dir, f"{base}_x.npy"))
            move_to_eos(local_y, os.path.join(eos_split_dir, f"{base}_y.npy"))

            print(f"✅ Saved {split}/{base}: {feats_cat.shape}")

            # change split if current split is filled
            if counters >= split_limit:
                counters = 0
                glob_split_idx += 1
                if glob_split_idx >= len(split_names):
                    break
                split_limit = split_counts[glob_split_idx]
                print(f"🟡 Building {split_names[glob_split_idx]} split for class {cname}...")

    print(f"✅ Finished vectorizing → {eos_vec_dir}")


# ============================================================
# RESEEDING EVERY EPOCH
# ============================================================
def worker_init_fn(worker_id):
    # seed everything based on global seed + epoch + worker id
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the actual dataset copy
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# CHECK IF ENOUGH FILES ARE PRESENT IN A TARGET DIRECTORY
# ============================================================
def has_enough_events(target: str, train_val_test_split_per_class, classnames, folder) -> bool:
    if not target or not os.path.exists(target):
        return False
    split_names = ["train", "val", "test"]
    # assume each _x.npy shard contains up to 10_000 events
    events_per_file = 10000
    for split, needed_events in zip(split_names, train_val_test_split_per_class):
        needed_files_per_class = math.ceil(needed_events / events_per_file)
        for cname in classnames:
            split_folder = os.path.join(target, split, folder[cname])
            if not os.path.exists(split_folder):
                return False
            existing = [f for f in os.listdir(split_folder) if f.endswith("_x.npy")]
            if len(existing) < needed_files_per_class:
                return False
    return True
