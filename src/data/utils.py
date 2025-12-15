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

def load_global_filelist() -> dict:
    """Load the precomputed file event counts JSON from nEvents_scan."""
    base_path = Path("/afs/.cern.ch/work/p/phploner/foundation_model_testing/src/utils/nEvents_scan/file_event_counts.json")
    if not base_path.exists():
        raise FileNotFoundError(f"Global file list not found: {base_path}")
    with open(base_path) as f:
        data = json.load(f)
    print(f"🟢 Loaded global file list ({sum(len(v) for v in data.values())} files) from {base_path}")
    return data

def make_split_manifest(global_filelist, split_counts, include_folders, seed=42):
    """
    Absolute target rows per class: greedily assign whole files until
    each split reaches its target in `split_counts` (within one file).
    Extra files are ignored once test target is met.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    split_names = ["train", "val", "test"]
    targets_abs = np.array(split_counts, dtype=int)  # e.g., [50000, 20000, 20000]

    manifest = {}
    for folder in include_folders:
        items = [(fn, int(n)) for fn, n in global_filelist.get(folder, {}).items()]
        if not items:
            manifest[folder] = {s: [] for s in split_names}
            continue

        rng.shuffle(items)

        buckets = {s: [] for s in split_names}
        split_idx = 0
        acc = 0

        for fname, n in items:
            # if all splits done, stop
            if split_idx >= len(split_names):
                break

            buckets[split_names[split_idx]].append(fname)
            acc += n

            # move to next split once target reached (allow overshoot by one file)
            if acc >= targets_abs[split_idx]:
                split_idx += 1
                acc = 0

        # ensure remaining splits exist as empty lists
        for s in split_names:
            buckets.setdefault(s, [])

        manifest[folder] = buckets

    return manifest

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
    read_batch_size: int = 512,
    split_manifest: dict | None = None,
):
    """Vectorize Parquet shards using a deterministic split manifest.

    If `split_manifest_path` is given, it defines which files belong
    to train/val/test. Otherwise, the manifest is created or reused
    in `eos_vec_dir/split_manifest.json`.

    Each file in the manifest is converted into .npy shards under:
        eos_vec_dir/{train,val,test}/{class_folder}/...
    """

    os.makedirs(tmp_vec_dir, exist_ok=True)
    os.makedirs(eos_vec_dir, exist_ok=True)
    save_feature_map(config, eos_vec_dir, vlen)

    # -------------------------------------------------------------------------
    # Load or create manifest
    # -------------------------------------------------------------------------
    if split_manifest is not None:
        print("🟡 Using in-memory split manifest (dict provided).")
    else:
        manifest_path = os.path.join(eos_vec_dir, "split_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                split_manifest = json.load(f)
            print(f"🟡 Using existing split manifest: {manifest_path}")
        else:
            print("🟢 Building new split manifest from global file list ...")
            global_filelist = load_global_filelist()
            include_folders = [folder_map[cname] for cname in class_names if cname in folder_map]
            print(f"🧩 Including {len(include_folders)} folders in manifest:")
            for f in include_folders:
                print(f"   • {f}")

            split_manifest = make_split_manifest(
                global_filelist=global_filelist,
                split_counts=split_counts,
                include_folders=include_folders,
                seed=42,
            )
            with open(manifest_path, "w") as f:
                json.dump(split_manifest, f, indent=2)
            print(f"✅ Wrote split manifest → {manifest_path}")

    # -------------------------------------------------------------------------
    # Vectorize files based on manifest
    # -------------------------------------------------------------------------
    split_names = ["train", "val", "test"]

    for cname in class_names:
        class_folder = folder_map[cname]
        label_id = labels_map[cname]
        if class_folder not in split_manifest:
            print(f"⚪ Skipping {cname}: not in manifest.")
            continue

        for split_name in split_names:
            file_list = split_manifest[class_folder].get(split_name, [])
            if not file_list:
                continue

            print(f"🟡 Processing {cname}/{split_name} ({len(file_list)} files)")

            for fname in file_list:
                folder = os.path.join(base_dir, class_folder)
                path = os.path.join(folder, fname)
                base = os.path.splitext(fname)[0]

                tmp_split_dir = os.path.join(tmp_vec_dir, split_name, class_folder)
                eos_split_dir = os.path.join(eos_vec_dir, split_name, class_folder)
                os.makedirs(tmp_split_dir, exist_ok=True)
                os.makedirs(eos_split_dir, exist_ok=True)

                dst_x = os.path.join(eos_split_dir, f"{base}_x.npy")
                dst_y = os.path.join(eos_split_dir, f"{base}_y.npy")

                # Idempotent skip if already vectorized
                if os.path.exists(dst_x) and os.path.exists(dst_y):
                    continue

                print(f"→ Processing {path}")
                all_feats, all_labels = [], []

                try:
                    with pq.ParquetFile(path) as pqf:
                        for batch in pqf.iter_batches(columns=all_cols, batch_size=read_batch_size):
                            tbl = pa.Table.from_batches([batch])
                            arrays = {col: ak.from_arrow(tbl[col]) for col in all_cols}
                            feats = build_vectors_batch(arrays, config, fill=0.0)
                            n = feats.shape[0]
                            all_feats.append(feats)
                            all_labels.append(np.full((n,), label_id, dtype=np.int64))
                except Exception as e:
                    print(f"❌ Error reading {path}: {e}")
                    continue

                if not all_feats:
                    continue

                feats_cat = np.concatenate(all_feats, axis=0)
                labels_cat = np.concatenate(all_labels, axis=0)

                local_x = os.path.join(tmp_split_dir, f"{base}_x.npy")
                local_y = os.path.join(tmp_split_dir, f"{base}_y.npy")
                np.save(local_x, feats_cat)
                np.save(local_y, labels_cat)

                move_to_eos(local_x, dst_x)
                move_to_eos(local_y, dst_y)

                print(f"✅ Saved {split_name}/{base}: {feats_cat.shape}")

    print(f"✅ Finished vectorizing → {eos_vec_dir}")


# ============================================================
# RESEEDING EVERY EPOCH
# ============================================================
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return  # not running inside a DataLoader worker
    # seed everything based on torch's worker-specific initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# CHECK IF ENOUGH FILES ARE PRESENT IN A TARGET DIRECTORY
# ============================================================
import os
import json
import math

def has_enough_events(
    target: str,
    train_val_test_split_per_class,
    classnames,
    folder_map,
    event_count_json_path="src/utils/nEvents_scan/file_event_counts.json",
) -> bool:
    """
    Returns True only if all splits/classes meet required total events

    Args:
        target: Directory that contains train/val/test/{folder}/
        train_val_test_split_per_class: e.g. [50_000, 20_000, 20_000]
        classnames: list like ["QCD", "ggHbb"]
        folder_map: mapping class_name -> folder name used in vectorized dir
        event_count_json_path: path to JSON with event counts per parquet file
    """

    if not target or not os.path.exists(target):
        return False

    # Load event count database
    if not os.path.exists(event_count_json_path):
        raise FileNotFoundError(f"Missing event count file: {event_count_json_path}")

    with open(event_count_json_path) as f:
        event_db = json.load(f)

    split_names = ["train", "val", "test"]

    for split, needed_events in zip(split_names, train_val_test_split_per_class):
        for cname in classnames:
            folder = folder_map[cname]         # e.g. "QCD_HT50toInf"
            split_dir = os.path.join(target, split, folder)

            if not os.path.isdir(split_dir):
                return False

            # list files: *_x.npy
            files = [f for f in os.listdir(split_dir) if f.endswith("_x.npy")]
            if not files:
                return False

            # sum events using event_db
            total_events = 0
            for npy_name in files:
                # convert e.g.
                #   QCD_HT50toInf-NEVENT10000-RS26000001_x.npy
                # → QCD_HT50toInf-NEVENT10000-RS26000001.parquet
                parquet_name = npy_name.replace("_x.npy", ".parquet")

                # event_db entry: event_db[folder][parquet_name]
                if folder not in event_db:
                    raise KeyError(f"Folder '{folder}' missing in event count DB")

                if parquet_name not in event_db[folder]:
                    raise KeyError(
                        f"File '{parquet_name}' missing in event count DB for folder '{folder}'"
                    )

                total_events += event_db[folder][parquet_name]

            # check whether this class meets its required events for this split
            if total_events < needed_events:
                # Not enough events for this class in this split
                return False

    # If all classes in all splits have enough events
    return True
