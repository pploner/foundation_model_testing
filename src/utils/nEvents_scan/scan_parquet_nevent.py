#!/usr/bin/env python3
"""
Scan all parquet files under each process folder defined in your Hydra config
and record their number of events into a global JSON metadata file.

Outputs:
    file_event_counts.json:
        {
            "QCD_HT50toInf": {"0001.parquet": 10000, "0002.parquet": 9850, ...},
            "ttH_incl": {"0001.parquet": 9725, ...},
            ...
        }
"""

import os
import json
from omegaconf import OmegaConf
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def count_events(path: str) -> tuple[str, int | str]:
    """Return the number of events in a Parquet file."""
    try:
        pf = pq.ParquetFile(path)
        # Try standard num_rows first
        nrows = pf.metadata.num_rows
        if nrows is None:
            # Try custom metadata fallback (nEvents from your writer)
            meta = pf.metadata.metadata or {}
            if b"nEvents" in meta:
                nrows = int(meta[b"nEvents"])
            else:
                # Sum row groups if both missing (rare)
                nrows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        return os.path.basename(path), nrows
    except Exception as e:
        return os.path.basename(path), f"error: {e}"


def scan_dataset(base_dir: str, process_to_folder: dict, output_json: str, n_workers: int = 8):
    """Iterate over all folders and scan parquet files in parallel."""
    summary = {}
    for process_name, folder_name in process_to_folder.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️  Missing folder: {folder_path}, skipping")
            continue

        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if f.endswith(".parquet")]
        print(f"🔍 Scanning {folder_name} ({len(files)} files)")

        class_meta = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for fname, nrows in tqdm(ex.map(count_events, files), total=len(files)):
                if isinstance(nrows, int):
                    class_meta[fname] = nrows
                else:
                    print(f"  ⚠️  {folder_name}/{fname}: {nrows}")

        summary[folder_name] = class_meta

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved {output_json} ({len(summary)} datasets)")


if __name__ == "__main__":
    # --- Load your Hydra data config ---
    cfg = OmegaConf.load("configs/data/collide2v.yaml")

    # Base directory of all Parquet folders
    base_dir = cfg.get("base_dir", "/eos/project/f/foundational-model-dataset/samples/production_final")

    # process_to_folder mapping
    process_to_folder = cfg["process_to_folder"]

    # Output file (you can change to EOS path if preferred)
    output_json = "file_event_counts.json"

    scan_dataset(base_dir, process_to_folder, output_json, n_workers=16)
