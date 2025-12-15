import argparse
import json
import hydra
import sys
import os
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.utils import vectorize_to_local, get_all_cols

"""
Vectorize job script for vectorizing data based on a given manifest file.
"""

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("🟢 vectorize_job.py starting...", flush=True)

# Parse manifest path before Hydra gets control
parser = argparse.ArgumentParser()
parser.add_argument("--manifest-path", type=str, required=True)
args, _ = parser.parse_known_args()

# Make manifest path absolute now (before Hydra may change cwd)
args.manifest_path = os.path.abspath(args.manifest_path)

# Prevent Hydra from seeing the CLI arguments
sys.argv = [sys.argv[0]]

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    with open(args.manifest_path, "r") as f:
        split_manifest = json.load(f)

    data_cfg = cfg.data
    to_classify = data_cfg.to_classify
    process_to_folder = data_cfg.process_to_folder

    # remap class name -> folder name
    folder_map = {
        class_name: process_to_folder[process_name]
        for class_name, process_name in to_classify.items()
    }

    vectorize_to_local(
        base_dir=data_cfg.paths.dataset_dir,
        config=data_cfg.datasets_config,
        class_names=list(data_cfg.to_classify.keys()),
        folder_map=folder_map,
        labels_map={k: i for i, k in enumerate(data_cfg.to_classify.keys())},
        all_cols=get_all_cols(data_cfg.datasets_config),
        vlen=getattr(data_cfg, "vlen", 0),
        tmp_vec_dir=data_cfg.paths.tmp_vec_dir,
        eos_vec_dir=data_cfg.paths.eos_vec_dir,
        split_counts=data_cfg.train_val_test_split_per_class,
        split_manifest=split_manifest,
    )

if __name__ == "__main__":
    main()
