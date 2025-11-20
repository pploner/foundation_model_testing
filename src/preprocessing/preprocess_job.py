import argparse
import json
import hydra
import sys
import os
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocessing.preprocess import PreprocessingPipeline

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("🟢 preprocess_job.py starting...", flush=True)

# ---------------------------------------------------------------------
# Parse manifest path before Hydra gets control
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--manifest-path", type=str, required=True)
args, _ = parser.parse_known_args()

# Make manifest path absolute now (before Hydra may change cwd)
args.manifest_path = os.path.abspath(args.manifest_path)

# ⚠️ Prevent Hydra from seeing our CLI arguments
sys.argv = [sys.argv[0]]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # Load the subset manifest for this job
    with open(args.manifest_path, "r") as f:
        subset_manifest = json.load(f)

    data_cfg = cfg.data
    preprocess_cfg = cfg.preprocess

    # Build paths dict in expected format
    paths = {
        "eos_vec_dir": data_cfg.paths.eos_vec_dir,
        "tmp_preproc_dir": data_cfg.paths.tmp_preproc_dir,
        "eos_preproc_dir": data_cfg.paths.eos_preproc_dir,
    }

    # Build class -> folder mapping
    to_classify = data_cfg.to_classify               # {"QCD": "QCD inclusive", ...}
    proc_to_folder = data_cfg.process_to_folder      # {"QCD inclusive": "QCD_HT50toInf", ...}
    class_order = list(to_classify.keys())
    process_to_folder = {
        class_name: proc_to_folder[proc_name]
        for class_name, proc_name in to_classify.items()
    }

    # Instantiate preprocessing pipeline
    pipeline = PreprocessingPipeline(
        paths=paths,
        preprocess_cfg=preprocess_cfg,
        process_to_folder=process_to_folder,
        class_order=class_order,
        device="cpu",
    )

    # Load already-fitted normalization stats (you created them with mode=fit_only)
    stats = pipeline._load_stats_json()
    print(f"🟢 Loaded stats from {pipeline.stats_out}")

    # Apply only to this subset
    pipeline.apply_manifest(subset_manifest, stats)
    print("✅ preprocess_job.py completed subset.")


if __name__ == "__main__":
    main()
