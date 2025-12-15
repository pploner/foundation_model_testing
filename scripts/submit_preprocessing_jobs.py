#!/usr/bin/env python3
"""
submit_preprocessing_jobs.py

Use this script to batch submit preprocessing jobs to Condor.

Reads Hydra configs, creates stats file, splits it into chunks of ≤ MAX_FILES_PER_JOB,
and directly submits each chunk to Condor to run `PreprocessingPipeline` inside the Apptainer container.
"""

import json
import subprocess
from math import ceil
from pathlib import Path
from omegaconf import OmegaConf

from src.preprocessing.preprocess import PreprocessingPipeline

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MAX_FILES_PER_JOB = 20
JOB_FLAVOUR = "tomorrow"  # ~24h jobs


PROJECT_DIR = Path("/afs/cern.ch/work/p/phploner/foundation_model_testing")
WRAPPER_SCRIPT = PROJECT_DIR / "src/preprocessing/wrapper_preprocess.sh"
LOG_DIR = PROJECT_DIR / "logs/condor_logs/preprocessing"

# -----------------------------------------------------------------------------
# LOAD CONFIGS
# -----------------------------------------------------------------------------
print("🟢 Loading Hydra configs...")

train_cfg = OmegaConf.load(PROJECT_DIR / "configs/train.yaml")

paths_cfg = train_cfg.paths
data_cfg = train_cfg.data
pre_cfg = train_cfg.preprocess
to_classify = data_cfg.to_classify              # {"QCD": "QCD inclusive", ...}
folder_map_all = data_cfg.process_to_folder     # {"QCD inclusive": "QCD_HT50toInf", ...}

class_order = list(to_classify.keys())

process_to_folder = {
    class_name: folder_map_all[proc_name]
    for class_name, proc_name in to_classify.items()
}

LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Build PreprocessingPipeline (fit_only)
# -----------------------------------------------------------------------------
print("🟡 Preparing preprocessing pipeline (fit_only)...")

paths = {
    "eos_vec_dir": data_cfg.paths.eos_vec_dir,
    "tmp_preproc_dir": data_cfg.paths.tmp_preproc_dir,
    "eos_preproc_dir": data_cfg.paths.eos_preproc_dir,
}

# Force mode = fit_only without touching config files
pre_cfg_local = OmegaConf.to_container(pre_cfg, resolve=True)
pre_cfg_local["mode"] = "fit_only"
pre_cfg_local["enabled"] = True
pre_cfg_local = OmegaConf.create(pre_cfg_local)

pipeline = PreprocessingPipeline(
    paths=paths,
    preprocess_cfg=pre_cfg_local,
    process_to_folder=process_to_folder,
    class_order=class_order,
    device="cpu",
)

pipeline.run()

stats_path = Path(paths["eos_preproc_dir"]) / "norm_stats.json"
if not stats_path.exists():
    raise RuntimeError(f"❌ norm_stats.json missing after fit_only: {stats_path}")

print(f"✅ Fitted normalization stats → {stats_path}")

# -----------------------------------------------------------------------------
# SCAN VECTORIZED FILES
# -----------------------------------------------------------------------------
print("\n🟢 Scanning vectorized files...")

entries = []  # list of (folder, split, filename)

for split in ("train", "val", "test"):
    for cls in class_order:
        cls_folder = process_to_folder[cls]
        vec_dir = Path(paths["eos_vec_dir"]) / split / cls_folder

        npy_files = list(vec_dir.glob("*_x.npy"))
        for f in npy_files:
            entries.append((cls_folder, split, f.name))

num_files = len(entries)
n_jobs = ceil(num_files / MAX_FILES_PER_JOB)

print(f"🟡 Found {num_files} vectorized files → {n_jobs} jobs (≤{MAX_FILES_PER_JOB}/job)")



# -----------------------------------------------------------------------------
# FUNCTION TO SUBMIT A SINGLE JOB
# -----------------------------------------------------------------------------
def submit_job(job_idx, chunk):
    # Build submanifest structure
    submanifest = {}
    for folder, split, fname in chunk:
        if folder not in submanifest:
            submanifest[folder] = {"train": [], "val": [], "test": []}
        submanifest[folder][split].append(fname)

    # Save submanifest
    manifest_path = LOG_DIR / f"manifest_{job_idx:04d}.json"
    with open(manifest_path, "w") as f:
        json.dump(submanifest, f, indent=2)

    log_out = LOG_DIR / f"job_{job_idx:04d}.out"
    log_err = LOG_DIR / f"job_{job_idx:04d}.err"
    log_log = LOG_DIR / f"job_{job_idx:04d}.log"

    submit_content = f"""\
executable = {WRAPPER_SCRIPT}
arguments  = {manifest_path}
initialdir = {LOG_DIR}

output = {log_out}
error  = {log_err}
log    = {log_log}

stream_output = True
stream_error = True

run_as_owner = True
+JobFlavour = "{JOB_FLAVOUR}"
getenv = True
request_cpus = 4
environment = "WRAPPER_DEBUG=1; OMP_NUM_THREADS=4; MKL_NUM_THREADS=4"

queue
"""

    sub_path = LOG_DIR / f"preprocess_job_{job_idx:04d}.sub"
    with open(sub_path, "w") as f:
        f.write(submit_content)

    subprocess.run(["condor_submit", str(sub_path)], check=False)
    print(f"🚀 Submitted job {job_idx:04d} ({len(chunk)} files)")


# -----------------------------------------------------------------------------
# SUBMIT ALL JOBS
# -----------------------------------------------------------------------------
for i in range(n_jobs):
    start = i * MAX_FILES_PER_JOB
    end = start + MAX_FILES_PER_JOB
    chunk = entries[start:end]
    submit_job(i, chunk)

print("\n✅ All preprocessing jobs submitted.")
