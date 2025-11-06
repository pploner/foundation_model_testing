#!/bin/bash
# =====================================================
#  HTCondor wrapper for foundation_model_testing pipeline
# =====================================================

set -euo pipefail

echo "[`date`] Starting Condor job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Working directory: $(pwd)"

# --- Paths ---
PROJECT_DIR=/afs/cern.ch/work/p/phploner/foundation_model_testing
IMAGE=${PROJECT_DIR}/fm_testing.sif

# --- Go to project directory ---
cd ${PROJECT_DIR}

# --- Sanity check: Python version ---
apptainer exec ${IMAGE} python -V || true

# --- Run training inside container ---
apptainer exec --cleanenv \
    --bind /afs:/afs \
    --bind /eos:/eos \
    --writable-tmpfs \
    ${IMAGE} bash -lc "python src/train.py"

EXIT_CODE=$?

echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
