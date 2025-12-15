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

## Determine accelerator from full Hydra composition
ACCELERATOR=$(python - << 'EOF'
from hydra import initialize, compose

with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(config_name="train.yaml")

print(cfg.trainer.get("accelerator", "cpu"))
EOF
)

# Pick Apptainer flags
APPTAINER_FLAGS="--cleanenv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs"

if [ "$ACCELERATOR" = "gpu" ]; then
    echo "[wrapper] GPU requested → enabling --nv"
    APPTAINER_FLAGS="--nv $APPTAINER_FLAGS"
else
    echo "[wrapper] CPU mode → running without --nv"
fi

# --- Run training inside container ---
echo "[wrapper] Running training with flags: $APPTAINER_FLAGS"

apptainer exec $APPTAINER_FLAGS \
    "${IMAGE}" bash -lc "python src/train.py"

EXIT_CODE=$?

echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
