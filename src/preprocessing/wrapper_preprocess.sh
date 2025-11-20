#!/bin/bash
# =====================================================
#  HTCondor wrapper for preprocessing jobs
# =====================================================

set -euo pipefail

MANIFEST_PATH="$1"

PROJECT_DIR=/afs/cern.ch/work/p/phploner/foundation_model_testing
IMAGE=${PROJECT_DIR}/fm_testing.sif

echo "[`date`] Starting preprocessing job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Manifest path: ${MANIFEST_PATH}"

cd ${PROJECT_DIR}

apptainer exec --cleanenv \
    --bind /eos:/eos \
    --writable-tmpfs \
    ${IMAGE} bash -lc "python src/preprocessing/preprocess_job.py --manifest-path ${MANIFEST_PATH}"

EXIT_CODE=$?
echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
