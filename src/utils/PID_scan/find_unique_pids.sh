#!/bin/bash

echo "=== Starting job on $(hostname) ==="

# Load Python from CVMFS
source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos9-gcc11-opt/setup.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load CVMFS environment"
    exit 1
fi

pip install --no-cache-dir --target=./site-packages pyarrow==20.0.0

export PYTHONPATH=$(pwd)/site-packages:$PYTHONPATH

python3 find_all_PIDs.py

echo "=== Job finished ==="
