import json
import os
import time
import glob
import pyarrow.compute as pc
import pyarrow.dataset as ds

# =====================
# CONFIGURATION
# =====================
EOS_PATH = "/eos/project/f/foundational-model-dataset/samples/production_final/**/*.parquet"
OUTPUT_PATH = "unique_pids.json"
BATCH_SIZE = 50

# =====================

print(f"[INFO] Listing files in {EOS_PATH} ... this may take a moment")
start = time.time()
files = sorted(glob.glob(EOS_PATH))
elapsed = time.time() - start

print(f"[INFO] Found {len(files)} files in {elapsed:.1f} seconds")

if not files:
    raise RuntimeError("No files found! Check your EOS path.")

unique_pids = set()
num_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"[INFO] Processing in {num_batches} batches of size {BATCH_SIZE}")

for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(len(files), (batch_idx + 1) * BATCH_SIZE)
    batch_files = files[start_idx:end_idx]

    print(f"[BATCH {batch_idx+1}/{num_batches}] Files {start_idx} to {end_idx-1}")

    t0 = time.time()
    dataset = ds.dataset(batch_files, format="parquet")
    # request only the column we need; use_threads=True lets Arrow parallelize IO/decoding
    table = dataset.scanner(columns=["FullReco_PUPPIPart_PID"], use_threads=True).to_table()


    # vectorized flatten + unique via pyarrow.compute (works on ChunkedArray / ListArray)
    pid_col = table.column(0)                         # list<...> column (ChunkedArray)
    flat = pc.list_flatten(pid_col)                   # primitive array of all pids in this batch
    unique_in_batch = pc.unique(flat)                 # unique pids (pyarrow.Array)

    # add to python set
    unique_pids.update(unique_in_batch.to_pylist())

    t1 = time.time()
    print(f"[BATCH {batch_idx+1}] PyArrow Dataset -> Arrow finished in {t1 - t0:.1f} seconds")

    print(f"[BATCH {batch_idx+1}] Unique PIDs so far: {len(unique_pids)}")

# Save final output
unique_pids = sorted(unique_pids)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(unique_pids, f, indent=2)

print(f"✅ Done! Found {len(unique_pids)} unique PIDs.")
print(f"✅ Saved to {OUTPUT_PATH}")
