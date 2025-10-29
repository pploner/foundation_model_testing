import csv
from glob import glob
import polars as pl
files = glob("/eos/project/f/foundational-model-dataset/samples/production_final/**/*.parquet", recursive=True)
print(len(files))
lazy = (
    pl.scan_parquet(files, extra_columns="ignore")
    .select(pl.col("FullReco_PUPPIPart_PID").explode().unique())
)
print("fertig gelazyed")
unique_pids = (
    lazy.collect(streaming=True)
        .to_series()
        .unique()
        .to_list()
)
with open("unique_pids.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for pid in unique_pids:
        writer.writerow([pid])
