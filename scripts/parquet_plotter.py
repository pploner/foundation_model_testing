"""
Given an input directory containing subfolders with parquet files,
this script plots the features grouped by <view>_<group> and saves
the histograms as PNG files in the specified output directory.

Can be used to do validation checks on the COLLIDE-2V dataset.

Example usage:
    python scripts/parquet_plotter.py \
        --input_dir /path/to/input_dir \
        --output_dir /path/to/output_dir
"""


#!/usr/bin/env python3
import argparse
from pathlib import Path
import awkward as ak
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


def parse_column(colname):
    """
    Parse column names of the form <view>_<group>_<feature>.
    Returns (view_group, feature) or (None, None) if not matching.
    """
    parts = colname.split("_", 2)
    if len(parts) != 3:
        return None, None
    view, group, feature = parts
    return f"{view}_{group}", feature


def process_parquet_file(parquet_path: Path, out_dir: Path):
    """
    - Read parquet into Awkward
    - For each <view_group>, collect all features
    - For each feature:
        - Extract leading element arr[:, 0]
        - Plot histogram
    - Save one PNG per <view_group>
    """
    print(f"Loading {parquet_path} as Awkward...")
    arr = ak.from_parquet(parquet_path)

    grouped = {}

    for col in arr.fields:
        view_group, feature = parse_column(col)
        if view_group is None:
            continue
        grouped.setdefault(view_group, []).append((col, feature))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plotting
    for view_group, columns in grouped.items():
        print(f"Plotting {view_group} with {len(columns)} features")
        n_features = len(columns)

        fig, axes = plt.subplots(
            n_features, 1, figsize=(8, 3 * n_features), squeeze=False
        )

        for ax, (col, feature) in zip(axes.flat, columns):
            print(f"  Extracting leading element for feature {feature} from column {col}")

            if feature == "Constituents":
                print("    Skipping 'Constituents' feature.")
                continue

            # Awkward extraction
            arrow_col = arr[col]

            try:
                leading = ak.firsts(arrow_col)
            except Exception:
                leading = arrow_col

            # Convert to plain list, filter None
            data = [x for x in ak.to_list(leading) if x is not None]

            ax.hist(data, bins=50)
            ax.set_title(feature)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

        fig.tight_layout()
        outfile = out_dir / f"{view_group}.png"
        print(f"Saving {outfile}")
        plt.savefig(outfile, dpi=150)
        plt.close(fig)


def main(input_dir: str, output_dir: str):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for subfolder in sorted(input_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        parquet_files = list(subfolder.glob("*.parquet"))
        if not parquet_files:
            print(f"No parquet files in {subfolder}, skipping.")
            continue

        parquet_file = parquet_files[0]
        print(f"Processing {parquet_file}")

        out_subfolder = output_dir / subfolder.name
        process_parquet_file(parquet_file, out_subfolder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot histograms from parquet files grouped by <view>_<group>."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
