import argparse
import os
import sys

# ensure project root is on PYTHONPATH so "src" package is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.plotting import plot_all_processes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot feature histograms per process from vectorized data."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the vectorized or preprocessed folder containing feature_map.json and 'train/'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Output directory for saved plots, will automatically create subdirectories for label and vectorized vs preprocessed.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split folder to use (default: train).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=5,
        help="Maximum number of files to process (default: 5).",
    )
    args = parser.parse_args()

    plot_all_processes(args.base_dir, output_dir=args.output_dir, split=args.split, max_files=args.max_files)
