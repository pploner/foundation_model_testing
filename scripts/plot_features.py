import argparse
from src.utils.plotting import plot_all_processes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot feature histograms per process from vectorized data.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to the vectorized or preprocessed folder containing feature_map.json and 'train/'.")
    parser.add_argument("--output_dir", type=str, default="plots/raw_features",
                        help="Output directory for saved plots.")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Which split folder to use (default: train).")
    args = parser.parse_args()

    plot_all_processes(args.base_dir, output_dir=args.output_dir, split=args.split)
