# src/preprocessing/preprocess.py

import os, json, glob, shutil
from typing import Dict, List
import numpy as np
import torch
from omegaconf import OmegaConf

from .utils import (
    load_feature_map,
    build_expanded_feature_map,
    expand_feature_columns,
    infer_group,
    save_feature_map,
)
from .transforms import apply_transform
from .normalization import create_normalizer


class PreprocessingPipeline:
    """
    Offline preprocessing pipeline:
      - Fit normalization on a subset of training files (N per class)
      - Apply transforms + normalization to all files (train/val/test)
      - Write to AFS, then move each file to EOS
      - Emit expanded feature_map.json and norm_stats.json on EOS
    """

    def __init__(
        self,
        paths: Dict[str, str],
        preprocess_cfg: Dict,
        process_to_folder: Dict[str, str],
        class_order: List[str],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            paths: expects keys:
                - eos_vec_dir
                - afs_preproc_dir
                - eos_preproc_dir
            preprocess_cfg: expects keys:
                - enabled (bool)
                - mode: "fit_and_apply" or "apply_only"
                - fit_num_files_per_class: int
                - feature_transforms: {group: transform_name}
                - feature_normalizations: {group: norm_name}
            process_to_folder: {"QCD": "QCD", "ggHbb": "ggHbb", ...}
            class_order: ["QCD", "ggHbb", ...]
        """
        self.paths = paths
        self.cfg = preprocess_cfg
        self.proc2fold = process_to_folder
        self.class_order = class_order
        self.device = torch.device(device)
        self.dtype = dtype

        # Hardcoded IO artifacts (derived from paths)
        self.fm_in = os.path.join(self.paths["eos_vec_dir"], "feature_map.json")
        self.fm_out = os.path.join(self.paths["eos_preproc_dir"], "feature_map.json")
        self.stats_out = os.path.join(self.paths["eos_preproc_dir"], "norm_stats.json")

        # Load original feature_map and prepare expanded one
        self.original_fm = load_feature_map(self.fm_in)
        self.feature_transforms = self.cfg["feature_transforms"]
        self.feature_normalizations = self.cfg["feature_normalizations"]

        self.expanded_fm = build_expanded_feature_map(
            self.original_fm, self.feature_transforms
        )
        self.expanded_columns_concat = self._concat_all_columns(self.expanded_fm)
        self.output_dim = self._infer_output_dim(self.expanded_fm)

        # Ensure output roots exist
        os.makedirs(self.paths["afs_preproc_dir"], exist_ok=True)
        os.makedirs(self.paths["eos_preproc_dir"], exist_ok=True)

    # ---------- public entry ----------

    def run(self):
        if not self.cfg.get("enabled", True):
            print("🟡 Preprocessing disabled. Performing safety checks on EOS preprocessed directory...")
            self._safety_check_existing_preprocessed()
            return

        mode = self.cfg.get("mode", "fit_and_apply")
        if mode not in ("fit_and_apply", "apply_only"):
            raise ValueError("preprocess.mode must be 'fit_and_apply' or 'apply_only'.")

        if os.path.exists(self.stats_out):
            print(f"🟢 Preprocessed data already found with {self.stats_out} — skipping full preprocessing.")
            return

        # Fit stats if requested
        if mode == "fit_and_apply":
            stats = self.fit_normalization()
            self._save_stats_json(stats)
        else:
            stats = self._load_stats_json()

        # Apply to all splits/classes/files
        self.apply_to_all_files(stats)

        # Save expanded feature_map to EOS (after success)
        save_feature_map(self.expanded_fm, self.fm_out)
        print(f"✅ Wrote expanded feature map to: {self.fm_out}")

    # ---------- core steps ----------

    def fit_normalization(self) -> Dict:
        """
        Fit column-wise normalization AFTER transforms on a subset of train files.
        Returns a dict containing both array_stats and human-readable column_stats.
        """
        print("🟡 Fitting normalization stats from training subset...")
        num_files = self.cfg.get("fit_num_files_per_class", 5)

        transformed_batches = []
        for cls in self.class_order:
            cls_folder = self.proc2fold[cls]
            pattern = os.path.join(self.paths["eos_vec_dir"], "train", cls_folder, "*_x.npy")
            files = sorted(glob.glob(pattern))[:num_files]
            if not files:
                print(f"⚠️ No files found for class '{cls}' at {pattern}")
                continue
            print(f"  • Class {cls}: using {len(files)} files for stats")
            for fpath in files:
                X = self._load_raw_npy(fpath)           # [N, D_raw]
                X_t = self._transform_only(X)           # [N, D_expanded]
                transformed_batches.append(X_t)

        if not transformed_batches:
            raise RuntimeError("No files collected for normalization fitting. Check paths/class mapping.")

        X_all = torch.cat(transformed_batches, dim=0)  # concatenate along events
        print(f"🟢 Fit sample shape after transform: {tuple(X_all.shape)} (events, features)")

        stats = self._compute_stats_both_forms(X_all)
        print("✅ Fitted normalization statistics.")
        return stats

    def apply_to_all_files(self, stats: Dict):
        """
        Apply transforms + normalization to every file in (train|val|test)/(each class),
        write to AFS then move to EOS, preserving structure.
        """
        for split in ("train", "val", "test"):
            for cls in self.class_order:
                cls_folder = self.proc2fold[cls]
                in_dir = os.path.join(self.paths["eos_vec_dir"], split, cls_folder)
                out_afs_dir = os.path.join(self.paths["afs_preproc_dir"], split, cls_folder)
                out_eos_dir = os.path.join(self.paths["eos_preproc_dir"], split, cls_folder)
                os.makedirs(out_afs_dir, exist_ok=True)
                os.makedirs(out_eos_dir, exist_ok=True)

                files = sorted(glob.glob(os.path.join(in_dir, "*_x.npy")))
                if not files:
                    print(f"⚠️ No input files at {in_dir}")
                    continue

                print(f"🟡 Processing {split}/{cls} ({len(files)} files)")
                for fpath in files:
                    fname = os.path.basename(fpath)
                    tmp_out = os.path.join(out_afs_dir, fname)
                    final_out = os.path.join(out_eos_dir, fname)

                    # Skip if already present at EOS
                    if os.path.exists(final_out):
                        continue

                    X = self._load_raw_npy(fpath)
                    X_t = self._transform_only(X)
                    X_n = self._apply_normalization(X_t, stats)

                    np.save(tmp_out, X_n.cpu().numpy())
                    shutil.move(tmp_out, final_out)

                    label_in = fpath.replace("_x.npy", "_y.npy")
                    if os.path.exists(label_in):
                        label_final_out = final_out.replace("_x.npy", "_y.npy")
                        shutil.copy(label_in, label_final_out)



                print(f"✅ Done split={split} class={cls}")

        # Ensure stats present at EOS root
        if not os.path.exists(self.stats_out):
            self._save_stats_json(stats)

    # ---------- helpers: IO & safety ----------

    def _safety_check_existing_preprocessed(self):
        if not os.path.exists(self.fm_out):
            raise FileNotFoundError(
                f"Preprocessing disabled, but missing feature map at {self.fm_out}"
            )
        for split in ("train", "val", "test"):
            for cls in self.class_order:
                cls_folder = self.proc2fold[cls]
                d = os.path.join(self.paths["eos_preproc_dir"], split, cls_folder)
                if not os.path.isdir(d):
                    raise FileNotFoundError(
                        f"Preprocessed folder missing: {d}. Re-run with preprocess.enabled=true."
                    )
        print("✅ Safety check passed: EOS preprocessed data present.")

    def _load_raw_npy(self, path: str) -> torch.Tensor:
        arr = np.load(path)
        return torch.from_numpy(arr).to(self.device, self.dtype)

    # ---------- helpers: transforms ----------

    def _transform_only(self, X_raw: torch.Tensor) -> torch.Tensor:
        """
        Apply per-column transforms using registry-based apply_transform().
        Returns [N, D_expanded] in the correct expanded order.
        """
        cols_out: List[torch.Tensor] = []

        for _, section in self.original_fm.items():
            raw_cols = section["columns"]
            topk_raw = section.get("topk", None)
            topk_internal = topk_raw if topk_raw is not None else 1
            has_count = section.get("count", False)
            num_fields_raw = len(raw_cols)

            # Iterate over repeated objects
            for obj_idx in range(topk_internal):
                base = section["start"] + obj_idx * num_fields_raw
                for raw_name_idx, raw_name in enumerate(raw_cols):
                    col_idx = base + raw_name_idx
                    x = X_raw[:, col_idx]                     # [N]
                    group_name = infer_group(raw_name)
                    transform_name = self.feature_transforms.get(group_name, "identity")

                    xt = apply_transform(x, transform_name, group_name)
                    if xt.dim() == 1:
                        xt = xt.unsqueeze(-1)
                    cols_out.append(xt)

            if has_count:
                count_idx = section["start"] + topk_internal * num_fields_raw
                cols_out.append(X_raw[:, count_idx].unsqueeze(-1))

        X_expanded = torch.cat(cols_out, dim=1)
        if X_expanded.shape[1] != self.output_dim:
            raise RuntimeError(
                f"Expanded feature dim mismatch: got {X_expanded.shape[1]} vs expected {self.output_dim}"
            )
        return X_expanded

    # ---------- helpers: stats & normalization ----------

    def _compute_stats_both_forms(self, X: torch.Tensor) -> Dict:
        """
        Compute normalization parameters per (block.feature_group).
        Uses the NORMALIZER_REGISTRY instead of manual mean/std logic.
        """
        stats = {"block_group_stats": {}, "_meta": {}}
        col_groups = self._expanded_column_groups()
        unique_groups = sorted(set(col_groups))

        for group_name in unique_groups:
            # Determine corresponding feature normalization type
            feature_group = group_name.split(".")[-1]  # e.g., "pt"
            norm_type = self.feature_normalizations.get(feature_group, "none")

            if norm_type == "none":
                stats["block_group_stats"][group_name] = {}  # no stats
                continue

            # Get flat values from all columns that belong to this group
            group_indices = [i for i, g in enumerate(col_groups) if g == group_name]
            values = X[:, group_indices].reshape(-1)  # flatten top-k together

            # Use the normalization registry
            normalizer = create_normalizer(norm_type, epsilon=1e-6)
            normalizer.fit(values)  # compute stats

            stats["block_group_stats"][group_name] = {
                k: float(v) if isinstance(v, torch.Tensor) else v
                for k, v in normalizer.state_dict().items()
            }

        stats["_meta"] = {
            "normalization_mode_per_feature_group": OmegaConf.to_container(self.feature_normalizations, resolve=True),
            "num_examples_fit": X.shape[0],
            "num_features_expanded": X.shape[1]
        }
        return stats



    def _apply_normalization(self, X_expanded: torch.Tensor, stats: Dict) -> torch.Tensor:
        """
        Apply normalization using the normalizer registry based on saved stats.
        """
        col_groups = self._expanded_column_groups()
        block_stats = stats["block_group_stats"]

        out = torch.empty_like(X_expanded)
        for j in range(X_expanded.shape[1]):
            group_name = col_groups[j]
            feature_group = group_name.split(".")[-1]
            norm_type = self.feature_normalizations.get(feature_group, "none")

            if norm_type == "none":
                out[:, j] = X_expanded[:, j]
                continue

            normalizer = create_normalizer(norm_type)
            normalizer.load_state_dict(block_stats[group_name])
            out[:, j] = normalizer.transform(X_expanded[:, j])

        return out


    # ---------- small utilities ----------

    def _concat_all_columns(self, fm: Dict) -> List[str]:
        cols = []
        for _, section in fm.items():
            cols.extend(section["columns"])
        return cols

    def _expanded_column_groups(self) -> List[str]:
        """
        Derive the normalization group for each expanded column, in exact alignment with the
        expanded feature order in `_transform_only()`.
        Normalization grouping is per (section + feature_group),
        repeated for each topk entry where applicable.
        """
        groups = []
        for section_name, section in self.original_fm.items():
            raw_cols = section["columns"]
            topk_raw = section.get("topk", None)
            topk_internal = topk_raw if topk_raw is not None else 1
            has_count = section.get("count", False)

            for _ in range(topk_internal):
                expanded_with_groups = expand_feature_columns(
                    raw_cols, self.feature_transforms, return_groups=True
                )

                for _, group in expanded_with_groups:
                    norm_group = f"{section_name}.{group}"
                    groups.append(norm_group)

            if has_count:
                groups.append(f"{section_name}.count")

        return groups



    def _infer_output_dim(self, fm_expanded: Dict) -> int:
        total = 0
        for _, section in fm_expanded.items():
            topk_raw = section.get("topk", None)
            topk_internal = topk_raw if topk_raw is not None else 1
            total += len(section["columns"]) * topk_internal
            if section.get("count", False):
                total += 1
        return total

    def _save_stats_json(self, stats: Dict):
        os.makedirs(self.paths["eos_preproc_dir"], exist_ok=True)
        with open(self.stats_out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Wrote normalization stats to: {self.stats_out}")

    def _load_stats_json(self) -> Dict:
        with open(self.stats_out, "r") as f:
            stats = json.load(f)
        # sanity check
        array_stats = stats.get("array_stats", {})
        fo = array_stats.get("feature_order", [])
        if len(fo) != self.output_dim:
            raise RuntimeError(
                "Normalization stats feature count/order does not match expanded feature map."
            )
        return stats
