# src/preprocessing/utils.py

import json
import re
from collections import OrderedDict

##########################################
# Hardcoded Category Definitions
##########################################

PID_CATEGORIES = [
    -106, -92, -65, -45, -40, -16, -13, -11, -8, -6,
     0, 6, 11, 13, 16, 22, 40, 45, 65, 92, 106
]
BTAG_CATEGORIES = [0, 1]
CHARGE_CATEGORIES = [-1, 0, 1]

CATEGORY_MAP = {
    "pid": PID_CATEGORIES,
    "btag": BTAG_CATEGORIES,
    "charge": CHARGE_CATEGORIES
}

##########################################
# Feature Group Assignment Rules
##########################################

PATTERN_TO_GROUP = [
    (r'_PT$', 'pt'),
    (r'_Eta$', 'eta'),
    (r'_Phi$', 'phi'),
    (r'_Mass$', 'mass'),
    (r'_PID$', 'pid'),
    (r'_Charge$', 'charge'),
    (r'_JetPuppiAK\d+_BTag$', 'puppibtag'),
    (r'_JetPuppiAK\d+_BTagPhys$', 'puppibtag'),
    (r'_JetAK\d+_BTag$', 'btag'),
    (r'_JetAK\d+_BTagPhys$', 'btag'),
    (r'_PuppiW$', 'puppiw'),
    (r'_MET$', 'met'),
    (r'_MET_Phi$', 'phi'),
    (r'_PrimaryVertex_SumPT2$', 'vertexpt2'),
    (r'_PrimaryVertex_[XYZT]$', 'vertexcoord'),
    (r'_IsolationVarRhoCorr$', 'isolation'),
    (r'_EhadOverEem$', 'EhadOverEem'),
]

def infer_group(feature_name: str) -> str:
    """Infer preprocessing group name from feature name using regex patterns."""
    for pattern, group in PATTERN_TO_GROUP:
        if re.search(pattern, feature_name):
            return group
    return 'other'


##########################################
# Load Original Feature Map
##########################################

def load_feature_map(path: str):
    with open(path, 'r') as f:
        fmap = json.load(f)
    return fmap


##########################################
# Expand Columns According to Transform Rules
##########################################

def expand_feature_columns(columns, feature_transforms, return_groups=False):
    """
    Expand feature names according to transformation rules.

    If return_groups=True, returns list of (expanded_feature_name, group_name)
    """
    expanded = []
    for col in columns:
        group = infer_group(col)
        transform = feature_transforms.get(group, "identity")
        if transform == "trig":
            expanded.append((f"{col}_sin", group))
            expanded.append((f"{col}_cos", group))
        elif transform == "onehot":
            categories = CATEGORY_MAP[group]
            for cat in categories:
                expanded.append((f"{col}_is_{cat}", group))
        else:
            expanded.append((col, group))
    if return_groups:
        return expanded
    else:
        return [name for name, _ in expanded]


##########################################
# Build New Feature Map Preserving Structure
##########################################

def build_expanded_feature_map(original_feature_map, feature_transforms):
    new_feature_map = OrderedDict()
    current_index = 0

    for section_name, section in original_feature_map.items():
        raw_columns = section["columns"]
        expanded_columns = expand_feature_columns(raw_columns, feature_transforms)

        raw_topk = section.get("topk", None)
        topk_internal = raw_topk if raw_topk is not None else 1  # treat None as 1 internally
        has_count = section.get("count", False)

        num_columns_per_object = len(expanded_columns)
        total_features = num_columns_per_object * topk_internal

        if has_count:
            total_features += 1

        # Preserve None for scalars
        topk = raw_topk if raw_topk is not None else None

        new_section = {
            "start": current_index,
            "end": current_index + total_features,
            "columns": expanded_columns,
            "topk": topk,
            "count": has_count
        }

        new_feature_map[section_name] = new_section
        current_index += total_features

    return new_feature_map


##########################################
# Save New Feature Map
##########################################

def save_feature_map(feature_map, path: str):
    with open(path, 'w') as f:
        json.dump(feature_map, f, indent=2)


##########################################
# Debugging Utility
##########################################

def summarize_feature_map(feature_map):
    summary = {}
    for section, meta in feature_map.items():
        summary[section] = {
            "start": meta["start"],
            "end": meta["end"],
            "num_columns": len(meta["columns"])
        }
    return summary
