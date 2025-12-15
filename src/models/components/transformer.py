import torch
import torch.nn as nn
import json
from typing import Dict, Any, List


class TinyTransformer(nn.Module):
    """
    A dynamic transformer classifier for physics event data.

    - Builds tokens from the flat feature vector using feature_map.json
    - Each object group contributes top-k tokens
    - If count=True, the last scalar in the slice is extracted as an event-level count feature
    - One event-level metadata token collects all count features
    - Each group has its own linear projection to d_model
    - All tokens get a type embedding
    - No positional embeddings
    - A standard TransformerEncoder processes the tokens
    """

    def __init__(
        self,
        feature_map: Dict[str, Any],
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.feature_map = feature_map
        self.d_model = d_model

        # ---- Build tokenization config ----
        self.group_names = list(feature_map.keys())

        # For each group: compute token_dim and number of tokens
        self.group_configs = []
        event_count_features = []

        for gname in self.group_names:
            cfg = feature_map[gname]
            start = cfg["start"]
            end = cfg["end"]
            topk = cfg["topk"]
            use_count = cfg.get("count", False)
            columns = cfg["columns"]
            token_dim = len(columns)

            if topk is None:
                # single object
                num_tokens = 1
            else:
                num_tokens = topk

            group_entry = {
                "name": gname,
                "start": start,
                "end": end,
                "token_dim": token_dim,
                "num_tokens": num_tokens,
                "use_count": use_count,
            }
            self.group_configs.append(group_entry)

            if use_count:
                # we will extract the last scalar in this slice later
                event_count_features.append(gname)

        # Save count feature names internally
        self.count_feature_groups = event_count_features

        # ---- Projection layers ----
        # For each group, build a Linear(token_dim → d_model)
        self.group_projections = nn.ModuleDict()
        for cfg in self.group_configs:
            self.group_projections[cfg["name"]] = nn.Linear(cfg["token_dim"], d_model)

        # ---- Event token projection ----
        # Number of count scalars = number of groups with count=True
        self.event_token_dim = len(self.count_feature_groups)
        if self.event_token_dim > 0:
            self.event_token_proj = nn.Linear(self.event_token_dim, d_model)
        else:
            # if no count features, still create learnable event token
            self.event_token_proj = nn.Linear(1, d_model)

        # ---- Type embeddings ----
        # One type embedding for each group + event token
        self.num_types = len(self.group_configs) + 1  # last type = event token
        self.type_embedding = nn.Embedding(self.num_types, d_model)

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(d_model)

        # ---- Classification head ----
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, feature_dim) into (B, total_tokens, d_model)
        """

        B = x.shape[0]
        all_tokens = []
        all_type_ids = []

        # Collect count scalars for event token
        count_values = []

        # ---- Process groups ----
        for type_id, cfg in enumerate(self.group_configs):
            start, end = cfg["start"], cfg["end"]
            num_tokens = cfg["num_tokens"]
            token_dim = cfg["token_dim"]
            use_count = cfg["use_count"]

            slice_raw = x[:, start:end]  # (B, slice_dim)

            if use_count:
                # last scalar = count
                topk_part = slice_raw[:, :-1]
                count_val = slice_raw[:, -1].unsqueeze(-1)
                count_values.append(count_val)
            else:
                topk_part = slice_raw

            # reshape top-k features to tokens
            # raw top-k part must be exactly num_tokens * token_dim
            topk_tokens = topk_part.reshape(B, num_tokens, token_dim)

            # project to d_model
            proj = self.group_projections[cfg["name"]](topk_tokens)

            # add type embedding
            type_emb = self.type_embedding(
                torch.full((B, num_tokens), type_id, device=x.device, dtype=torch.long)
            )
            #proj = proj + 0.1 * type_emb

            all_tokens.append(proj)

            # type ids stored only for debugging (optional)
            all_type_ids.append(type_id)

        # ---- Build event token ----
        if len(count_values) > 0:
            event_vec = torch.cat(count_values, dim=-1)  # (B, num_counts)
        else:
            # fallback: dummy scalar 0
            event_vec = torch.zeros(B, 1, device=x.device)

        event_token = self.event_token_proj(event_vec).unsqueeze(1)  # (B,1,d_model)

        # event token type id = last index
        event_type_id = len(self.group_configs)
        event_type_emb = self.type_embedding(
            torch.full((B, 1), event_type_id, device=x.device, dtype=torch.long)
        )
        #event_token = event_token + 0.1 * event_type_emb

        # ---- Concatenate tokens ----
        tokens = torch.cat([event_token] + all_tokens, dim=1)  # (B, total_tokens, d_model)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenize(x)
        tokens = self.input_norm(tokens)
        enc = self.encoder(tokens)
        # Take the mean over all tokens for classification
        cls_token = enc.mean(dim=1)
        logits = self.cls_head(cls_token)
        return logits
