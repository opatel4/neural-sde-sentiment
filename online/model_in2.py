"""
model_in2.py  —  ONLINE fine-tuning model for the In2 (two-channel) architecture.

Replaces fine_tuning/Sentiment_Train/model.py when fine-tuning from your friend's
In2 offline weights.

WHY THIS EXISTS
---------------
The old online model was:
    Conv2d(1, 16, ...)                       # 1 channel: IV surface only
    flattened_size = 32*8*11 + context_size  # sentiment as 2 SCALARS appended
That is the "scalar sentiment" design. The In2 offline checkpoints are:
    conv_block.0.weight  (16, 2, 3, 3)       # 2 channels: IV + sentiment heatmap
    dense_block.0.weight (256, 2816)         # 2816 = 32*8*11 -> NO scalar context
so sentiment now enters ONLY as the second image channel. The scalar path is gone.

Verified: the In2 state_dict loads into this module with strict=True.

INPUT CONTRACT
--------------
forward(x) expects x of shape (B, 2, 8, 11):
    x[:, 0] = implied-volatility surface   (as before)
    x[:, 1] = sentiment heatmap            (see build_in2_input below)

The baseline (no-sentiment) variant uses the SAME module with the sentiment
channel filled with a constant (see build_in2_input(..., use_sentiment=False)),
so the two variants stay architecturally identical — only the channel content
differs. That keeps the comparison clean.
"""

import torch
import torch.nn as nn


class ParameterCNN_In2(nn.Module):
    def __init__(self, param_bounds, in_channels=2):
        super().__init__()
        self.param_bounds = param_bounds
        self.num_params = len(param_bounds)

        # --- must match the offline conv_block exactly ---
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),   # (16, 2, 3, 3)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),            # (32, 16, 3, 3)
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        # 32 channels * 8 maturities * 11 strikes = 2816.  NO context term.
        self.flattened_size = 32 * 8 * 11

        # NOTE: index 5 is a BatchNorm1d — the offline checkpoint contains
        # dense_block.5.{weight,bias,running_mean,running_var}. Must match.
        self.dense_block = nn.Sequential(
            nn.Linear(self.flattened_size, 256),   # 0  (256, 2816)
            nn.BatchNorm1d(256),                   # 1
            nn.ELU(),                              # 2
            nn.Dropout(0.2),                       # 3
            nn.Linear(256, 128),                   # 4  (128, 256)
            nn.BatchNorm1d(128),                   # 5  <-- required by checkpoint
            nn.ELU(),                              # 6
            nn.Linear(128, self.num_params),       # 7  (num_params, 128)
            nn.Sigmoid(),                          # 8
        )

        # Buffer names must match the offline checkpoint exactly.
        mins = torch.tensor([b[0] for b in param_bounds], dtype=torch.float32)
        maxs = torch.tensor([b[1] for b in param_bounds], dtype=torch.float32)
        self.register_buffer("param_mins", mins)
        self.register_buffer("param_maxs", maxs)
        self.register_buffer("param_ranges", maxs - mins)

    def forward(self, x):
        """x: (B, 2, 8, 11) -> (normalized_output, scaled_params)"""
        if x.dim() == 3:                      # (B, 8, 11) -> add channel dim
            x = x.unsqueeze(1)
        if x.size(1) == 1:                    # tolerate 1-channel by zero-padding
            x = torch.cat([x, torch.zeros_like(x)], dim=1)
        feats = self.conv_block(x)
        flat = feats.reshape(feats.size(0), -1)
        out = self.dense_block(flat)
        params = self.param_mins + out * self.param_ranges
        return out, params


def build_in2_input(iv_surface, sentiment=None, use_sentiment=True,
                    fill_value=0.0):
    """
    Stack the IV surface and the sentiment heatmap into (B, 2, 8, 11).

    iv_surface : (B, 1, 8, 11) or (B, 8, 11)
    sentiment  : one of
                   None                      -> constant `fill_value` channel
                   (B,)                      -> daily scalar broadcast over the grid
                   (B, 8)                    -> per-maturity, broadcast across strikes
                                                (this matches the paper's description)
                   (B, 8, 11)                -> full per-cell heatmap
    use_sentiment : False -> constant channel (the BASELINE variant).

    IMPORTANT: the offline data carries per-cell SS_T*_K* values, so the second
    channel there varies across the grid. Match whatever scaling your friend used
    offline (see note in the header of this file) or the channel statistics will
    differ between pretraining and fine-tuning.
    """
    iv = iv_surface if iv_surface.dim() == 4 else iv_surface.unsqueeze(1)
    B = iv.size(0)
    if (not use_sentiment) or sentiment is None:
        sent = torch.full_like(iv, float(fill_value))
    else:
        s = sentiment
        if s.dim() == 1:                       # (B,) -> (B,1,8,11)
            sent = s.view(B, 1, 1, 1).expand(B, 1, 8, 11)
        elif s.dim() == 2:                     # (B,8) -> broadcast across strikes
            sent = s.view(B, 1, 8, 1).expand(B, 1, 8, 11)
        elif s.dim() == 3:                     # (B,8,11)
            sent = s.unsqueeze(1)
        else:
            raise ValueError(f"unexpected sentiment shape {tuple(s.shape)}")
        sent = sent.to(iv.dtype).to(iv.device)
    return torch.cat([iv, sent], dim=1)


def load_in2_offline(model, ckpt_path, device="cpu", strict=True):
    """Load an In2 offline checkpoint into ParameterCNN_In2."""
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    missing, unexpected = [], []
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(f"In2 weights did not load cleanly:\n{e}")
    return model
