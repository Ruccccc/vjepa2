# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from app.tsjepa.models import TimeSeriesEncoder
from src.utils.checkpoint_loader import robust_checkpoint_loader


def _encoder_from_pretrain_configuration(configuration):
    if configuration.get("app") != "tsjepa":
        raise ValueError("TSJEPA evaluation requires a TSJEPA pretrain checkpoint")
    data = configuration["data"]
    model = configuration["model"]
    return TimeSeriesEncoder(
        sequence_length=data["sequence_length"],
        patch_size=model["patch_size"],
        in_chans=model["in_chans"],
        embed_dim=model["embed_dim"],
        depth=model["depth"],
        num_heads=model["num_heads"],
        mlp_ratio=model["mlp_ratio"],
        qkv_bias=model["qkv_bias"],
        drop_rate=model["drop_rate"],
        attn_drop_rate=model["attn_drop_rate"],
        drop_path_rate=model["drop_path_rate"],
        norm_eps=model["norm_eps"],
        init_std=model["init_std"],
        activation=model["activation"],
        wide_silu=model["wide_silu"],
        pos_embedding=model["pos_embedding"],
        use_sdpa=model["use_sdpa"],
        use_activation_checkpointing=False,
    )


def load_frozen_encoder(checkpoint_path, device, random_encoder=False):
    """Rebuild the TSJEPA encoder and optionally load its pretrained EMA target weights."""
    checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location="cpu")
    if "configuration" not in checkpoint or "target_encoder" not in checkpoint:
        raise ValueError("TSJEPA pretrain checkpoint is missing configuration or target_encoder")
    encoder = _encoder_from_pretrain_configuration(checkpoint["configuration"])
    if not random_encoder:
        encoder.load_state_dict(checkpoint["target_encoder"])
    encoder.to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder, checkpoint["configuration"]


class TimeSeriesForecastHead(nn.Module):
    """Predict a residual over the final observed scalar from frozen encoder tokens."""

    def __init__(self, embed_dim, hidden_dim=128, dropout=0.1, norm_eps=1.0e-6):
        super().__init__()
        if embed_dim < 1 or hidden_dim < 1:
            raise ValueError("embed_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=norm_eps),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoder_tokens, last_value):
        if encoder_tokens.ndim != 3:
            raise ValueError("encoder_tokens must have shape [B, N, D]")
        if last_value.ndim != 1 or last_value.size(0) != encoder_tokens.size(0):
            raise ValueError("last_value must have shape [B]")
        predicted_delta = self.network(encoder_tokens[:, -1]).squeeze(-1)
        return last_value + predicted_delta


class FrozenEncoderForecaster(nn.Module):
    """Composition used for evaluation; only the forecast head is trainable."""

    def __init__(self, encoder, forecast_head):
        super().__init__()
        self.encoder = encoder
        self.forecast_head = forecast_head

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, context):
        with torch.no_grad():
            tokens = self.encoder(context)
        return self.forecast_head(tokens, context[:, -1])
