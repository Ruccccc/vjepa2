# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_1d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


def gather_tokens(tokens, indices):
    """Gather token positions independently for every batch element."""
    if tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("tokens must be [B, N, D] and indices must be [B, K]")
    if tokens.size(0) != indices.size(0):
        raise ValueError("tokens and indices must have the same batch size")
    gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, dim=1, index=gather_index)


def _activation_layer(name):
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


class PatchEmbed1D(nn.Module):
    """Convert a scalar sequence into non-overlapping 1D patch tokens."""

    def __init__(self, patch_size=4, in_chans=1, embed_dim=128):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("Time-series input must have shape [B, L] or [B, C, L]")
        return self.proj(x).transpose(1, 2)


class TimeSeriesEncoder(nn.Module):
    """TSJEPA Transformer encoder for fixed-length univariate time-series crops."""

    def __init__(
        self,
        sequence_length=32,
        patch_size=4,
        in_chans=1,
        embed_dim=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1.0e-6,
        init_std=0.02,
        activation="gelu",
        wide_silu=True,
        pos_embedding="sincos",
        use_sdpa=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        if sequence_length % patch_size != 0:
            raise ValueError("sequence_length must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for sincos position embeddings")
        if pos_embedding != "sincos":
            raise ValueError("The initial TS implementation supports only sincos position embeddings")

        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.num_patches = sequence_length // patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.use_activation_checkpointing = use_activation_checkpointing
        self.patch_embed = PatchEmbed1D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        position = get_1d_sincos_pos_embed(embed_dim, self.num_patches)
        self.register_buffer("pos_embed", torch.from_numpy(position).float().unsqueeze(0), persistent=True)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        act_layer = _activation_layer(activation)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_paths[index],
                    act_layer=act_layer,
                    wide_silu=wide_silu,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                    use_rope=False,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.blocks, 1):
            block.attn.proj.weight.data.div_((2.0 * layer_id) ** 0.5)
            output_layer = block.mlp.fc3 if hasattr(block.mlp, "fc3") else block.mlp.fc2
            output_layer.weight.data.div_((2.0 * layer_id) ** 0.5)

    def forward(self, x, masks_context=None):
        if x.size(-1) != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, received {x.size(-1)}")
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(dtype=x.dtype)
        if masks_context is not None:
            x = gather_tokens(x, masks_context)

        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)


class TimeSeriesMaskPredictor(nn.Module):
    """Predict target encoder latents at masked 1D token positions."""

    def __init__(
        self,
        num_patches,
        encoder_embed_dim=128,
        predictor_embed_dim=128,
        depth=3,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1.0e-6,
        init_std=0.02,
        activation="gelu",
        wide_silu=True,
        num_mask_tokens=1,
        zero_init_mask_tokens=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        if predictor_embed_dim % num_heads != 0:
            raise ValueError("predictor_embed_dim must be divisible by num_heads")
        if predictor_embed_dim % 2 != 0:
            raise ValueError("predictor_embed_dim must be even for sincos position embeddings")
        if num_mask_tokens < 1:
            raise ValueError("num_mask_tokens must be positive")

        self.num_patches = num_patches
        self.num_mask_tokens = num_mask_tokens
        self.init_std = init_std
        self.use_activation_checkpointing = use_activation_checkpointing
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.mask_tokens = nn.Parameter(torch.zeros(num_mask_tokens, 1, 1, predictor_embed_dim))

        position = get_1d_sincos_pos_embed(predictor_embed_dim, num_patches)
        self.register_buffer("pos_embed", torch.from_numpy(position).float().unsqueeze(0), persistent=True)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        act_layer = _activation_layer(activation)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_paths[index],
                    act_layer=act_layer,
                    wide_silu=wide_silu,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                    use_rope=False,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self.apply(self._init_weights)
        if not zero_init_mask_tokens:
            trunc_normal_(self.mask_tokens, std=init_std)
        self._rescale_blocks()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.blocks, 1):
            block.attn.proj.weight.data.div_((2.0 * layer_id) ** 0.5)
            output_layer = block.mlp.fc3 if hasattr(block.mlp, "fc3") else block.mlp.fc2
            output_layer.weight.data.div_((2.0 * layer_id) ** 0.5)

    def forward(self, context_latents, masks_context, masks_target, mask_index=0):
        if masks_context.ndim != 2 or masks_target.ndim != 2:
            raise ValueError("masks_context and masks_target must be [B, K]")
        if context_latents.size(0) != masks_context.size(0) or masks_context.size(0) != masks_target.size(0):
            raise ValueError("Latents and masks must have the same batch size")

        batch_size, num_context, _ = context_latents.shape
        position = self.pos_embed.expand(batch_size, -1, -1)
        context = self.predictor_embed(context_latents)
        context = context + gather_tokens(position, masks_context).to(dtype=context.dtype)

        selected_mask_token = self.mask_tokens[mask_index % self.num_mask_tokens].to(dtype=context.dtype)
        target = selected_mask_token.expand(batch_size, masks_target.size(1), -1)
        target = target + gather_tokens(position, masks_target).to(dtype=target.dtype)

        token_indices = torch.cat([masks_context, masks_target], dim=1)
        x = torch.cat([context, target], dim=1)
        sort_order = torch.argsort(token_indices, dim=1)
        x = gather_tokens(x, sort_order)

        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)

        restore_order = torch.argsort(sort_order, dim=1)
        x = gather_tokens(x, restore_order)
        return self.proj(x[:, num_context:])
