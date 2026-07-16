# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import default_collate


class MaskCollator1D:
    """Build one contiguous interior or suffix mask for each batch."""

    def __init__(
        self,
        num_tokens,
        mask_ratio,
        interior_probability,
        suffix_probability,
        min_context_tokens=1,
        seed=None,
    ):
        if num_tokens < 2:
            raise ValueError("num_tokens must be at least 2")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        if abs(interior_probability + suffix_probability - 1.0) > 1.0e-8:
            raise ValueError("interior_probability and suffix_probability must sum to 1")
        if not 1 <= min_context_tokens < num_tokens:
            raise ValueError("min_context_tokens must be in [1, num_tokens)")

        self.num_tokens = num_tokens
        self.interior_probability = interior_probability
        self.suffix_probability = suffix_probability
        self.min_context_tokens = min_context_tokens
        self.num_masked = round(num_tokens * mask_ratio)
        self.num_masked = min(num_tokens - min_context_tokens, max(1, self.num_masked))
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)

    def sample_masks(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        use_interior = bool(torch.rand((), generator=self.generator) < self.interior_probability)
        targets = []
        contexts = []
        for _ in range(batch_size):
            if use_interior and self.num_tokens - self.num_masked > 1:
                high = self.num_tokens - self.num_masked
                start = int(torch.randint(1, high, size=(1,), generator=self.generator).item())
            else:
                start = self.num_tokens - self.num_masked

            target = torch.arange(start, start + self.num_masked, dtype=torch.long)
            keep = torch.ones(self.num_tokens, dtype=torch.bool)
            keep[target] = False
            context = torch.nonzero(keep, as_tuple=False).flatten()
            targets.append(target)
            contexts.append(context)

        return torch.stack(contexts), torch.stack(targets)

    def __call__(self, batch):
        series = default_collate(batch)
        masks_context, masks_target = self.sample_masks(series.size(0))
        return series, masks_context, masks_target
