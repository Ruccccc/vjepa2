# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def discover_ucr_files(root_path, file_patterns, exclude_datasets=None):
    """Return sorted UCR files while excluding complete dataset directories."""
    root_path = Path(root_path)
    if not root_path.is_dir():
        raise FileNotFoundError(f"UCR root directory does not exist: {root_path}")

    excluded = {name.lower() for name in (exclude_datasets or [])}
    files = {
        path
        for pattern in file_patterns
        for path in root_path.glob(pattern)
        if path.is_file() and path.parent.name.lower() not in excluded
    }
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No UCR files matched {file_patterns} below {root_path}")
    return files


def load_ucr_sequences(
    root_path,
    file_patterns,
    exclude_datasets=None,
    delimiter="\t",
    label_column=0,
    min_length=32,
    min_std=1.0e-6,
):
    """Load finite, non-constant univariate rows from UCR wide-format files."""
    files = discover_ucr_files(root_path, file_patterns, exclude_datasets)
    sequences = []
    skipped = 0

    for path in files:
        values = pd.read_csv(path, sep=delimiter, header=None).to_numpy(dtype=np.float32)
        if label_column is not None:
            if not -values.shape[1] <= label_column < values.shape[1]:
                raise ValueError(f"label_column={label_column} is invalid for {path} with {values.shape[1]} columns")
            values = np.delete(values, label_column, axis=1)

        for row in values:
            sequence = np.asarray(row, dtype=np.float32)
            if len(sequence) < min_length or not np.isfinite(sequence).all() or float(sequence.std()) < min_std:
                skipped += 1
                continue
            sequences.append(sequence.copy())

    if len(sequences) < 2:
        raise ValueError("At least two valid time-series sequences are required")

    logger.info("Loaded %d sequences from %d UCR files; skipped %d rows", len(sequences), len(files), skipped)
    return sequences


def split_sequences(sequences, validation_ratio, seed):
    """Split by original sequence before any crops are sampled."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(sequences) < 2:
        raise ValueError("At least two sequences are required for a train/validation split")

    indices = np.random.default_rng(seed).permutation(len(sequences))
    num_validation = min(len(sequences) - 1, max(1, round(len(sequences) * validation_ratio)))
    validation_indices = indices[:num_validation]
    train_indices = indices[num_validation:]
    train_sequences = [sequences[index] for index in train_indices]
    validation_sequences = [sequences[index] for index in validation_indices]
    return train_sequences, validation_sequences


class TimeSeriesCropDataset(Dataset):
    """Sample fixed-length normalized crops for TSJEPA pretrain."""

    def __init__(
        self,
        sequences,
        sequence_length,
        random_crop,
        normalize=True,
        normalization_eps=1.0e-6,
        min_std=1.0e-6,
        crop_attempts=16,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if crop_attempts < 1:
            raise ValueError("crop_attempts must be positive")
        if normalization_eps <= 0.0 or min_std < 0.0:
            raise ValueError("normalization_eps must be positive and min_std must be non-negative")

        self.sequences = [sequence for sequence in sequences if len(sequence) >= sequence_length]
        if not self.sequences:
            raise ValueError(f"No sequences are at least {sequence_length} values long")
        self.sequence_length = sequence_length
        self.random_crop = random_crop
        self.normalize = normalize
        self.normalization_eps = normalization_eps
        self.min_std = min_std
        self.crop_attempts = crop_attempts

    def __len__(self):
        return len(self.sequences)

    def _crop(self, sequence):
        max_start = len(sequence) - self.sequence_length
        if not self.random_crop or max_start == 0:
            start = max_start // 2
        else:
            start = int(torch.randint(max_start + 1, size=(1,)).item())
        return torch.from_numpy(sequence[start : start + self.sequence_length].copy())

    def _find_valid_crop(self, start_index):
        """Fallback used only when random attempts repeatedly select flat crops."""
        for offset in range(len(self.sequences)):
            sequence = self.sequences[(start_index + offset) % len(self.sequences)]
            max_start = len(sequence) - self.sequence_length
            for crop_start in range(max_start + 1):
                crop = torch.from_numpy(sequence[crop_start : crop_start + self.sequence_length].copy()).float()
                if float(crop.std(unbiased=False)) >= self.min_std:
                    return crop
        raise ValueError("No crop meets data.min_std; lower min_std or inspect the input sequences")

    def __getitem__(self, index):
        best_crop = None
        best_std = -1.0
        candidate_index = index

        for attempt in range(self.crop_attempts):
            if attempt > 0:
                candidate_index = int(torch.randint(len(self.sequences), size=(1,)).item())
            crop = self._crop(self.sequences[candidate_index]).float()
            crop_std = float(crop.std(unbiased=False))
            if crop_std > best_std:
                best_crop, best_std = crop, crop_std
            if crop_std >= self.min_std:
                break

        crop = best_crop if best_std >= self.min_std else self._find_valid_crop(index)
        if self.normalize:
            crop = (crop - crop.mean()) / crop.std(unbiased=False).clamp_min(self.normalization_eps)
        return crop
