# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_ucr_rows(path, delimiter="\t", label_column=0, min_length=33):
    """Load finite UCR rows while keeping classification labels only as metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"UCR file does not exist: {path}")

    values = pd.read_csv(path, sep=delimiter, header=None).to_numpy(dtype=np.float32)
    if not -values.shape[1] <= label_column < values.shape[1]:
        raise ValueError(f"label_column={label_column} is invalid for {path} with {values.shape[1]} columns")

    labels = values[:, label_column].copy()
    sequences = np.delete(values, label_column, axis=1)
    valid = np.isfinite(labels) & np.isfinite(sequences).all(axis=1) & (sequences.shape[1] >= min_length)
    sequences = sequences[valid]
    labels = labels[valid]
    if len(sequences) == 0:
        raise ValueError(f"No finite rows with at least {min_length} values were found in {path}")
    return sequences, labels


def stratified_row_split(labels, validation_ratio, seed):
    """Return deterministic train/validation row indices, stratified by UCR label."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("At least two one-dimensional labels are required")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    generator = np.random.default_rng(seed)
    train_indices, validation_indices = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = generator.permutation(indices)
        if len(indices) == 1:
            train_indices.extend(indices.tolist())
            continue
        num_validation = min(len(indices) - 1, max(1, round(len(indices) * validation_ratio)))
        validation_indices.extend(indices[:num_validation].tolist())
        train_indices.extend(indices[num_validation:].tolist())

    if not validation_indices:
        moved_index = train_indices.pop()
        validation_indices.append(moved_index)
    if not train_indices:
        raise ValueError("The row split produced no training rows")
    return np.sort(np.asarray(train_indices)), np.sort(np.asarray(validation_indices))


class ForecastWindowDataset(Dataset):
    """Build TSJEPA evaluation windows without crossing an original UCR row boundary."""

    def __init__(
        self,
        sequences,
        labels,
        context_length,
        stride=1,
        final_only=False,
        row_indices=None,
    ):
        sequences = np.asarray(sequences, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if sequences.ndim != 2 or labels.ndim != 1 or len(sequences) != len(labels):
            raise ValueError("sequences must be [rows, values] with one label per row")
        if context_length < 1 or stride < 1:
            raise ValueError("context_length and stride must be positive")
        if sequences.shape[1] <= context_length:
            raise ValueError("Every sequence must contain at least context_length + 1 values")
        if not np.isfinite(sequences).all():
            raise ValueError("Forecast sequences must contain only finite values")

        self.sequences = torch.from_numpy(sequences.copy())
        self.labels = torch.from_numpy(labels.copy())
        self.context_length = context_length
        self.final_only = final_only
        if row_indices is None:
            row_indices = np.arange(len(sequences))
        self.row_indices = torch.as_tensor(row_indices, dtype=torch.long)
        if len(self.row_indices) != len(sequences):
            raise ValueError("row_indices must identify every provided sequence")

        self.examples = []
        for row_position in range(len(sequences)):
            if final_only:
                target_indices = [sequences.shape[1] - 1]
            else:
                target_indices = range(context_length, sequences.shape[1], stride)
            self.examples.extend((row_position, target_index) for target_index in target_indices)
        if not self.examples:
            raise ValueError("No forecasting examples were generated")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row_position, target_index = self.examples[index]
        sequence = self.sequences[row_position]
        context = sequence[target_index - self.context_length : target_index]
        target = sequence[target_index]
        return context, target, self.labels[row_position], self.row_indices[row_position]
