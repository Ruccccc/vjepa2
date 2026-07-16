# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from app.tsjepa.models import TimeSeriesEncoder
from evals.tsjepa.dataset import ForecastWindowDataset, load_ucr_rows, stratified_row_split
from evals.tsjepa.eval import main as eval_main
from evals.tsjepa.model import FrozenEncoderForecaster, TimeSeriesForecastHead, load_frozen_encoder


def _pretrain_configuration(sequence_length=16):
    return {
        "app": "tsjepa",
        "data": {"sequence_length": sequence_length},
        "model": {
            "in_chans": 1,
            "patch_size": 4,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "norm_eps": 1.0e-6,
            "init_std": 0.02,
            "activation": "gelu",
            "wide_silu": True,
            "pos_embedding": "sincos",
            "use_sdpa": False,
        },
    }


def _write_pretrain_checkpoint(path, sequence_length=16):
    configuration = _pretrain_configuration(sequence_length)
    model = configuration["model"]
    encoder = TimeSeriesEncoder(sequence_length=sequence_length, **model)
    torch.save(
        {
            "configuration": configuration,
            "target_encoder": encoder.state_dict(),
        },
        path,
    )


class TestEarthquakesForecastDataset(unittest.TestCase):
    def test_split_occurs_by_row_before_windows_are_generated(self):
        sequences = np.stack([np.arange(20, dtype=np.float32) + row * 100 for row in range(10)])
        labels = np.asarray([0] * 5 + [1] * 5, dtype=np.float32)
        train_indices, validation_indices = stratified_row_split(labels, validation_ratio=0.2, seed=7)

        self.assertFalse(set(train_indices).intersection(validation_indices))
        self.assertEqual(set(labels[validation_indices].tolist()), {0.0, 1.0})
        train_dataset = ForecastWindowDataset(
            sequences[train_indices],
            labels[train_indices],
            context_length=8,
            stride=3,
            row_indices=train_indices,
        )
        validation_dataset = ForecastWindowDataset(
            sequences[validation_indices],
            labels[validation_indices],
            context_length=8,
            final_only=True,
            row_indices=validation_indices,
        )

        self.assertGreater(len(train_dataset), len(train_indices))
        self.assertEqual(len(validation_dataset), len(validation_indices))
        for index in range(len(train_dataset)):
            context, target, _, row_index = train_dataset[index]
            expected_row = sequences[int(row_index)]
            target_positions = np.flatnonzero(expected_row == float(target))
            matching_context = any(
                np.array_equal(expected_row[position - 8 : position], context.numpy())
                for position in target_positions
            )
            self.assertTrue(matching_context)

    def test_ucr_loader_keeps_label_as_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "Earthquakes_TRAIN.tsv"
            path.write_text("1\t1\t2\t3\t4\n0\t5\t6\t7\t8\n", encoding="utf-8")
            rows, labels = load_ucr_rows(path, min_length=4)
            self.assertEqual(rows.shape, (2, 4))
            self.assertTrue(np.array_equal(labels, np.asarray([1.0, 0.0])))


class TestForecastModel(unittest.TestCase):
    def test_head_predicts_one_scalar_residual_per_sample(self):
        head = TimeSeriesForecastHead(embed_dim=16, hidden_dim=8, dropout=0.0)
        torch.nn.init.zeros_(head.network[-1].weight)
        torch.nn.init.zeros_(head.network[-1].bias)
        tokens = torch.randn(3, 4, 16)
        last_value = torch.tensor([1.0, 2.0, 3.0])
        prediction = head(tokens, last_value)
        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(torch.equal(prediction, last_value))

    def test_pretrained_target_encoder_loads_frozen(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pt"
            _write_pretrain_checkpoint(checkpoint_path)
            encoder, configuration = load_frozen_encoder(checkpoint_path, torch.device("cpu"))
            self.assertEqual(configuration["app"], "tsjepa")
            self.assertFalse(encoder.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
            self.assertEqual(encoder(torch.randn(2, 16)).shape, (2, 4, 16))

    def test_backward_updates_head_without_encoder_gradients(self):
        encoder = TimeSeriesEncoder(
            sequence_length=16,
            patch_size=4,
            embed_dim=16,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            drop_path_rate=0.0,
            use_sdpa=False,
        ).requires_grad_(False)
        head = TimeSeriesForecastHead(embed_dim=16, hidden_dim=8, dropout=0.0)
        model = FrozenEncoderForecaster(encoder, head)
        prediction = model(torch.randn(2, 16))
        prediction.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))


class TestTSJEPAEvaluationSmoke(unittest.TestCase):
    def _write_ucr_file(self, path, rows_per_label):
        rows = []
        for label in (0, 1):
            for row_index in range(rows_per_label):
                offset = label * 0.25 + row_index * 0.05
                values = [offset + np.sin(value / 4.0) for value in range(25)]
                rows.append(str(label) + "\t" + "\t".join(str(value) for value in values))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows), encoding="utf-8")

    def test_random_and_pretrained_controls_complete_and_test_only_selected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "UCRArchive_2018"
            self._write_ucr_file(data_root / "Earthquakes" / "Earthquakes_TRAIN.tsv", rows_per_label=4)
            self._write_ucr_file(data_root / "Earthquakes" / "Earthquakes_TEST.tsv", rows_per_label=2)
            pretrain_checkpoint = root / "tsjepa-pretrain-best.pt"
            _write_pretrain_checkpoint(pretrain_checkpoint)
            output = root / "forecast"
            config = {
                "eval_name": "tsjepa",
                "folder": str(output),
                "meta": {
                    "seed": 5,
                    "head_seed": 9,
                    "device": "cpu",
                    "dtype": "float32",
                    "use_amp": False,
                    "deterministic": True,
                },
                "pretrain": {"checkpoint_path": str(pretrain_checkpoint)},
                "data": {
                    "root_path": str(data_root),
                    "train_file": "Earthquakes/Earthquakes_TRAIN.tsv",
                    "test_file": "Earthquakes/Earthquakes_TEST.tsv",
                    "delimiter": "\t",
                    "label_column": 0,
                    "context_length": 16,
                    "window_stride": 4,
                    "validation_ratio": 0.25,
                    "batch_size": 4,
                    "num_workers": 0,
                    "evaluation_num_workers": 0,
                    "pin_mem": False,
                    "persistent_workers": False,
                },
                "forecast_head": {"hidden_dim": 8, "dropout": 0.0, "norm_eps": 1.0e-6},
                "loss": {"type": "huber", "delta": 1.0},
                "optimization": {
                    "epochs": 1,
                    "start_lr": 1.0e-4,
                    "lr": 1.0e-3,
                    "final_lr": 1.0e-5,
                    "warmup_ratio": 0.0,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "grad_clip": 1.0,
                },
                "experiment": {"encoder_modes": ["random", "pretrained"]},
            }

            result = eval_main(config)
            self.assertIn(result["selected_mode"], ("random", "pretrained"))
            self.assertTrue((output / "random" / "best.pt").is_file())
            self.assertTrue((output / "pretrained" / "best.pt").is_file())
            for mode, metrics in result["results"].items():
                self.assertTrue(math_is_finite(metrics["validation"]["mae"]))
                self.assertEqual("test" in metrics, mode == result["selected_mode"])
            self.assertTrue(math_is_finite(result["persistence_test"]["rmse"]))


def math_is_finite(value):
    return bool(torch.isfinite(torch.tensor(value)))


if __name__ == "__main__":
    unittest.main()
