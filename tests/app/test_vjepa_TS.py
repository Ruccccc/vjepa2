# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from app.vjepa_TS.dataset import TimeSeriesCropDataset, discover_ucr_files, load_ucr_sequences, split_sequences
from app.vjepa_TS.masks import MaskCollator1D
from app.vjepa_TS.models import TimeSeriesEncoder, TimeSeriesMaskPredictor, gather_tokens
from app.vjepa_TS.train import (
    _load_checkpoint,
    _make_optimizer,
    _make_schedulers,
    _save_checkpoint,
    _update_ema,
    main as train_main,
)


class TestTimeSeriesDataset(unittest.TestCase):
    def test_ucr_loading_excludes_earthquakes_and_normalizes_crops(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            other = root / "OtherDataset"
            earthquake = root / "Earthquakes"
            other.mkdir()
            earthquake.mkdir()

            valid_a = "0\t" + "\t".join(str(value) for value in range(40))
            valid_b = "1\t" + "\t".join(str(value * 2) for value in range(40))
            constant = "0\t" + "\t".join("1" for _ in range(40))
            (other / "OtherDataset_TRAIN.tsv").write_text(
                f"{valid_a}\n{valid_b}\n{constant}\n", encoding="utf-8"
            )
            (other / "OtherDataset_TEST.tsv").write_text(f"{valid_a}\n{valid_b}\n", encoding="utf-8")
            (earthquake / "Earthquakes_TRAIN.tsv").write_text(f"{valid_a}\n", encoding="utf-8")
            (earthquake / "Earthquakes_TEST.tsv").write_text(f"{valid_b}\n", encoding="utf-8")

            patterns = ["*/*_TRAIN.tsv", "*/*_TEST.tsv"]
            files = discover_ucr_files(root, patterns, exclude_datasets=["Earthquakes"])
            self.assertEqual([path.parent.name for path in files], ["OtherDataset", "OtherDataset"])

            sequences = load_ucr_sequences(
                root,
                patterns,
                exclude_datasets=["Earthquakes"],
                min_length=32,
                min_std=1.0e-6,
            )
            self.assertEqual(len(sequences), 4)

            train_sequences, validation_sequences = split_sequences(sequences, validation_ratio=0.5, seed=7)
            dataset = TimeSeriesCropDataset(
                train_sequences + validation_sequences,
                sequence_length=32,
                random_crop=False,
                normalize=True,
            )
            crop = dataset[0]
            self.assertEqual(crop.shape, (32,))
            self.assertAlmostEqual(float(crop.mean()), 0.0, places=5)
            self.assertAlmostEqual(float(crop.std(unbiased=False)), 1.0, places=5)


class TestMaskCollator1D(unittest.TestCase):
    def test_interior_masks_are_disjoint_and_cover_all_tokens(self):
        collator = MaskCollator1D(
            num_tokens=8,
            mask_ratio=0.375,
            interior_probability=1.0,
            suffix_probability=0.0,
            seed=11,
        )
        contexts, targets = collator.sample_masks(batch_size=4)

        self.assertEqual(contexts.shape, (4, 5))
        self.assertEqual(targets.shape, (4, 3))
        for context, target in zip(contexts, targets):
            self.assertGreater(int(target.min()), 0)
            self.assertLess(int(target.max()), 7)
            self.assertEqual(set(context.tolist()).intersection(target.tolist()), set())
            combined = torch.sort(torch.cat([context, target])).values
            self.assertTrue(torch.equal(combined, torch.arange(8)))

    def test_suffix_mask_uses_the_end_of_the_sequence(self):
        collator = MaskCollator1D(
            num_tokens=8,
            mask_ratio=0.375,
            interior_probability=0.0,
            suffix_probability=1.0,
            seed=11,
        )
        _, targets = collator.sample_masks(batch_size=2)
        self.assertTrue(torch.equal(targets, torch.tensor([[5, 6, 7], [5, 6, 7]])))


class TestTimeSeriesModels(unittest.TestCase):
    def _make_encoder(self):
        return TimeSeriesEncoder(
            sequence_length=16,
            patch_size=4,
            embed_dim=32,
            depth=2,
            num_heads=4,
            drop_path_rate=0.0,
            use_sdpa=False,
        )

    def test_masked_latent_shapes_and_target_encoder_gradients(self):
        encoder = self._make_encoder()
        target_encoder = copy.deepcopy(encoder).requires_grad_(False)
        predictor = TimeSeriesMaskPredictor(
            num_patches=4,
            encoder_embed_dim=32,
            predictor_embed_dim=32,
            depth=1,
            num_heads=4,
            use_sdpa=False,
        )
        collator = MaskCollator1D(
            num_tokens=4,
            mask_ratio=0.5,
            interior_probability=0.0,
            suffix_probability=1.0,
            seed=3,
        )
        masks_context, masks_target = collator.sample_masks(batch_size=2)
        series = torch.randn(2, 16)

        context_latents = encoder(series, masks_context)
        with torch.no_grad():
            target_latents = F.layer_norm(target_encoder(series), (32,))
            target_latents = gather_tokens(target_latents, masks_target)
        prediction = predictor(context_latents, masks_context, masks_target)

        self.assertEqual(context_latents.shape, (2, 2, 32))
        self.assertEqual(target_latents.shape, (2, 2, 32))
        self.assertEqual(prediction.shape, target_latents.shape)

        loss = F.l1_loss(prediction, target_latents)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in predictor.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in target_encoder.parameters()))

    def test_ema_and_checkpoint_round_trip(self):
        encoder = self._make_encoder()
        target_encoder = copy.deepcopy(encoder).requires_grad_(False)
        predictor = TimeSeriesMaskPredictor(
            num_patches=4,
            encoder_embed_dim=32,
            predictor_embed_dim=32,
            depth=1,
            num_heads=4,
            use_sdpa=False,
        )
        optimization = {
            "lr": 1.0e-3,
            "start_lr": 1.0e-4,
            "final_lr": 1.0e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.04,
            "final_weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        }
        optimizer = _make_optimizer(encoder, predictor, optimization)
        lr_scheduler, wd_scheduler = _make_schedulers(optimizer, optimization, total_steps=10)
        lr_scheduler.step()
        wd_scheduler.step()

        old_target = next(target_encoder.parameters()).detach().clone()
        with torch.no_grad():
            next(encoder.parameters()).add_(1.0)
        changed_online = next(encoder.parameters()).detach().clone()
        _update_ema(encoder, target_encoder, momentum=0.5)
        self.assertTrue(torch.allclose(next(target_encoder.parameters()), 0.5 * old_target + 0.5 * changed_online))

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "latest.pt"
            _save_checkpoint(
                checkpoint_path,
                encoder,
                target_encoder,
                predictor,
                optimizer,
                lr_scheduler,
                wd_scheduler,
                scaler=None,
                epoch=2,
                global_step=7,
                configuration={"app": "vjepa_TS"},
                train_loss=0.4,
                validation_loss=0.5,
                best_metric=0.5,
            )

            restored_encoder = self._make_encoder()
            restored_target = copy.deepcopy(restored_encoder).requires_grad_(False)
            restored_predictor = copy.deepcopy(predictor)
            restored_optimizer = _make_optimizer(restored_encoder, restored_predictor, optimization)
            restored_lr_scheduler, restored_wd_scheduler = _make_schedulers(
                restored_optimizer,
                optimization,
                total_steps=10,
            )
            state = _load_checkpoint(
                checkpoint_path,
                restored_encoder,
                restored_target,
                restored_predictor,
                restored_optimizer,
                restored_lr_scheduler,
                restored_wd_scheduler,
                scaler=None,
                device=torch.device("cpu"),
            )

            self.assertEqual(state["start_epoch"], 2)
            self.assertEqual(state["global_step"], 7)
            self.assertEqual(restored_lr_scheduler._step, lr_scheduler._step)
            self.assertEqual(restored_wd_scheduler._step, wd_scheduler._step)
            for expected, restored in zip(encoder.state_dict().values(), restored_encoder.state_dict().values()):
                self.assertTrue(torch.equal(expected, restored))
            for expected, restored in zip(
                target_encoder.state_dict().values(), restored_target.state_dict().values()
            ):
                self.assertTrue(torch.equal(expected, restored))
            for expected, restored in zip(
                predictor.state_dict().values(), restored_predictor.state_dict().values()
            ):
                self.assertTrue(torch.equal(expected, restored))


class TestStage1TrainingSmoke(unittest.TestCase):
    def _config(self, root, output):
        return {
            "app": "vjepa_TS",
            "folder": str(output),
            "meta": {
                "seed": 5,
                "device": "cpu",
                "dtype": "float32",
                "use_amp": False,
                "deterministic": True,
                "log_freq": 1,
                "eval_freq": 1,
                "save_every_freq": -1,
                "load_checkpoint": False,
                "read_checkpoint": None,
            },
            "data": {
                "root_path": str(root),
                "file_patterns": ["*/*_TRAIN.tsv"],
                "exclude_datasets": ["Earthquakes"],
                "delimiter": "\t",
                "label_column": 0,
                "sequence_length": 16,
                "train_random_crop": False,
                "validation_crop": "center",
                "crop_attempts": 2,
                "normalize": True,
                "normalization_eps": 1.0e-6,
                "min_std": 1.0e-6,
                "validation_ratio": 0.25,
                "batch_size": 2,
                "shuffle_train": False,
                "num_workers": 0,
                "validation_num_workers": 0,
                "pin_mem": False,
                "persistent_workers": False,
                "train_drop_last": True,
                "validation_drop_last": False,
            },
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
                "use_activation_checkpointing": False,
            },
            "predictor": {
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
                "num_mask_tokens": 1,
                "zero_init_mask_tokens": True,
                "use_sdpa": False,
                "use_activation_checkpointing": False,
            },
            "mask": {
                "validation_seed": 6,
                "interior_probability": 0.0,
                "suffix_probability": 1.0,
                "ratio": 0.5,
                "min_context_tokens": 1,
            },
            "loss": {"type": "l1", "target_layer_norm": True},
            "optimization": {
                "epochs": 1,
                "iterations_per_epoch": 2,
                "start_lr": 1.0e-4,
                "lr": 1.0e-3,
                "final_lr": 1.0e-5,
                "warmup_ratio": 0.0,
                "weight_decay": 0.0,
                "final_weight_decay": 0.0,
                "ema": 0.9,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "grad_clip": 1.0,
            },
            "checkpoint": {
                "latest_name": "latest.pt",
                "best_name": "best.pt",
                "monitor": "validation_loss",
                "mode": "min",
            },
            "monitoring": {"log_embedding_std": True, "collapse_std_threshold": 0.001},
        }

    def test_one_epoch_cpu_smoke_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            root = temporary_directory / "UCRArchive_2018"
            dataset_directory = root / "Synthetic"
            dataset_directory.mkdir(parents=True)
            rows = []
            for row_index in range(8):
                values = [row_index + value / 10.0 for value in range(24)]
                rows.append("0\t" + "\t".join(str(value) for value in values))
            (dataset_directory / "Synthetic_TRAIN.tsv").write_text("\n".join(rows), encoding="utf-8")

            output = temporary_directory / "output"
            result = train_main(self._config(root, output))
            self.assertTrue(torch.isfinite(torch.tensor(result["train_loss"])))
            self.assertTrue(torch.isfinite(torch.tensor(result["validation_loss"])))
            self.assertTrue((output / "latest.pt").is_file())
            self.assertTrue((output / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
