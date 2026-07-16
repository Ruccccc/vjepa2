# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from evals.tsjepa.dataset import ForecastWindowDataset, load_ucr_rows, stratified_row_split
from evals.tsjepa.model import FrozenEncoderForecaster, TimeSeriesForecastHead, load_frozen_encoder
from src.utils.logging import AverageMeter, CSVLogger
from src.utils.schedulers import WarmupCosineSchedule

logger = logging.getLogger(__name__)


def _validate_config(args):
    required = {"folder", "meta", "data", "pretrain", "forecast_head", "loss", "optimization", "experiment"}
    missing = required.difference(args)
    if missing:
        raise ValueError(f"TSJEPA evaluation config is missing sections: {sorted(missing)}")
    modes = args["experiment"]["encoder_modes"]
    if not modes or any(mode not in ("random", "pretrained") for mode in modes):
        raise ValueError("experiment.encoder_modes must contain random and/or pretrained")
    if len(set(modes)) != len(modes):
        raise ValueError("experiment.encoder_modes must not contain duplicates")
    if args["loss"]["type"] != "huber":
        raise ValueError("The initial TSJEPA evaluation supports only Huber loss")


def _seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {name}")
    return device


def _resolve_dtype(name):
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtypes[name]


def _autocast_context(device, dtype, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _resolve_data_files(data_config):
    root = Path(data_config["root_path"])
    train_path = root / data_config["train_file"]
    test_path = root / data_config["test_file"]
    return train_path, test_path


def _make_datasets(data_config, seed):
    train_path, test_path = _resolve_data_files(data_config)
    context_length = data_config["context_length"]
    train_rows, train_labels = load_ucr_rows(
        train_path,
        delimiter=data_config["delimiter"],
        label_column=data_config["label_column"],
        min_length=context_length + 1,
    )
    test_rows, test_labels = load_ucr_rows(
        test_path,
        delimiter=data_config["delimiter"],
        label_column=data_config["label_column"],
        min_length=context_length + 1,
    )
    train_indices, validation_indices = stratified_row_split(
        train_labels,
        validation_ratio=data_config["validation_ratio"],
        seed=seed,
    )

    training_dataset = ForecastWindowDataset(
        train_rows[train_indices],
        train_labels[train_indices],
        context_length=context_length,
        stride=data_config["window_stride"],
        final_only=False,
        row_indices=train_indices,
    )
    validation_dataset = ForecastWindowDataset(
        train_rows[validation_indices],
        train_labels[validation_indices],
        context_length=context_length,
        final_only=True,
        row_indices=validation_indices,
    )
    test_dataset = ForecastWindowDataset(
        test_rows,
        test_labels,
        context_length=context_length,
        final_only=True,
        row_indices=np.arange(len(test_rows)),
    )
    return training_dataset, validation_dataset, test_dataset


def _make_loader(dataset, data_config, shuffle, seed, evaluation=False):
    num_workers = data_config["evaluation_num_workers"] if evaluation else data_config["num_workers"]
    return DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=data_config["pin_mem"],
        persistent_workers=data_config["persistent_workers"] and num_workers > 0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def _make_optimizer(head, optimization_config):
    decay, no_decay = [], []
    for name, parameter in head.named_parameters():
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=optimization_config["lr"],
        weight_decay=optimization_config["weight_decay"],
        betas=tuple(optimization_config["betas"]),
        eps=optimization_config["eps"],
    )


def _metric_accumulator():
    return {"absolute_error": 0.0, "squared_error": 0.0, "count": 0}


def _update_metrics(accumulator, prediction, target):
    error = prediction.float() - target.float()
    accumulator["absolute_error"] += error.abs().sum().item()
    accumulator["squared_error"] += error.square().sum().item()
    accumulator["count"] += target.numel()


def _finalize_metrics(accumulator):
    if accumulator["count"] == 0:
        raise ValueError("Cannot compute metrics for zero samples")
    count = accumulator["count"]
    return {
        "mae": accumulator["absolute_error"] / count,
        "rmse": math.sqrt(accumulator["squared_error"] / count),
    }


@torch.no_grad()
def _evaluate(model, loader, loss_function, device, dtype, amp_enabled):
    model.eval()
    loss_meter = AverageMeter()
    model_metrics = _metric_accumulator()
    persistence_metrics = _metric_accumulator()
    for context, target, _, _ in loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with _autocast_context(device, dtype, amp_enabled):
            prediction = model(context)
            loss = loss_function(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("TSJEPA evaluation loss is not finite")
        loss_meter.update(loss.item(), target.numel())
        _update_metrics(model_metrics, prediction, target)
        _update_metrics(persistence_metrics, context[:, -1], target)
    return {
        "loss": loss_meter.avg,
        **_finalize_metrics(model_metrics),
        "persistence": _finalize_metrics(persistence_metrics),
    }


def _save_checkpoint(path, model, optimizer, scheduler, epoch, configuration, mode, pretrain_checkpoint, metrics):
    checkpoint = {
        "forecast_head": model.forecast_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"step": scheduler._step},
        "epoch": epoch,
        "configuration": configuration,
        "encoder_mode": mode,
        "pretrain_checkpoint": os.fspath(pretrain_checkpoint),
        "validation_metrics": metrics,
    }
    if mode == "random":
        checkpoint["random_encoder"] = model.encoder.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load_best_head(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.forecast_head.load_state_dict(checkpoint["forecast_head"])
    if checkpoint["encoder_mode"] == "random":
        model.encoder.load_state_dict(checkpoint["random_encoder"])
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    return checkpoint


def _train_mode(args, mode, datasets, device, dtype, amp_enabled):
    meta = args["meta"]
    data_config = args["data"]
    head_config = args["forecast_head"]
    optimization = args["optimization"]
    pretrain_path = Path(args["pretrain"]["checkpoint_path"])

    _seed_everything(meta["seed"], meta["deterministic"])
    encoder, pretrain_config = load_frozen_encoder(pretrain_path, device, random_encoder=(mode == "random"))
    if data_config["context_length"] != pretrain_config["data"]["sequence_length"]:
        raise ValueError("data.context_length must equal the TSJEPA pretrain encoder sequence_length")

    # Reset before constructing the head so both encoder controls start from identical head weights.
    _seed_everything(meta["head_seed"], meta["deterministic"])
    head = TimeSeriesForecastHead(
        embed_dim=encoder.embed_dim,
        hidden_dim=head_config["hidden_dim"],
        dropout=head_config["dropout"],
        norm_eps=head_config["norm_eps"],
    ).to(device)
    model = FrozenEncoderForecaster(encoder, head).to(device)
    optimizer = _make_optimizer(head, optimization)
    training_dataset, validation_dataset, _ = datasets
    train_loader = _make_loader(training_dataset, data_config, shuffle=True, seed=meta["seed"])
    validation_loader = _make_loader(
        validation_dataset,
        data_config,
        shuffle=False,
        seed=meta["seed"],
        evaluation=True,
    )
    total_steps = optimization["epochs"] * len(train_loader)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=round(total_steps * optimization["warmup_ratio"]),
        start_lr=optimization["start_lr"],
        ref_lr=optimization["lr"],
        final_lr=optimization["final_lr"],
        T_max=total_steps,
    )
    loss_function = nn.HuberLoss(delta=args["loss"]["delta"])
    scaler = torch.cuda.amp.GradScaler(enabled=True) if amp_enabled and dtype == torch.float16 else None

    output = Path(args["folder"]) / mode
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    output.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(
        output / "metrics.csv",
        ("%d", "epoch"),
        ("%.8f", "train_loss"),
        ("%.8f", "validation_loss"),
        ("%.8f", "validation_mae"),
        ("%.8f", "validation_rmse"),
        ("%.8f", "learning_rate"),
        mode="w",
    )

    best_mae = float("inf")
    current_lr = optimization["start_lr"]
    for epoch in range(optimization["epochs"]):
        model.train()
        loss_meter = AverageMeter()
        for context, target, _, _ in train_loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            current_lr = scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, dtype, amp_enabled):
                prediction = model(context)
                loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"TSJEPA evaluation loss is not finite for {mode} at epoch {epoch + 1}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if optimization["grad_clip"] is not None and optimization["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), optimization["grad_clip"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            loss_meter.update(loss.detach().item(), target.numel())

        validation_metrics = _evaluate(model, validation_loader, loss_function, device, dtype, amp_enabled)
        logger.info(
            "TSJEPA evaluation mode=%s epoch=%d train_loss=%.6f validation_loss=%.6f mae=%.6f rmse=%.6f",
            mode,
            epoch + 1,
            loss_meter.avg,
            validation_metrics["loss"],
            validation_metrics["mae"],
            validation_metrics["rmse"],
        )
        checkpoint_args = (
            model,
            optimizer,
            scheduler,
            epoch + 1,
            args,
            mode,
            pretrain_path,
            validation_metrics,
        )
        _save_checkpoint(latest_path, *checkpoint_args)
        if validation_metrics["mae"] < best_mae:
            best_mae = validation_metrics["mae"]
            _save_checkpoint(best_path, *checkpoint_args)
        csv_logger.log(
            epoch + 1,
            loss_meter.avg,
            validation_metrics["loss"],
            validation_metrics["mae"],
            validation_metrics["rmse"],
            current_lr,
        )

    checkpoint = _load_best_head(best_path, model, device)
    reproduced_validation = _evaluate(model, validation_loader, loss_function, device, dtype, amp_enabled)
    expected_mae = checkpoint["validation_metrics"]["mae"]
    if not math.isclose(reproduced_validation["mae"], expected_mae, rel_tol=1.0e-6, abs_tol=1.0e-7):
        raise RuntimeError("Reloaded TSJEPA evaluation checkpoint did not reproduce its validation MAE")
    return model, reproduced_validation, os.fspath(best_path)


def main(args_eval, resume_preempt=False):
    del resume_preempt  # TSJEPA evaluation runs do not resume mid-run yet.
    args = args_eval
    _validate_config(args)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
        raise RuntimeError("evals.tsjepa supports single-GPU TSJEPA evaluation only")
    meta = args["meta"]
    device = _resolve_device(meta["device"])
    dtype = _resolve_dtype(meta["dtype"])
    amp_enabled = meta["use_amp"] and device.type == "cuda" and dtype != torch.float32
    datasets = _make_datasets(args["data"], meta["seed"])
    logger.info(
        "TSJEPA evaluation: device=%s train_windows=%d validation_rows=%d test_rows=%d",
        device,
        len(datasets[0]),
        len(datasets[1]),
        len(datasets[2]),
    )

    mode_results = {}
    trained_models = {}
    for mode in args["experiment"]["encoder_modes"]:
        model, validation_metrics, checkpoint_path = _train_mode(
            args,
            mode,
            datasets,
            device,
            dtype,
            amp_enabled,
        )
        trained_models[mode] = model
        mode_results[mode] = {
            "validation": validation_metrics,
            "best_checkpoint": checkpoint_path,
        }

    selected_mode = min(mode_results, key=lambda name: mode_results[name]["validation"]["mae"])
    test_loader = _make_loader(datasets[2], args["data"], shuffle=False, seed=meta["seed"], evaluation=True)
    loss_function = nn.HuberLoss(delta=args["loss"]["delta"])
    test_metrics = _evaluate(
        trained_models[selected_mode],
        test_loader,
        loss_function,
        device,
        dtype,
        amp_enabled,
    )
    mode_results[selected_mode]["test"] = test_metrics
    logger.info(
        "Selected mode=%s using validation MAE; official test mae=%.6f rmse=%.6f "
        "persistence_mae=%.6f persistence_rmse=%.6f",
        selected_mode,
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["persistence"]["mae"],
        test_metrics["persistence"]["rmse"],
    )
    return {
        "selected_mode": selected_mode,
        "results": mode_results,
        "persistence_validation": mode_results[selected_mode]["validation"]["persistence"],
        "persistence_test": test_metrics["persistence"],
    }
