# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.vjepa_TS.dataset import TimeSeriesCropDataset, load_ucr_sequences, split_sequences
from app.vjepa_TS.masks import MaskCollator1D
from app.vjepa_TS.models import TimeSeriesEncoder, TimeSeriesMaskPredictor, gather_tokens
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.logging import AverageMeter, CSVLogger, get_logger
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logger = get_logger(__name__)


def _seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _resolve_dtype(dtype_name):
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtypes[dtype_name]


def _validate_config(args):
    required_sections = ("meta", "data", "model", "predictor", "mask", "loss", "optimization", "checkpoint")
    missing = [section for section in required_sections if section not in args]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    data = args["data"]
    model = args["model"]
    optimization = args["optimization"]
    checkpoint = args["checkpoint"]
    if model["in_chans"] != 1:
        raise ValueError("Stage 1 TS training accepts univariate input, so model.in_chans must be 1")
    if data["sequence_length"] % model["patch_size"] != 0:
        raise ValueError("data.sequence_length must be divisible by model.patch_size")
    if data["validation_crop"] not in ("center", "random"):
        raise ValueError("data.validation_crop must be 'center' or 'random'")
    if optimization["epochs"] < 1:
        raise ValueError("optimization.epochs must be positive")
    if args["meta"]["log_freq"] < 1 or args["meta"]["eval_freq"] < 1:
        raise ValueError("meta.log_freq and meta.eval_freq must be positive")
    if not 0.0 <= optimization["warmup_ratio"] < 1.0:
        raise ValueError("optimization.warmup_ratio must be in [0, 1)")
    if not 0.0 <= optimization["ema"] < 1.0:
        raise ValueError("optimization.ema must be in [0, 1)")
    if checkpoint["monitor"] not in ("validation_loss", "train_loss"):
        raise ValueError("checkpoint.monitor must be validation_loss or train_loss")
    if checkpoint["mode"] not in ("min", "max"):
        raise ValueError("checkpoint.mode must be min or max")


def _make_datasets(data_config, seed):
    sequences = load_ucr_sequences(
        root_path=data_config["root_path"],
        file_patterns=data_config["file_patterns"],
        exclude_datasets=data_config.get("exclude_datasets"),
        delimiter=data_config["delimiter"],
        label_column=data_config["label_column"],
        min_length=data_config["sequence_length"],
        min_std=data_config["min_std"],
    )
    train_sequences, validation_sequences = split_sequences(
        sequences,
        validation_ratio=data_config["validation_ratio"],
        seed=seed,
    )
    common = {
        "sequence_length": data_config["sequence_length"],
        "normalize": data_config["normalize"],
        "normalization_eps": data_config["normalization_eps"],
        "min_std": data_config["min_std"],
        "crop_attempts": data_config["crop_attempts"],
    }
    train_dataset = TimeSeriesCropDataset(
        train_sequences,
        random_crop=data_config["train_random_crop"],
        **common,
    )
    validation_dataset = TimeSeriesCropDataset(
        validation_sequences,
        random_crop=data_config["validation_crop"] == "random",
        **common,
    )
    return train_dataset, validation_dataset


def _make_mask_collator(mask_config, num_tokens, seed=None):
    return MaskCollator1D(
        num_tokens=num_tokens,
        mask_ratio=mask_config["ratio"],
        interior_probability=mask_config["interior_probability"],
        suffix_probability=mask_config["suffix_probability"],
        min_context_tokens=mask_config["min_context_tokens"],
        seed=seed,
    )


def _make_train_loader(dataset, data_config, mask_config, num_tokens, seed):
    num_workers = data_config["num_workers"]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=data_config["shuffle_train"],
        num_workers=num_workers,
        pin_memory=data_config["pin_mem"],
        persistent_workers=data_config["persistent_workers"] and num_workers > 0,
        drop_last=data_config["train_drop_last"],
        collate_fn=_make_mask_collator(mask_config, num_tokens),
        generator=generator,
    )


def _make_validation_loader(dataset, data_config, mask_config, num_tokens):
    num_workers = data_config["validation_num_workers"]
    seed = mask_config["validation_seed"]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_config["pin_mem"],
        persistent_workers=data_config["persistent_workers"] and num_workers > 0,
        drop_last=data_config["validation_drop_last"],
        collate_fn=_make_mask_collator(mask_config, num_tokens, seed=seed),
        generator=generator,
    )


def _make_models(data_config, model_config, predictor_config, device):
    encoder = TimeSeriesEncoder(
        sequence_length=data_config["sequence_length"],
        patch_size=model_config["patch_size"],
        in_chans=model_config["in_chans"],
        embed_dim=model_config["embed_dim"],
        depth=model_config["depth"],
        num_heads=model_config["num_heads"],
        mlp_ratio=model_config["mlp_ratio"],
        qkv_bias=model_config["qkv_bias"],
        drop_rate=model_config["drop_rate"],
        attn_drop_rate=model_config["attn_drop_rate"],
        drop_path_rate=model_config["drop_path_rate"],
        norm_eps=model_config["norm_eps"],
        init_std=model_config["init_std"],
        activation=model_config["activation"],
        wide_silu=model_config["wide_silu"],
        pos_embedding=model_config["pos_embedding"],
        use_sdpa=model_config["use_sdpa"],
        use_activation_checkpointing=model_config["use_activation_checkpointing"],
    ).to(device)
    target_encoder = copy.deepcopy(encoder).requires_grad_(False)
    target_encoder.eval()
    predictor = TimeSeriesMaskPredictor(
        num_patches=encoder.num_patches,
        encoder_embed_dim=model_config["embed_dim"],
        predictor_embed_dim=predictor_config["embed_dim"],
        depth=predictor_config["depth"],
        num_heads=predictor_config["num_heads"],
        mlp_ratio=predictor_config["mlp_ratio"],
        qkv_bias=predictor_config["qkv_bias"],
        drop_rate=predictor_config["drop_rate"],
        attn_drop_rate=predictor_config["attn_drop_rate"],
        drop_path_rate=predictor_config["drop_path_rate"],
        norm_eps=predictor_config["norm_eps"],
        init_std=predictor_config["init_std"],
        activation=predictor_config["activation"],
        wide_silu=predictor_config["wide_silu"],
        num_mask_tokens=predictor_config["num_mask_tokens"],
        zero_init_mask_tokens=predictor_config["zero_init_mask_tokens"],
        use_sdpa=predictor_config["use_sdpa"],
        use_activation_checkpointing=predictor_config["use_activation_checkpointing"],
    ).to(device)
    return encoder, target_encoder, predictor


def _make_optimizer(encoder, predictor, optimization_config):
    decay_parameters = []
    no_decay_parameters = []
    for module in (encoder, predictor):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or name.endswith(".bias"):
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)

    parameter_groups = [
        {"params": decay_parameters},
        {"params": no_decay_parameters, "weight_decay": 0.0, "WD_exclude": True},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=optimization_config["lr"],
        weight_decay=optimization_config["weight_decay"],
        betas=tuple(optimization_config["betas"]),
        eps=optimization_config["eps"],
    )


def _make_schedulers(optimizer, optimization_config, total_steps):
    warmup_steps = round(total_steps * optimization_config["warmup_ratio"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=optimization_config["start_lr"],
        ref_lr=optimization_config["lr"],
        final_lr=optimization_config["final_lr"],
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=optimization_config["weight_decay"],
        final_wd=optimization_config["final_weight_decay"],
        T_max=total_steps,
    )
    return lr_scheduler, wd_scheduler


def _loss_function(prediction, target, loss_type):
    if loss_type == "l1":
        return F.l1_loss(prediction, target)
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"Unsupported loss type: {loss_type}")


def _forward_loss(
    series,
    masks_context,
    masks_target,
    encoder,
    target_encoder,
    predictor,
    loss_config,
    mask_index=0,
):
    with torch.no_grad():
        target_latents = target_encoder(series)
        if loss_config["target_layer_norm"]:
            target_latents = F.layer_norm(target_latents, (target_latents.size(-1),))
        embedding_std = target_latents.float().std(dim=(0, 1), unbiased=False).mean()
        target_latents = gather_tokens(target_latents, masks_target)

    context_latents = encoder(series, masks_context)
    prediction = predictor(context_latents, masks_context, masks_target, mask_index=mask_index)
    loss = _loss_function(prediction, target_latents, loss_config["type"])
    return loss, embedding_std


@torch.no_grad()
def _update_ema(encoder, target_encoder, momentum):
    for online_parameter, target_parameter in zip(encoder.parameters(), target_encoder.parameters()):
        target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)


def _save_checkpoint(
    path,
    encoder,
    target_encoder,
    predictor,
    optimizer,
    lr_scheduler,
    wd_scheduler,
    scaler,
    epoch,
    global_step,
    configuration,
    train_loss,
    validation_loss,
    best_metric,
):
    checkpoint = {
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "mask_predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": {"step": lr_scheduler._step},
        "wd_scheduler": {"step": wd_scheduler._step},
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "configuration": configuration,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_metric": best_metric,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load_checkpoint(
    path,
    encoder,
    target_encoder,
    predictor,
    optimizer,
    lr_scheduler,
    wd_scheduler,
    scaler,
    device,
):
    checkpoint = robust_checkpoint_loader(str(path), map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    target_encoder.load_state_dict(checkpoint["target_encoder"])
    predictor.load_state_dict(checkpoint["mask_predictor"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler._step = checkpoint["lr_scheduler"]["step"]
    wd_scheduler._step = checkpoint["wd_scheduler"]["step"]
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    target_encoder.requires_grad_(False)
    target_encoder.eval()
    return {
        "start_epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "best_metric": float(checkpoint["best_metric"]),
        "train_loss": float(checkpoint["train_loss"]),
        "validation_loss": float(checkpoint["validation_loss"]),
    }


def _autocast_context(device, dtype, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def _evaluate(
    loader,
    encoder,
    target_encoder,
    predictor,
    loss_config,
    device,
    dtype,
    amp_enabled,
    mask_index,
):
    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    loss_meter = AverageMeter()
    embedding_std_meter = AverageMeter()

    for series, masks_context, masks_target in loader:
        series = series.to(device, non_blocking=True)
        masks_context = masks_context.to(device, non_blocking=True)
        masks_target = masks_target.to(device, non_blocking=True)
        with _autocast_context(device, dtype, amp_enabled):
            loss, embedding_std = _forward_loss(
                series,
                masks_context,
                masks_target,
                encoder,
                target_encoder,
                predictor,
                loss_config,
                mask_index=mask_index,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss is not finite")
        loss_meter.update(float(loss), series.size(0))
        embedding_std_meter.update(float(embedding_std), series.size(0))

    if loss_meter.count == 0:
        raise ValueError("Validation loader produced no batches")
    return loss_meter.avg, embedding_std_meter.avg


def _is_better(metric, best_metric, mode):
    return metric < best_metric if mode == "min" else metric > best_metric


def main(args, resume_preempt=False):
    _validate_config(args)
    meta = args["meta"]
    data_config = args["data"]
    model_config = args["model"]
    predictor_config = args["predictor"]
    mask_config = args["mask"]
    loss_config = args["loss"]
    optimization_config = args["optimization"]
    checkpoint_config = args["checkpoint"]
    monitoring_config = args.get("monitoring", {})

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
        raise RuntimeError("app.vjepa_TS supports the planned single-GPU Kaggle run only")

    _seed_everything(meta["seed"], meta["deterministic"])
    device = _resolve_device(meta["device"])
    dtype = _resolve_dtype(meta["dtype"])
    amp_enabled = meta["use_amp"] and device.type == "cuda" and dtype != torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=True) if amp_enabled and dtype == torch.float16 else None
    non_blocking = data_config["pin_mem"] and device.type == "cuda"

    folder = Path(args["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    num_tokens = data_config["sequence_length"] // model_config["patch_size"]
    train_dataset, validation_dataset = _make_datasets(data_config, meta["seed"])
    train_loader = _make_train_loader(train_dataset, data_config, mask_config, num_tokens, meta["seed"])
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; reduce batch_size or disable train_drop_last")

    iterations_per_epoch = optimization_config["iterations_per_epoch"]
    if iterations_per_epoch is None:
        iterations_per_epoch = len(train_loader)
    if iterations_per_epoch < 1:
        raise ValueError("optimization.iterations_per_epoch must be positive or null")
    total_steps = optimization_config["epochs"] * iterations_per_epoch

    encoder, target_encoder, predictor = _make_models(data_config, model_config, predictor_config, device)
    optimizer = _make_optimizer(encoder, predictor, optimization_config)
    lr_scheduler, wd_scheduler = _make_schedulers(optimizer, optimization_config, total_steps)

    latest_path = folder / checkpoint_config["latest_name"]
    read_checkpoint = meta.get("read_checkpoint")
    load_path = Path(read_checkpoint) if read_checkpoint else latest_path
    should_load = (meta["load_checkpoint"] or resume_preempt) and load_path.is_file()
    start_epoch = 0
    global_step = 0
    resumed_train_loss = float("nan")
    resumed_validation_loss = float("nan")
    initial_best = float("inf") if checkpoint_config["mode"] == "min" else float("-inf")
    best_metric = initial_best
    if should_load:
        state = _load_checkpoint(
            load_path,
            encoder,
            target_encoder,
            predictor,
            optimizer,
            lr_scheduler,
            wd_scheduler,
            scaler,
            device,
        )
        start_epoch = state["start_epoch"]
        global_step = state["global_step"]
        best_metric = state["best_metric"]
        resumed_train_loss = state["train_loss"]
        resumed_validation_loss = state["validation_loss"]
        logger.info("Resumed Stage 1 training from %s at epoch %d", load_path, start_epoch)

    log_path = folder / "metrics.csv"
    csv_logger = CSVLogger(
        log_path,
        ("%d", "epoch"),
        ("%.8f", "train_loss"),
        ("%.8f", "validation_loss"),
        ("%.8f", "target_embedding_std"),
        ("%.8f", "learning_rate"),
        ("%.8f", "weight_decay"),
        mode="+a" if start_epoch > 0 and log_path.exists() else "w",
    )

    trainable_parameters = sum(
        parameter.numel()
        for module in (encoder, predictor)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    logger.info(
        "Stage 1 TS: device=%s train=%d validation=%d tokens=%d trainable_parameters=%d",
        device,
        len(train_dataset),
        len(validation_dataset),
        num_tokens,
        trainable_parameters,
    )

    final_train_loss = resumed_train_loss
    final_validation_loss = resumed_validation_loss
    final_embedding_std = float("nan")
    for epoch in range(start_epoch, optimization_config["epochs"]):
        encoder.train()
        predictor.train()
        target_encoder.eval()
        loss_meter = AverageMeter()
        loader_iterator = iter(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        current_wd = optimizer.param_groups[0]["weight_decay"]

        for iteration in range(iterations_per_epoch):
            try:
                series, masks_context, masks_target = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(train_loader)
                series, masks_context, masks_target = next(loader_iterator)

            series = series.to(device, non_blocking=non_blocking)
            masks_context = masks_context.to(device, non_blocking=non_blocking)
            masks_target = masks_target.to(device, non_blocking=non_blocking)
            current_lr = lr_scheduler.step()
            current_wd = wd_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, dtype, amp_enabled):
                loss, _ = _forward_loss(
                    series,
                    masks_context,
                    masks_target,
                    encoder,
                    target_encoder,
                    predictor,
                    loss_config,
                    mask_index=global_step,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Training loss is not finite at epoch {epoch + 1}, iteration {iteration}")

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_clip = optimization_config["grad_clip"]
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    max_norm=grad_clip,
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            _update_ema(encoder, target_encoder, optimization_config["ema"])

            global_step += 1
            loss_meter.update(float(loss), series.size(0))
            if iteration % meta["log_freq"] == 0 or iteration == iterations_per_epoch - 1:
                logger.info(
                    "epoch=%d iteration=%d/%d loss=%.6f lr=%.3e wd=%.3e",
                    epoch + 1,
                    iteration + 1,
                    iterations_per_epoch,
                    loss_meter.avg,
                    current_lr,
                    current_wd,
                )

        final_train_loss = loss_meter.avg
        should_evaluate = (epoch + 1) % meta["eval_freq"] == 0 or epoch + 1 == optimization_config["epochs"]
        if should_evaluate:
            validation_loader = _make_validation_loader(
                validation_dataset,
                data_config,
                mask_config,
                num_tokens,
            )
            final_validation_loss, final_embedding_std = _evaluate(
                validation_loader,
                encoder,
                target_encoder,
                predictor,
                loss_config,
                device,
                dtype,
                amp_enabled,
                mask_index=global_step,
            )
            logger.info(
                "epoch=%d train_loss=%.6f validation_loss=%.6f target_embedding_std=%.6f",
                epoch + 1,
                final_train_loss,
                final_validation_loss,
                final_embedding_std,
            )
            collapse_threshold = monitoring_config.get("collapse_std_threshold")
            if (
                monitoring_config.get("log_embedding_std", True)
                and collapse_threshold is not None
                and final_embedding_std < collapse_threshold
            ):
                logger.warning(
                    "Target embedding std %.6f is below collapse threshold %.6f",
                    final_embedding_std,
                    collapse_threshold,
                )

        metrics = {
            "train_loss": final_train_loss,
            "validation_loss": final_validation_loss,
        }
        monitor_value = metrics[checkpoint_config["monitor"]]
        metric_available = checkpoint_config["monitor"] == "train_loss" or should_evaluate
        improved = metric_available and _is_better(monitor_value, best_metric, checkpoint_config["mode"])
        if improved:
            best_metric = monitor_value

        checkpoint_arguments = {
            "encoder": encoder,
            "target_encoder": target_encoder,
            "predictor": predictor,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "wd_scheduler": wd_scheduler,
            "scaler": scaler,
            "epoch": epoch + 1,
            "global_step": global_step,
            "configuration": args,
            "train_loss": final_train_loss,
            "validation_loss": final_validation_loss,
            "best_metric": best_metric,
        }
        _save_checkpoint(latest_path, **checkpoint_arguments)
        if improved:
            _save_checkpoint(folder / checkpoint_config["best_name"], **checkpoint_arguments)
        save_every_freq = meta["save_every_freq"]
        if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
            _save_checkpoint(folder / f"epoch-{epoch + 1}.pt", **checkpoint_arguments)

        csv_logger.log(
            epoch + 1,
            final_train_loss,
            final_validation_loss,
            final_embedding_std,
            current_lr,
            current_wd,
        )

    return {
        "train_loss": final_train_loss,
        "validation_loss": final_validation_loss,
        "target_embedding_std": final_embedding_std,
        "latest_checkpoint": os.fspath(latest_path),
        "best_checkpoint": os.fspath(folder / checkpoint_config["best_name"]),
    }
