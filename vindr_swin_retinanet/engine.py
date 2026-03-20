from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .metrics import apply_class_thresholds, compute_detection_summary, compute_froc_auc, compute_map_metrics, format_thresholds_for_logging, optimize_class_thresholds
from .model import optimizer_lrs
from .utils import append_jsonl, move_output_to_cpu, move_target_to_device, tensor_item, write_json


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    best_thresholds: dict[int, float],
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": best_metric,
        "best_thresholds": best_thresholds,
        "config": config,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    accumulation_steps: int,
    grad_clip_norm: float,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_steps = 0
    last_grad_norm = 0.0
    amp_overflow_count = 0

    progress = tqdm(dataloader, desc=f"train {epoch:02d}", leave=False)
    for step, (images, targets) in enumerate(progress, start=1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [move_target_to_device(target, device) for target in targets]

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / accumulation_steps

        scaler.scale(loss).backward()
        loss_scale_before = scaler.get_scale()

        should_step = step % accumulation_steps == 0 or step == len(dataloader)
        if should_step:
            scaler.unscale_(optimizer)
            last_grad_norm = float(clip_grad_norm_(model.parameters(), grad_clip_norm).item())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < loss_scale_before:
                amp_overflow_count += 1

        total_loss += float(loss_dict["classification"] + loss_dict["bbox_regression"])
        total_cls_loss += float(loss_dict["classification"])
        total_bbox_loss += float(loss_dict["bbox_regression"])
        total_steps += 1

        progress.set_postfix(
            loss=f"{total_loss / total_steps:.4f}",
            cls=f"{total_cls_loss / total_steps:.4f}",
            bbox=f"{total_bbox_loss / total_steps:.4f}",
        )

    metrics = {
        "train/loss": total_loss / max(1, total_steps),
        "train/loss_cls": total_cls_loss / max(1, total_steps),
        "train/loss_bbox": total_bbox_loss / max(1, total_steps),
        "train/grad_norm": last_grad_norm,
        "train/amp_overflow_count": float(amp_overflow_count),
    }
    metrics.update(optimizer_lrs(optimizer))
    return metrics


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    thresholds: dict[int, float] | None = None,
) -> tuple[dict[str, Any], dict[int, float]]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []

    progress = tqdm(dataloader, desc="eval", leave=False)
    for images, batch_targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            batch_outputs = model(images)
        predictions.extend(move_output_to_cpu(output) for output in batch_outputs)
        targets.extend({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in batch_targets)

    if thresholds is None:
        thresholds = optimize_class_thresholds(predictions=predictions, targets=targets, iou_threshold=0.4)

    thresholded_predictions = apply_class_thresholds(predictions, thresholds)
    metrics = {}
    metrics.update(compute_map_metrics(predictions=predictions, targets=targets))
    metrics.update(compute_froc_auc(predictions=predictions, targets=targets, iou_threshold=0.4))
    metrics.update(compute_detection_summary(predictions=thresholded_predictions, targets=targets))
    metrics.update(format_thresholds_for_logging(thresholds))
    return metrics, thresholds


def log_epoch_metrics(
    output_dir: Path,
    writer: SummaryWriter,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    serializable_metrics: dict[str, Any] = {"epoch": epoch}
    for key, value in metrics.items():
        serializable_metrics[key] = tensor_item(value)
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, epoch)
    append_jsonl(output_dir / "metrics.jsonl", serializable_metrics)
    write_json(output_dir / f"metrics_epoch_{epoch:03d}.json", serializable_metrics)
