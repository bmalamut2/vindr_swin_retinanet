from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .constants import NUM_CLASSES
from .data import build_dataset_records, build_weighted_sampler, collate_fn, compute_sampler_weights, load_test_annotations, prepare_train_val_data, VinDrDetectionDataset
from .engine import evaluate, load_checkpoint, log_epoch_metrics, save_checkpoint, train_one_epoch
from .model import build_model, build_optimizer, build_scheduler
from .utils import ensure_dir, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin-L RetinaNet on VinDr-CXR")
    parser.add_argument("--data-dir", type=Path, default=Path("vincxr"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/swin_retinanet_1024"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vindr_swin_retinanet"))
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-step batch size. Effective batch size stays at 16 by default via accumulation.")
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split-mode", choices=("provided", "iterative"), default="provided")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--run-test-eval", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def _resolve_accumulation_steps(batch_size: int, effective_batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if effective_batch_size < batch_size:
        raise ValueError("effective_batch_size must be at least batch_size")
    if effective_batch_size % batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by batch_size in the single-process trainer")
    return effective_batch_size // batch_size


def _flatten_thresholds(thresholds: dict[int, float]) -> dict[str, float]:
    return {str(label): float(threshold) for label, threshold in thresholds.items()}


def _serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def _build_loader(
    dataset: VinDrDetectionDataset,
    batch_size: int,
    num_workers: int,
    training: bool,
    sampler: torch.utils.data.Sampler[int] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=training and sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.cache_dir)
    write_json(args.output_dir / "config.json", _serialize_config(args))

    accumulation_steps = _resolve_accumulation_steps(
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
    )

    merged_annotations, split_manifest, prep_summary = prepare_train_val_data(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        rebuild_cache=args.rebuild_cache,
    )

    sampler_weights, sampler_summary = compute_sampler_weights(
        merged_annotations=merged_annotations,
        train_image_ids=split_manifest["train"],
    )
    prep_summary.update(sampler_summary)
    write_json(args.output_dir / "prep_summary.json", prep_summary)

    train_records = build_dataset_records(merged_annotations, split_manifest["train"])
    val_records = build_dataset_records(merged_annotations, split_manifest["val"])
    test_record_map = load_test_annotations(
        annotations_path=args.data_dir / "annotations" / "annotations_test.csv",
        image_dir=args.data_dir / "test",
    )
    test_records = build_dataset_records(test_record_map, sorted(test_record_map))

    train_dataset = VinDrDetectionDataset(records=train_records, image_size=args.image_size)
    val_dataset = VinDrDetectionDataset(records=val_records, image_size=args.image_size)
    test_dataset = VinDrDetectionDataset(records=test_records, image_size=args.image_size)

    train_loader = _build_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training=True,
        sampler=build_weighted_sampler(sampler_weights, num_samples=len(train_dataset)),
    )
    val_loader = _build_loader(
        dataset=val_dataset,
        batch_size=max(1, args.batch_size),
        num_workers=args.num_workers,
        training=False,
    )
    test_loader = _build_loader(
        dataset=test_dataset,
        batch_size=max(1, args.batch_size),
        num_workers=args.num_workers,
        training=False,
    )

    device = torch.device(args.device)
    amp_enabled = not args.disable_amp and device.type == "cuda"
    model = build_model(
        num_classes=NUM_CLASSES,
        image_size=args.image_size,
        pretrained_backbone=not args.no_pretrained_backbone,
    )
    model.to(device)

    optimizer = build_optimizer(
        model=model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=args.epochs * optimizer_steps_per_epoch,
        warmup_steps=args.warmup_steps,
    )
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    start_epoch = 0
    best_metric = float("-inf")
    best_thresholds: dict[int, float] = {}
    checkpoint_path = args.resume or args.checkpoint
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=None if args.eval_only else optimizer,
            scheduler=None if args.eval_only else scheduler,
            scaler=None if args.eval_only else scaler,
            device=device,
        )
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_thresholds = {int(label): float(value) for label, value in checkpoint.get("best_thresholds", {}).items()}
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    if args.eval_only:
        eval_loader = val_loader if args.eval_split == "val" else test_loader
        metrics, thresholds = evaluate(
            model=model,
            dataloader=eval_loader,
            device=device,
            amp_enabled=amp_enabled,
            thresholds=best_thresholds or None,
        )
        metrics["eval/split"] = args.eval_split
        write_json(args.output_dir / f"eval_{args.eval_split}.json", metrics)
        write_json(args.output_dir / f"thresholds_{args.eval_split}.json", _flatten_thresholds(thresholds))
        return

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            accumulation_steps=accumulation_steps,
            grad_clip_norm=args.grad_clip_norm,
            amp_enabled=amp_enabled,
        )
        val_metrics, current_thresholds = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            thresholds=None,
        )
        combined_metrics: dict[str, Any] = {}
        combined_metrics.update(train_metrics)
        combined_metrics.update(val_metrics)
        log_epoch_metrics(output_dir=args.output_dir, writer=writer, epoch=epoch, metrics=combined_metrics)
        write_json(args.output_dir / "thresholds_val.json", _flatten_thresholds(current_thresholds))

        current_metric = float(val_metrics["val/mAP40"])
        if current_metric >= best_metric:
            best_metric = current_metric
            best_thresholds = current_thresholds
            save_checkpoint(
                checkpoint_path=args.output_dir / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                best_thresholds=best_thresholds,
                config=_serialize_config(args),
            )
        save_checkpoint(
            checkpoint_path=args.output_dir / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            best_thresholds=best_thresholds,
            config=_serialize_config(args),
        )

    if args.run_test_eval:
        best_checkpoint = args.output_dir / "checkpoints" / "best.pt"
        if best_checkpoint.exists():
            checkpoint = load_checkpoint(best_checkpoint, model=model, device=device)
            best_thresholds = {int(label): float(value) for label, value in checkpoint.get("best_thresholds", {}).items()}
        test_metrics, thresholds = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            amp_enabled=amp_enabled,
            thresholds=best_thresholds or None,
        )
        write_json(args.output_dir / "eval_test.json", test_metrics)
        write_json(args.output_dir / "thresholds_test.json", _flatten_thresholds(thresholds))

    writer.close()
