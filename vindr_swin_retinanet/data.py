from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms.functional import to_tensor

from .boxes import Box, clip_box, iou, is_degenerate, max_ioa, max_out_of_bounds_px, median_box, pairwise_high_overlap_count
from .constants import CLASS_TO_LABEL, LOCAL_CLASS_NAMES, NO_FINDING_CLASS_NAME
from .utils import ensure_dir, load_json, write_json


@dataclass(frozen=True)
class AnnotationRecord:
    class_name: str
    label: int
    box: Box
    rad_id: str | None = None
    support: int = 1


def _merge_connected_components(boxes: list[Box], iou_threshold: float, ioa_threshold: float) -> list[Box]:
    if len(boxes) <= 1:
        return list(boxes)

    parents = list(range(len(boxes)))

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_idx in range(len(boxes)):
        for right_idx in range(left_idx + 1, len(boxes)):
            if iou(boxes[left_idx], boxes[right_idx]) >= iou_threshold or max_ioa(boxes[left_idx], boxes[right_idx]) >= ioa_threshold:
                union(left_idx, right_idx)

    groups: dict[int, list[Box]] = defaultdict(list)
    for box_idx, box in enumerate(boxes):
        groups[find(box_idx)].append(box)
    return [median_box(group_boxes) for group_boxes in groups.values()]


def merge_training_annotations(
    annotations_path: Path,
    image_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reader_groups: dict[tuple[str, str, str], list[Box]] = defaultdict(list)
    all_image_ids: set[str] = set()
    raw_box_count = 0

    with annotations_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            all_image_ids.add(image_id)
            if row["class_name"] == NO_FINDING_CLASS_NAME:
                continue
            box = (
                float(row["x_min"]),
                float(row["y_min"]),
                float(row["x_max"]),
                float(row["y_max"]),
            )
            reader_groups[(image_id, row["rad_id"], row["class_name"])].append(box)
            raw_box_count += 1

    after_reader_cleanup = 0
    pass_a: dict[str, dict[str, list[AnnotationRecord]]] = defaultdict(lambda: defaultdict(list))
    for (image_id, rad_id, class_name), boxes in reader_groups.items():
        merged_boxes = _merge_connected_components(boxes, iou_threshold=0.70, ioa_threshold=0.90)
        after_reader_cleanup += len(merged_boxes)
        for merged_box in merged_boxes:
            pass_a[image_id][class_name].append(
                AnnotationRecord(
                    class_name=class_name,
                    label=CLASS_TO_LABEL[class_name],
                    box=merged_box,
                    rad_id=rad_id,
                    support=1,
                )
            )

    final_annotations: dict[str, dict[str, Any]] = {}
    support_counts = Counter()
    high_overlap_after = 0
    coord_oob_max = 0.0
    degenerate_box_count = 0
    empty_image_count = 0

    for image_id in sorted(all_image_ids):
        image_path = image_dir / f"{image_id}.jpeg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {image_id}: {image_path}")

        with Image.open(image_path) as image:
            width, height = image.size
        merged_records: list[AnnotationRecord] = []

        for class_name, records in pass_a.get(image_id, {}).items():
            clusters: list[dict[str, Any]] = []
            sorted_records = sorted(
                records,
                key=lambda record: (
                    record.rad_id or "",
                    record.box[0],
                    record.box[1],
                    record.box[2],
                    record.box[3],
                ),
            )

            for record in sorted_records:
                candidate_clusters: list[tuple[float, int]] = []
                for cluster_idx, cluster in enumerate(clusters):
                    if record.rad_id in cluster["rad_ids"]:
                        continue
                    cluster_iou = iou(record.box, cluster["ref_box"])
                    if cluster_iou >= 0.50 or max_ioa(record.box, cluster["ref_box"]) >= 0.80:
                        candidate_clusters.append((cluster_iou, cluster_idx))

                if candidate_clusters:
                    _, best_cluster_idx = max(candidate_clusters, key=lambda item: item[0])
                    cluster = clusters[best_cluster_idx]
                    cluster["records"].append(record)
                    cluster["rad_ids"].add(record.rad_id)
                    cluster["ref_box"] = median_box([item.box for item in cluster["records"]])
                else:
                    clusters.append(
                        {
                            "records": [record],
                            "rad_ids": {record.rad_id},
                            "ref_box": record.box,
                        }
                    )

            class_boxes_after: list[Box] = []
            for cluster in clusters:
                box = clip_box(median_box([item.box for item in cluster["records"]]), width=width, height=height)
                coord_oob_max = max(coord_oob_max, max_out_of_bounds_px(box, width, height))
                if is_degenerate(box):
                    degenerate_box_count += 1
                    continue
                support = len(cluster["rad_ids"])
                support_counts[support] += 1
                class_boxes_after.append(box)
                merged_records.append(
                    AnnotationRecord(
                        class_name=class_name,
                        label=CLASS_TO_LABEL[class_name],
                        box=box,
                        rad_id=None,
                        support=support,
                    )
                )

            high_overlap_after += pairwise_high_overlap_count(class_boxes_after, iou_threshold=0.50, ioa_threshold=0.80)

        merged_records.sort(key=lambda record: (record.label, record.box[1], record.box[0]))
        final_annotations[image_id] = {
            "image_id": image_id,
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "annotations": [
                {
                    "class_name": record.class_name,
                    "label": record.label,
                    "box": [float(coord) for coord in record.box],
                    "support": record.support,
                }
                for record in merged_records
            ],
        }
        if not merged_records:
            empty_image_count += 1

    stats = {
        "merge": {
            "raw_box_count": raw_box_count,
            "after_reader_cleanup": after_reader_cleanup,
            "final_box_count": sum(len(record["annotations"]) for record in final_annotations.values()),
            "support_1": support_counts.get(1, 0),
            "support_2": support_counts.get(2, 0),
            "support_3": support_counts.get(3, 0),
            "high_overlap_same_class_pairs_after": high_overlap_after,
        },
        "coord": {
            "max_out_of_bounds_px": coord_oob_max,
            "degenerate_box_count": degenerate_box_count,
        },
        "images": {
            "total_train_images": len(final_annotations),
            "empty_image_count": empty_image_count,
        },
    }
    return final_annotations, stats


def load_test_annotations(annotations_path: Path, image_dir: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with annotations_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["class_name"] == NO_FINDING_CLASS_NAME:
                continue
            if row["class_name"] not in CLASS_TO_LABEL:
                continue
            if not row["x_min"] or not row["y_min"] or not row["x_max"] or not row["y_max"]:
                continue
            grouped[row["image_id"]].append(row)

    records: dict[str, dict[str, Any]] = {}
    for image_path in sorted(image_dir.glob("*.jpeg")):
        image_id = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = []
        for row in grouped.get(image_id, []):
            box = clip_box(
                (
                    float(row["x_min"]),
                    float(row["y_min"]),
                    float(row["x_max"]),
                    float(row["y_max"]),
                ),
                width=width,
                height=height,
            )
            if is_degenerate(box):
                continue
            annotations.append(
                {
                    "class_name": row["class_name"],
                    "label": CLASS_TO_LABEL[row["class_name"]],
                    "box": [float(coord) for coord in box],
                    "support": 1,
                }
            )
        records[image_id] = {
            "image_id": image_id,
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "annotations": annotations,
        }
    return records


def _load_split_ids_from_csv(path: Path) -> list[str]:
    seen: set[str] = set()
    split_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            if image_id in seen:
                continue
            seen.add(image_id)
            split_ids.append(image_id)
    return split_ids


def build_split_manifest(
    merged_annotations: dict[str, dict[str, Any]],
    annotations_dir: Path,
    split_mode: str,
    split_seed: int,
    val_fraction: float = 0.10,
    num_common_classes: int = 8,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if split_mode == "provided":
        train_ids = _load_split_ids_from_csv(annotations_dir / "vindr_cxr_train_90pct.csv")
        val_ids = _load_split_ids_from_csv(annotations_dir / "vindr_cxr_val_10pct.csv")
    else:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        except ImportError as exc:
            raise RuntimeError("iterative-stratification is required for split_mode='iterative'") from exc

        image_ids = sorted(merged_annotations)
        class_counter = Counter()
        for image_id in image_ids:
            for label in {annotation["label"] for annotation in merged_annotations[image_id]["annotations"]}:
                class_counter[label] += 1
        top_labels = [label for label, _ in class_counter.most_common(num_common_classes)]

        targets = np.zeros((len(image_ids), 1 + len(top_labels) + 3), dtype=np.int32)
        for idx, image_id in enumerate(image_ids):
            annotations = merged_annotations[image_id]["annotations"]
            labels = {annotation["label"] for annotation in annotations}
            targets[idx, 0] = int(bool(annotations))
            for label_idx, label in enumerate(top_labels, start=1):
                targets[idx, label_idx] = int(label in labels)
            box_count = len(annotations)
            box_bin = 0 if box_count == 0 else 1 if box_count == 1 else 2
            targets[idx, 1 + len(top_labels) + box_bin] = 1

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=split_seed)
        train_indices, val_indices = next(splitter.split(np.zeros(len(image_ids)), targets))
        train_ids = [image_ids[idx] for idx in train_indices]
        val_ids = [image_ids[idx] for idx in val_indices]

    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise RuntimeError(f"Train/val overlap detected for {len(overlap)} images")

    split_summary = {
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "empty_frac_train": float(
            sum(1 for image_id in train_ids if not merged_annotations[image_id]["annotations"]) / max(1, len(train_ids))
        ),
        "empty_frac_val": float(
            sum(1 for image_id in val_ids if not merged_annotations[image_id]["annotations"]) / max(1, len(val_ids))
        ),
        "split_mode": split_mode,
        "split_seed": split_seed,
    }
    return {"train": train_ids, "val": val_ids}, split_summary


def prepare_train_val_data(
    data_dir: Path,
    cache_dir: Path,
    split_mode: str,
    split_seed: int,
    rebuild_cache: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    ensure_dir(cache_dir)
    merged_cache = cache_dir / "merged_train_annotations.json"
    summary_cache = cache_dir / "merged_train_summary.json"
    split_cache = cache_dir / f"split_{split_mode}_{split_seed}.json"

    if merged_cache.exists() and summary_cache.exists() and not rebuild_cache:
        merged_annotations = load_json(merged_cache)
        prep_summary = load_json(summary_cache)
    else:
        merged_annotations, prep_summary = merge_training_annotations(
            annotations_path=data_dir / "annotations" / "annotations_train.csv",
            image_dir=data_dir / "train",
        )
        write_json(merged_cache, merged_annotations)
        write_json(summary_cache, prep_summary)

    if split_cache.exists() and not rebuild_cache:
        split_manifest = load_json(split_cache)
    else:
        split_manifest, split_summary = build_split_manifest(
            merged_annotations=merged_annotations,
            annotations_dir=data_dir / "annotations",
            split_mode=split_mode,
            split_seed=split_seed,
        )
        prep_summary["split"] = split_summary
        write_json(summary_cache, prep_summary)
        write_json(split_cache, split_manifest)

    return merged_annotations, split_manifest, prep_summary


def compute_sampler_weights(
    merged_annotations: dict[str, dict[str, Any]],
    train_image_ids: list[str],
) -> tuple[list[float], dict[str, Any]]:
    image_freq = Counter()
    image_classes: dict[str, set[int]] = {}
    for image_id in train_image_ids:
        labels = {annotation["label"] for annotation in merged_annotations[image_id]["annotations"]}
        image_classes[image_id] = labels
        for label in labels:
            image_freq[label] += 1

    non_zero_freqs = [freq for freq in image_freq.values() if freq > 0]
    median_freq = float(np.median(non_zero_freqs)) if non_zero_freqs else 1.0
    class_weights = {
        label: float(np.clip(math.sqrt(median_freq / freq), 1.0, 8.0))
        for label, freq in image_freq.items()
        if freq > 0
    }

    image_weights: list[float] = []
    sampled_positive = 0
    sampled_empty = 0
    per_class_exposure = Counter()

    for image_id in train_image_ids:
        labels = image_classes[image_id]
        if not labels:
            weight = 0.5
            sampled_empty += 1
        else:
            sampled_positive += 1
            weight = max(class_weights[label] for label in labels)
            for label in labels:
                per_class_exposure[label] += weight
        image_weights.append(weight)

    stats = {
        "sampler": {
            "empty_frac": sampled_empty / max(1, len(train_image_ids)),
            "positive_frac": sampled_positive / max(1, len(train_image_ids)),
            "class_weights": {
                LOCAL_CLASS_NAMES[label - 1]: class_weights[label]
                for label in sorted(class_weights)
            },
            "exposure_ratio_per_class": {
                LOCAL_CLASS_NAMES[label - 1]: per_class_exposure[label] / max(1, image_freq[label])
                for label in sorted(image_freq)
            },
        }
    }
    return image_weights, stats


class VinDrDetectionDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: int,
    ) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image_tensor = to_tensor(image.convert("L")).repeat(3, 1, 1)

        annotations = record["annotations"]
        if annotations:
            boxes = torch.tensor([annotation["box"] for annotation in annotations], dtype=torch.float32)
            labels = torch.tensor([annotation["label"] for annotation in annotations], dtype=torch.int64)
            supports = torch.tensor([annotation["support"] for annotation in annotations], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            supports = torch.zeros((0,), dtype=torch.int64)

        image_tensor, boxes, valid_mask = resize_and_pad_image_and_boxes(image_tensor, boxes, image_size=self.image_size)
        if valid_mask is not None:
            labels = labels[valid_mask]
            supports = supports[valid_mask]
        areas = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes) > 0
            else torch.zeros((0,), dtype=torch.float32)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "support": supports,
        }
        return image_tensor, target

    def raw_image_id(self, index: int) -> str:
        return self.records[index]["image_id"]


def resize_and_pad_image_and_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    _, height, width = image.shape
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized_image = torch_f.interpolate(
        image.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2
    pad_right = image_size - resized_width - pad_left
    pad_bottom = image_size - resized_height - pad_top

    padded_image = torch_f.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom))

    if boxes.numel() == 0:
        padded_boxes = torch.zeros((0, 4), dtype=torch.float32)
        valid_mask = None
    else:
        padded_boxes = boxes.clone()
        padded_boxes[:, [0, 2]] *= scale
        padded_boxes[:, [1, 3]] *= scale
        padded_boxes[:, [0, 2]] += pad_left
        padded_boxes[:, [1, 3]] += pad_top
        padded_boxes[:, 0::2] = padded_boxes[:, 0::2].clamp(0.0, float(image_size))
        padded_boxes[:, 1::2] = padded_boxes[:, 1::2].clamp(0.0, float(image_size))
        valid_mask = (padded_boxes[:, 2] > padded_boxes[:, 0]) & (padded_boxes[:, 3] > padded_boxes[:, 1])
        padded_boxes = padded_boxes[valid_mask]

    return padded_image, padded_boxes, valid_mask


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataset_records(record_map: dict[str, dict[str, Any]], image_ids: list[str]) -> list[dict[str, Any]]:
    return [record_map[image_id] for image_id in image_ids]


def build_weighted_sampler(weights: list[float], num_samples: int) -> WeightedRandomSampler:
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )
