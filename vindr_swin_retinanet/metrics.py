from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .boxes import iou
from .constants import LABEL_TO_CLASS, LOCAL_CLASS_NAMES


def _match_predictions_for_class(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    class_label: int | None,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[list[tuple[float, bool]], int]:
    pred_boxes = prediction["boxes"]
    pred_scores = prediction["scores"]
    pred_labels = prediction["labels"]
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    if class_label is not None:
        pred_mask = pred_labels == class_label
        gt_mask = gt_labels == class_label
        pred_boxes = pred_boxes[pred_mask]
        pred_scores = pred_scores[pred_mask]
        gt_boxes = gt_boxes[gt_mask]

    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    matched_gt: set[int] = set()
    ordered_indices = torch.argsort(pred_scores, descending=True)
    matched_predictions: list[tuple[float, bool]] = []
    for pred_index in ordered_indices.tolist():
        best_gt_index = -1
        best_iou = 0.0
        pred_box = tuple(float(value) for value in pred_boxes[pred_index].tolist())
        pred_label = int(pred_labels[pred_index].item()) if class_label is None else class_label
        for gt_index in range(len(gt_boxes)):
            if gt_index in matched_gt:
                continue
            if class_label is None and int(gt_labels[gt_index].item()) != pred_label:
                continue
            gt_box = tuple(float(value) for value in gt_boxes[gt_index].tolist())
            overlap = iou(pred_box, gt_box)
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index
        if best_gt_index >= 0:
            matched_gt.add(best_gt_index)
            matched_predictions.append((float(pred_scores[pred_index].item()), True))
        else:
            matched_predictions.append((float(pred_scores[pred_index].item()), False))
    return matched_predictions, int(len(gt_boxes))


def compute_map_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    map_predictions = [
        {"boxes": prediction["boxes"], "scores": prediction["scores"], "labels": prediction["labels"]}
        for prediction in predictions
    ]
    map_targets = [{"boxes": target["boxes"], "labels": target["labels"]} for target in targets]

    coco_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    map40_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.4],
        class_metrics=True,
    )
    coco_metric.update(map_predictions, map_targets)
    map40_metric.update(map_predictions, map_targets)

    coco_result = coco_metric.compute()
    map40_result = map40_metric.compute()

    per_class_ap = {}
    if "classes" in coco_result and "map_per_class" in coco_result:
        for label, ap in zip(coco_result["classes"].tolist(), coco_result["map_per_class"].tolist()):
            if label <= 0:
                continue
            per_class_ap[LABEL_TO_CLASS[int(label)]] = float(ap)

    per_class_ap40 = {}
    if "classes" in map40_result and "map_per_class" in map40_result:
        for label, ap in zip(map40_result["classes"].tolist(), map40_result["map_per_class"].tolist()):
            if label <= 0:
                continue
            per_class_ap40[LABEL_TO_CLASS[int(label)]] = float(ap)

    return {
        "val/mAP": float(coco_result["map"]),
        "val/AP50": float(coco_result["map_50"]),
        "val/mAP40": float(map40_result["map"]),
        "val/per_class_ap": per_class_ap,
        "val/per_class_ap40": per_class_ap40,
    }


def optimize_class_thresholds(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.4,
) -> dict[int, float]:
    thresholds = np.linspace(0.05, 0.95, num=19)
    best_thresholds: dict[int, float] = {}

    for class_label in range(1, len(LOCAL_CLASS_NAMES) + 1):
        best_threshold = 0.05
        best_f1 = -1.0
        best_precision = -1.0
        best_recall = -1.0
        for threshold in thresholds:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for prediction, target in zip(predictions, targets):
                matched_predictions, num_gt = _match_predictions_for_class(
                    prediction=prediction,
                    target=target,
                    class_label=class_label,
                    score_threshold=float(threshold),
                    iou_threshold=iou_threshold,
                )
                image_tp = sum(1 for _, is_tp in matched_predictions if is_tp)
                image_fp = len(matched_predictions) - image_tp
                true_positives += image_tp
                false_positives += image_fp
                false_negatives += max(0, num_gt - image_tp)
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            if precision + recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)
            if (f1, precision, recall, threshold) > (best_f1, best_precision, best_recall, best_threshold):
                best_threshold = float(threshold)
                best_f1 = f1
                best_precision = precision
                best_recall = recall
        best_thresholds[class_label] = best_threshold
    return best_thresholds


def apply_class_thresholds(
    predictions: list[dict[str, torch.Tensor]],
    thresholds: dict[int, float],
) -> list[dict[str, torch.Tensor]]:
    filtered_predictions = []
    for prediction in predictions:
        score_mask = torch.zeros_like(prediction["scores"], dtype=torch.bool)
        for label, threshold in thresholds.items():
            score_mask |= (prediction["labels"] == label) & (prediction["scores"] >= threshold)
        filtered_predictions.append({key: value[score_mask] for key, value in prediction.items()})
    return filtered_predictions


def compute_detection_summary(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    empty_target_indices = [idx for idx, target in enumerate(targets) if len(target["boxes"]) == 0]
    detections_on_empty = sum(len(predictions[idx]["boxes"]) for idx in empty_target_indices)
    detections_all = sum(len(prediction["boxes"]) for prediction in predictions)
    return {
        "val/fp_per_empty_image": detections_on_empty / max(1, len(empty_target_indices)),
        "val/avg_detections_per_image": detections_all / max(1, len(predictions)),
    }


def compute_froc_auc(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.4,
    max_fps_per_image: float = 8.0,
) -> dict[str, Any]:
    matched_predictions: list[tuple[float, bool]] = []
    total_ground_truth = 0

    for prediction, target in zip(predictions, targets):
        image_matches, num_gt = _match_predictions_for_class(
            prediction=prediction,
            target=target,
            class_label=None,
            score_threshold=0.0,
            iou_threshold=iou_threshold,
        )
        matched_predictions.extend(image_matches)
        total_ground_truth += num_gt

    matched_predictions.sort(key=lambda item: item[0], reverse=True)
    fps = [0.0]
    sensitivities = [0.0]
    true_positives = 0
    false_positives = 0
    num_images = max(1, len(targets))

    for _, is_tp in matched_predictions:
        if is_tp:
            true_positives += 1
        else:
            false_positives += 1
        fps.append(min(max_fps_per_image, false_positives / num_images))
        sensitivities.append(true_positives / max(1, total_ground_truth))

    if fps[-1] < max_fps_per_image:
        fps.append(max_fps_per_image)
        sensitivities.append(sensitivities[-1])

    integrate_trapezoid = getattr(np, "trapezoid", None)
    if integrate_trapezoid is None:
        integrate_trapezoid = np.trapz
    auc = float(integrate_trapezoid(y=sensitivities, x=fps) / max_fps_per_image)

    sensitivity_at = {}
    for operating_point in (0.5, 1.0, 2.0, 4.0, 8.0):
        valid = [sens for fp, sens in zip(fps, sensitivities) if fp <= operating_point]
        sensitivity_at[f"val/froc_sensitivity_at_{operating_point:g}_fp_per_image"] = max(valid) if valid else 0.0

    result = {
        "val/froc_auc_0_8_fp_per_image": auc,
    }
    result.update(sensitivity_at)
    return result


def format_thresholds_for_logging(thresholds: dict[int, float]) -> dict[str, float]:
    return {f"val/threshold/{LABEL_TO_CLASS[label]}": threshold for label, threshold in thresholds.items()}
