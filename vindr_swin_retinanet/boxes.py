from __future__ import annotations

from statistics import median
from typing import Iterable, Sequence


Box = tuple[float, float, float, float]


def box_area(box: Box) -> float:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width * height


def iou(box_a: Box, box_b: Box) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = box_area(box_a) + box_area(box_b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def max_ioa(box_a: Box, box_b: Box) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    ioa_a = intersection / area_a if area_a > 0.0 else 0.0
    ioa_b = intersection / area_b if area_b > 0.0 else 0.0
    return max(ioa_a, ioa_b)


def median_box(boxes: Sequence[Box]) -> Box:
    if not boxes:
        raise ValueError("median_box requires at least one box")
    return (
        float(median(box[0] for box in boxes)),
        float(median(box[1] for box in boxes)),
        float(median(box[2] for box in boxes)),
        float(median(box[3] for box in boxes)),
    )


def clip_box(box: Box, width: int, height: int) -> Box:
    return (
        max(0.0, min(float(width), box[0])),
        max(0.0, min(float(height), box[1])),
        max(0.0, min(float(width), box[2])),
        max(0.0, min(float(height), box[3])),
    )


def max_out_of_bounds_px(box: Box, width: int, height: int) -> float:
    return max(
        0.0,
        -box[0],
        -box[1],
        box[2] - float(width),
        box[3] - float(height),
    )


def is_degenerate(box: Box, eps: float = 1e-3) -> bool:
    return (box[2] - box[0]) <= eps or (box[3] - box[1]) <= eps


def pairwise_high_overlap_count(boxes: Iterable[Box], iou_threshold: float, ioa_threshold: float) -> int:
    box_list = list(boxes)
    overlaps = 0
    for idx in range(len(box_list)):
        for jdx in range(idx + 1, len(box_list)):
            if iou(box_list[idx], box_list[jdx]) >= iou_threshold or max_ioa(box_list[idx], box_list[jdx]) >= ioa_threshold:
                overlaps += 1
    return overlaps
