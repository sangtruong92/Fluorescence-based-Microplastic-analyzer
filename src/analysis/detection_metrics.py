"""Spatial detection metrics used by the ML batch benchmark.

The evaluator performs confidence-descending, one-to-one matching. A true
positive must have the same canonical class and meet the configured IoU
threshold. Count agreement is available only as a labelled diagnostic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config.constants import (
    BENCHMARK_CANONICAL_CLASSES,
    BENCHMARK_CLASS_ALIASES,
    BENCHMARK_EVALUATION_IOU_THRESHOLD,
)


def normalize_benchmark_class(value: Any) -> Optional[str]:
    """Return the canonical three-class benchmark label, or ``None``."""
    if value is None:
        return None
    return BENCHMARK_CLASS_ALIASES.get(str(value).strip())


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_bbox_xywh(
    bbox: Sequence[Any], image_size: Optional[Tuple[int, int]] = None
) -> List[float]:
    """Validate a pixel ``(x, y, width, height)`` box."""
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox_xywh must contain exactly four values")
    x, y, width, height = (
        _finite_float(value, name)
        for value, name in zip(bbox, ("x", "y", "width", "height"))
    )
    if x < 0 or y < 0:
        raise ValueError("bbox origin must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    if image_size is not None:
        image_width, image_height = image_size
        if x + width > image_width or y + height > image_height:
            raise ValueError("bbox must remain inside the image")
    return [x, y, width, height]


def bbox_xywh_to_xyxy(bbox: Sequence[Any]) -> List[float]:
    x, y, width, height = validate_bbox_xywh(bbox)
    return [x, y, x + width, y + height]


def bbox_iou_xywh(first: Sequence[Any], second: Sequence[Any]) -> float:
    """Calculate IoU for two pixel ``xywh`` boxes."""
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(second)
    left, top = max(ax1, bx1), max(ay1, by1)
    right, bottom = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def parse_benchmark_ground_truth(
    text: str,
    *,
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Parse the detailed synthetic GT format and require a valid bbox."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ground-truth text is empty")
    particles: List[Dict[str, Any]] = []
    blocks = re.split(r"(?m)^Particle\s+(\d+):\s*$", text)
    for offset in range(1, len(blocks), 2):
        particle_id = int(blocks[offset])
        block = blocks[offset + 1]
        fields: Dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        raw_class = fields.get("Shape")
        canonical_class = normalize_benchmark_class(raw_class)
        if canonical_class is None:
            raise ValueError(f"Particle {particle_id} has unsupported Shape: {raw_class!r}")
        bbox_match = re.fullmatch(
            r"\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)",
            fields.get("Bounding Box", ""),
        )
        if not bbox_match:
            raise ValueError(f"Particle {particle_id} is missing a valid Bounding Box")
        bbox = validate_bbox_xywh(bbox_match.groups(), image_size=image_size)
        particle: Dict[str, Any] = {
            "id": particle_id,
            "shape": canonical_class,
            "raw_shape": raw_class,
            "class_name": canonical_class,
            "bbox_xywh": bbox,
            "bounding_box": bbox,
        }
        if "Color" in fields:
            particle["color_label"] = fields["Color"]
        if "Area" in fields:
            particle["area"] = _finite_float(fields["Area"].split()[0], "area")
        if "Size" in fields:
            particle["size"] = _finite_float(fields["Size"].split()[0], "size")
        particles.append(particle)
    if not particles:
        raise ValueError("no detailed Particle blocks found")
    declared = re.search(r"(?m)^Total Particles:\s*(\d+)\s*$", text)
    if declared and int(declared.group(1)) != len(particles):
        raise ValueError("declared particle count does not match parsed particles")
    return particles


def _class_name(item: Dict[str, Any], *, ground_truth: bool) -> str:
    value = item.get("class_name", item.get("shape", item.get("class")))
    canonical = normalize_benchmark_class(value)
    if canonical is None:
        if ground_truth:
            raise ValueError(f"unsupported ground-truth class: {value!r}")
        return "Unknown"
    return canonical


def _bbox(item: Dict[str, Any]) -> List[float]:
    value = item.get("bbox_xywh", item.get("bounding_box", item.get("bbox_global")))
    return validate_bbox_xywh(value)


def _scores(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def match_detections(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    *,
    iou_threshold: float = BENCHMARK_EVALUATION_IOU_THRESHOLD,
) -> Dict[str, Any]:
    """Match predictions to GT by confidence, class and IoU."""
    threshold = _finite_float(iou_threshold, "iou_threshold")
    if not 0 < threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    prepared_predictions = [
        {
            "index": index,
            "prediction_id": item.get("prediction_id", index + 1),
            "class_name": _class_name(item, ground_truth=False),
            "bbox_xywh": _bbox(item),
            "confidence": _finite_float(item.get("confidence", 1.0), "confidence"),
        }
        for index, item in enumerate(predictions)
    ]
    prepared_truth = [
        {
            "index": index,
            "ground_truth_id": item.get("id", index + 1),
            "class_name": _class_name(item, ground_truth=True),
            "bbox_xywh": _bbox(item),
        }
        for index, item in enumerate(ground_truth)
    ]
    unmatched_truth = set(range(len(prepared_truth)))
    matches: List[Dict[str, Any]] = []
    ordered_predictions = sorted(
        prepared_predictions,
        key=lambda item: (-item["confidence"], item["index"]),
    )
    for prediction in ordered_predictions:
        candidates = []
        for truth_index in unmatched_truth:
            truth = prepared_truth[truth_index]
            if prediction["class_name"] != truth["class_name"]:
                continue
            overlap = bbox_iou_xywh(prediction["bbox_xywh"], truth["bbox_xywh"])
            candidates.append((overlap, -truth_index, truth_index))
        best_iou, _, best_truth_index = max(candidates, default=(0.0, 0, -1))
        if best_iou >= threshold:
            unmatched_truth.remove(best_truth_index)
            truth = prepared_truth[best_truth_index]
            matches.append({
                "prediction_index": prediction["index"],
                "prediction_id": prediction["prediction_id"],
                "ground_truth_index": best_truth_index,
                "ground_truth_id": truth["ground_truth_id"],
                "class_name": prediction["class_name"],
                "iou": best_iou,
            })
    matched_prediction_indices = {item["prediction_index"] for item in matches}
    result = _scores(
        len(matches),
        len(prepared_predictions) - len(matches),
        len(prepared_truth) - len(matches),
    )
    result.update({
        "available": True,
        "method": "confidence_descending_greedy_class_aware_iou",
        "iou_threshold": threshold,
        "matches": matches,
        "unmatched_prediction_indices": sorted(set(range(len(prepared_predictions))) - matched_prediction_indices),
        "unmatched_ground_truth_indices": sorted(unmatched_truth),
    })
    return result


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _summarize_matched_attributes(
    prediction_sets: Sequence[Sequence[Dict[str, Any]]],
    ground_truth_sets: Sequence[Sequence[Dict[str, Any]]],
    per_image: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare color/area on already-matched (same class, IoU>=threshold) pairs.

    Unlike the aggregate shape/color/area distributions in benchmark_batch,
    this compares each prediction directly against the exact ground-truth
    object it was matched to. Only present when the caller's prediction and
    ground-truth dicts carry 'color'/'color_label' and 'area' fields; other
    callers simply get zero matched pairs, not an error.
    """
    color_hits = 0
    color_total = 0
    color_confusion: Counter = Counter()
    area_ratios: List[float] = []

    for image_matches, predictions, truths in zip(per_image, prediction_sets, ground_truth_sets):
        for match in image_matches["matches"]:
            prediction = predictions[match["prediction_index"]]
            truth = truths[match["ground_truth_index"]]

            pred_color = prediction.get("color")
            gt_color = truth.get("color_label")
            if pred_color is not None and gt_color is not None:
                color_total += 1
                if pred_color == gt_color:
                    color_hits += 1
                color_confusion[(str(gt_color), str(pred_color))] += 1

            pred_area = prediction.get("area")
            gt_area = truth.get("area")
            if pred_area is not None and gt_area:
                try:
                    gt_area_f = float(gt_area)
                except (TypeError, ValueError):
                    gt_area_f = 0.0
                if gt_area_f > 0:
                    area_ratios.append(float(pred_area) / gt_area_f)

    return {
        "color": {
            "matched_pairs": color_total,
            "accuracy": (color_hits / color_total) if color_total else None,
            "confusion_gt_pred": {
                f"{gt}->{pred}": count
                for (gt, pred), count in color_confusion.most_common()
            },
        },
        "area": {
            "matched_pairs": len(area_ratios),
            "ratio_median": _median(area_ratios),
            "ratio_mean": (sum(area_ratios) / len(area_ratios)) if area_ratios else None,
        },
    }


def unavailable_evaluation(reason: str) -> Dict[str, Any]:
    """Return an explicit unavailable state; never substitute count agreement."""
    return {
        "available": False,
        "reason": str(reason),
        "method": "class_aware_one_to_one_iou",
        "iou_threshold": BENCHMARK_EVALUATION_IOU_THRESHOLD,
        "true_positives": None,
        "false_positives": None,
        "false_negatives": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "per_class": {},
        "per_image": [],
        "matched_attributes": {
            "color": {"matched_pairs": 0, "accuracy": None, "confusion_gt_pred": {}},
            "area": {"matched_pairs": 0, "ratio_median": None, "ratio_mean": None},
        },
    }


def evaluate_detection_dataset(
    prediction_sets: Sequence[Sequence[Dict[str, Any]]],
    ground_truth_sets: Sequence[Sequence[Dict[str, Any]]],
    *,
    image_ids: Optional[Sequence[str]] = None,
    iou_threshold: float = BENCHMARK_EVALUATION_IOU_THRESHOLD,
) -> Dict[str, Any]:
    """Aggregate spatial detection metrics over ordered images."""
    if len(prediction_sets) != len(ground_truth_sets):
        raise ValueError("prediction and ground-truth image counts must match")
    if image_ids is None:
        image_ids = [str(index) for index in range(len(prediction_sets))]
    if len(image_ids) != len(prediction_sets):
        raise ValueError("image_ids length must match prediction sets")
    per_image = []
    for image_id, predictions, truths in zip(image_ids, prediction_sets, ground_truth_sets):
        matched = match_detections(predictions, truths, iou_threshold=iou_threshold)
        matched["image_id"] = str(image_id)
        per_image.append(matched)
    tp = sum(item["true_positives"] for item in per_image)
    fp = sum(item["false_positives"] for item in per_image)
    fn = sum(item["false_negatives"] for item in per_image)
    result = _scores(tp, fp, fn)
    result.update({
        "available": True,
        "reason": None,
        "method": "confidence_descending_greedy_class_aware_iou",
        "iou_threshold": float(iou_threshold),
        "per_image": per_image,
        "matched_attributes": _summarize_matched_attributes(prediction_sets, ground_truth_sets, per_image),
    })
    per_class: Dict[str, Dict[str, Any]] = {}
    class_names = list(BENCHMARK_CANONICAL_CLASSES)
    prediction_classes = Counter(
        _class_name(item, ground_truth=False)
        for predictions in prediction_sets for item in predictions
    )
    if prediction_classes.get("Unknown"):
        class_names.append("Unknown")
    for class_name in class_names:
        class_prediction_sets = [
            [item for item in predictions if _class_name(item, ground_truth=False) == class_name]
            for predictions in prediction_sets
        ]
        class_truth_sets = [
            [item for item in truths if _class_name(item, ground_truth=True) == class_name]
            for truths in ground_truth_sets
        ]
        class_rows = [
            match_detections(predictions, truths, iou_threshold=iou_threshold)
            for predictions, truths in zip(class_prediction_sets, class_truth_sets)
        ]
        class_tp = sum(item["true_positives"] for item in class_rows)
        class_fp = sum(item["false_positives"] for item in class_rows)
        class_fn = sum(item["false_negatives"] for item in class_rows)
        per_class[class_name] = _scores(class_tp, class_fp, class_fn)
        per_class[class_name]["ground_truth_count"] = sum(len(items) for items in class_truth_sets)
        per_class[class_name]["prediction_count"] = sum(len(items) for items in class_prediction_sets)
    result["per_class"] = per_class
    return result


def legacy_count_agreement(
    prediction_sets: Sequence[Sequence[Any]],
    ground_truth_sets: Sequence[Sequence[Any]],
) -> Dict[str, Any]:
    """Reproduce the old count formula as a labelled diagnostic only."""
    if len(prediction_sets) != len(ground_truth_sets):
        raise ValueError("prediction and ground-truth image counts must match")
    tp = sum(min(len(predictions), len(truths)) for predictions, truths in zip(prediction_sets, ground_truth_sets))
    fp = sum(max(0, len(predictions) - len(truths)) for predictions, truths in zip(prediction_sets, ground_truth_sets))
    fn = sum(max(0, len(truths) - len(predictions)) for predictions, truths in zip(prediction_sets, ground_truth_sets))
    return {
        "diagnostic_only": True,
        "method": "legacy_count_agreement_not_detection_quality",
        **_scores(tp, fp, fn),
    }
