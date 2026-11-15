"""
JSON-safe serialization helpers for the API layer.

AnalysisResult.features dicts carry values the core analyzers were never
designed to serialize (numpy scalars, raw cv2 contours, binary masks as
ndarrays). This module converts them into plain JSON types without touching
the analyzer classes themselves, so the desktop GUI keeps working unchanged.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config.settings import AnalysisResult

# Keys that are either internal-only (raw cv2 data structures) or duplicated
# by other fields; the frontend has no use for them and they are not
# JSON-serializable as-is.
_DROP_KEYS = {"Contour", "contour"}


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/tuple values into JSON-compatible data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if k not in _DROP_KEYS}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def encode_image_base64(image: np.ndarray, *, as_png: bool = True) -> Optional[str]:
    """Encode an RGB/BGR/grayscale numpy image as a base64 data URL."""
    if image is None or image.size == 0:
        return None
    ext = ".png" if as_png else ".jpg"
    mime = "image/png" if as_png else "image/jpeg"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        return None
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def serialize_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [json_safe(feature) for feature in (features or [])]


def serialize_analysis_result(
    result: AnalysisResult,
    *,
    original_bgr: Optional[np.ndarray] = None,
    include_mask: bool = True,
    include_annotated: bool = True,
) -> Dict[str, Any]:
    """Build an API-friendly response payload from an AnalysisResult."""
    payload: Dict[str, Any] = {
        "analysis_type": result.analysis_type,
        "num_detections": result.num_detections,
        "processing_time": result.processing_time,
        "background_color": result.background_color,
        "features": serialize_features(result.features),
    }
    if include_mask and result.mask is not None:
        mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR) if result.mask.ndim == 2 else result.mask
        payload["mask_image"] = encode_image_base64(mask_bgr)
    if include_annotated and original_bgr is not None:
        payload["annotated_image"] = encode_image_base64(
            draw_annotations(original_bgr, result.features)
        )
    return payload


def draw_annotations(image_bgr: np.ndarray, features: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes + shape/color labels, same visual style as the GUI."""
    annotated = image_bgr.copy()
    for feature in features or []:
        bbox = feature.get("bbox_global") or feature.get("bounding_box")
        if not bbox:
            continue
        x, y, w, h = (int(round(v)) for v in bbox)
        confidence = feature.get("ml_confidence", 1.0)
        color = (0, 255, 0) if confidence > 0.5 else (255, 165, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{feature.get('shape', '')}/{feature.get('color', '')}"
        cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return annotated
