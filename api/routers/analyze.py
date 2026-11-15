"""POST /api/analyze - single-image analysis (Quick / Deep / ML)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.model_registry import registry
from api.schemas import AnalysisMethod, AnalyzeResponse
from api.serialization import serialize_analysis_result
from config.settings import PreprocessingParams
from src.analysis import DeepAnalyzer, QuickAnalyzer

router = APIRouter(prefix="/api", tags=["analyze"])

_quick = QuickAnalyzer()
_deep = DeepAnalyzer()


def _decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="could not decode image; expected PNG/JPEG bytes")
    return image_bgr


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    method: AnalysisMethod = Form(AnalysisMethod.ml),
    preprocess_method: str = Form("advanced"),
    min_area: int = Form(20),
    max_area: int = Form(10000),
    blur: int = Form(5),
    distance_thresh: float = Form(0.5),
    marker_size: int = Form(3),
    brightness_threshold: int = Form(0),
    min_blur_score: float = Form(0.0),
    adaptive_c_value: int = Form(2),
    confidence_threshold: float = Form(0.25),
    model_name: Optional[str] = Form(None),
    include_mask: bool = Form(True),
    include_annotated: bool = Form(True),
) -> AnalyzeResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    image_bgr = _decode_image(raw)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    params = PreprocessingParams(
        method=preprocess_method,
        min_area=min_area,
        max_area=max_area,
        blur=blur,
        distance_thresh=distance_thresh,
        marker_size=marker_size,
        brightness_threshold=brightness_threshold,
        min_blur_score=min_blur_score,
        adaptive_c_value=adaptive_c_value,
    )

    if method == AnalysisMethod.quick:
        result = _quick.analyze(image_rgb, params)
    elif method == AnalysisMethod.deep:
        result = _deep.analyze(image_rgb, params)
    else:
        try:
            analyzer = registry.load(model_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        result = analyzer.analyze(image_rgb, params, confidence_threshold=confidence_threshold)

    payload = serialize_analysis_result(
        result,
        original_bgr=image_bgr,
        include_mask=include_mask,
        include_annotated=include_annotated,
    )
    return AnalyzeResponse(**payload)
