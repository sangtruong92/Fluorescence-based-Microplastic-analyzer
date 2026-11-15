"""POST /api/calibration - pixel-to-micrometer camera calibration.

Wraps src.core.calibration.CameraCalibration, the engine behind the GUI's
Tools > Camera Calibration dialog. Deliberately stateless: there is no
server-side "current calibration" singleton (that would leak across
concurrent users on a shared server), so each request computes and returns
um_per_pixel_x/y, and the caller resends it to /convert for any feature list
it wants converted.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    CalibrationRequest,
    CalibrationResult,
    ConvertFeaturesRequest,
    ConvertFeaturesResponse,
)
from api.serialization import json_safe
from src.core.calibration import CameraCalibration

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


def _require(request: CalibrationRequest, *fields: str) -> None:
    missing = [name for name in fields if getattr(request, name) is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"method={request.method} requires: {', '.join(missing)}",
        )


@router.post("", response_model=CalibrationResult)
def compute_calibration(request: CalibrationRequest) -> CalibrationResult:
    calib = CameraCalibration()

    if request.method == "reference":
        _require(request, "reference_size_um", "measured_pixels")
        calib.calibrate_from_reference(request.reference_size_um, request.measured_pixels, request.axis)
    elif request.method == "fov":
        _require(request, "fov_width_mm", "fov_height_mm", "image_width_px", "image_height_px")
        calib.calibrate_from_fov(
            request.fov_width_mm, request.fov_height_mm, request.image_width_px, request.image_height_px
        )
    else:  # theoretical
        _require(request, "sensor_width_mm", "fov_degrees", "working_distance_mm", "resolution_width_px")
        calib.calibrate_theoretical(
            request.sensor_width_mm,
            request.fov_degrees,
            request.working_distance_mm,
            request.resolution_width_px,
            request.aspect_ratio,
        )

    return CalibrationResult(
        um_per_pixel_x=calib.um_per_pixel_x,
        um_per_pixel_y=calib.um_per_pixel_y,
        is_calibrated=calib.is_calibrated,
    )


@router.post("/convert", response_model=ConvertFeaturesResponse)
def convert_features(request: ConvertFeaturesRequest) -> ConvertFeaturesResponse:
    """Add *_um/diameter_um/size_category fields to analyze() features."""
    calib = CameraCalibration((request.um_per_pixel_x, request.um_per_pixel_y))
    converted = [calib.convert_features(feature) for feature in request.features]
    return ConvertFeaturesResponse(features=json_safe(converted))
