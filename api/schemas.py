"""Pydantic request/response models for the API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AnalysisMethod(str, Enum):
    quick = "quick"
    deep = "deep"
    ml = "ml"


class AnalyzeParams(BaseModel):
    """Mirrors config.settings.PreprocessingParams, exposed for form/JSON input."""
    method: str = "advanced"
    min_area: int = 20
    max_area: int = 10000
    blur: int = 5
    distance_thresh: float = 0.5
    marker_size: int = 3
    brightness_threshold: int = 0
    min_blur_score: float = 0.0
    adaptive_c_value: int = 2
    confidence_threshold: float = Field(0.25, description="Only used when method=ml")


class AnalyzeResponse(BaseModel):
    analysis_type: str
    num_detections: int
    processing_time: float
    background_color: str
    features: List[Dict[str, Any]]
    mask_image: Optional[str] = None
    annotated_image: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    path: str


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    active_model: Optional[str] = None
    available_models: List[ModelInfo] = []


class BatchBenchmarkRequest(BaseModel):
    dataset_dir: str = Field(..., description="Server-side folder with *.png + *_groundtruth.txt pairs")
    model_name: Optional[str] = Field(None, description="One of the names from GET /api/models; defaults to the active model")
    confidence_threshold: float = 0.25


class GenerateParams(BaseModel):
    """Mirrors config.settings.SyntheticImageParams."""
    num_particles: int = 15
    shape_type: str = "Mixed"
    color_type: str = "Fluorescent"
    enable_blur: bool = True
    blur_kernel: int = 10
    enable_glow: bool = True
    glow_intensity: float = 0.4
    glow_sigma_min: float = 2.5
    glow_sigma_max: float = 6.0
    background_noise_min: float = 0.0
    background_noise_max: float = 0.02
    image_width: int = 1280
    image_height: int = 1280
    background_brightness: int = 0
    particle_brightness: int = 255


class GenerateImageResponse(BaseModel):
    image: str = Field(..., description="base64 PNG data URL")
    ground_truth: List[Dict[str, Any]]
    num_particles: int


class GenerateDatasetRequest(BaseModel):
    num_images: int = Field(10, gt=0, le=2000)
    params: GenerateParams = GenerateParams()
    dataset_name: Optional[str] = Field(
        None, description="Folder name under benchmark_results/dataset/; auto-generated if omitted"
    )


class ExportYoloDatasetRequest(BaseModel):
    """Mirrors the GUI's Tools > Export YOLO Training Dataset dialog."""
    train_images: int = Field(500, gt=0, le=20000)
    val_images: int = Field(100, ge=0, le=5000)
    test_images: int = Field(0, ge=0, le=5000)
    params: GenerateParams = GenerateParams()
    dataset_name: str = "microplastic"


class CalibrationRequest(BaseModel):
    """One of three calibration strategies; mirrors CameraCalibration's three
    calibrate_* methods. Only the fields for the chosen method are required."""
    method: Literal["reference", "fov", "theoretical"] = "reference"

    # method=reference
    reference_size_um: Optional[float] = None
    measured_pixels: Optional[float] = None
    axis: str = "both"

    # method=fov
    fov_width_mm: Optional[float] = None
    fov_height_mm: Optional[float] = None
    image_width_px: Optional[int] = None
    image_height_px: Optional[int] = None

    # method=theoretical
    sensor_width_mm: Optional[float] = None
    fov_degrees: Optional[float] = None
    working_distance_mm: Optional[float] = None
    resolution_width_px: Optional[int] = None
    aspect_ratio: float = 16 / 9


class CalibrationResult(BaseModel):
    um_per_pixel_x: float
    um_per_pixel_y: float
    is_calibrated: bool


class ConvertFeaturesRequest(BaseModel):
    """Apply an already-computed calibration to a list of analyze() features.

    Stateless by design: the API does not keep a server-side calibration
    singleton (that would leak across concurrent users); the caller re-sends
    um_per_pixel_x/y (from POST /api/calibration) with every conversion."""
    um_per_pixel_x: float
    um_per_pixel_y: float
    features: List[Dict[str, Any]]


class ConvertFeaturesResponse(BaseModel):
    features: List[Dict[str, Any]]


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    html_report_url: Optional[str] = None
    sidecar_json_url: Optional[str] = None
