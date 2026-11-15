"""POST /api/generate - synthetic image generation (single preview + batch dataset job).

Wraps SyntheticImageGenerator, the same engine the GUI's "Generate Synthetic
Images" flow uses, so API-generated datasets are byte-for-byte compatible
with POST /api/benchmark/batch (same *.png + *_groundtruth.txt contract).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.jobs import create_job, get_job, update_job
from api.schemas import (
    ExportYoloDatasetRequest,
    GenerateDatasetRequest,
    GenerateImageResponse,
    GenerateParams,
    JobResponse,
    JobStatus,
)
from api.serialization import encode_image_base64, json_safe
from config.settings import SyntheticImageParams
from src.data_generation import SyntheticImageGenerator
from src.data_generation.yolo_exporter import YOLODatasetExporter

router = APIRouter(prefix="/api/generate", tags=["generate"])

DATASET_ROOT = Path("benchmark_results") / "dataset"


def _to_synthetic_params(params: GenerateParams) -> SyntheticImageParams:
    return SyntheticImageParams(**params.model_dump())


@router.post("", response_model=GenerateImageResponse)
def generate_image(params: GenerateParams = GenerateParams()) -> GenerateImageResponse:
    """Generate one synthetic image on demand, for a live preview in the frontend."""
    image_rgb, ground_truth = SyntheticImageGenerator().generate(_to_synthetic_params(params))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encoded = encode_image_base64(image_bgr)
    if encoded is None:
        raise HTTPException(status_code=500, detail="failed to encode generated image")
    return GenerateImageResponse(
        image=encoded,
        ground_truth=json_safe(ground_truth),
        num_particles=len(ground_truth),
    )


def _execute_dataset_generation(job_id: str, request: GenerateDatasetRequest) -> None:
    try:
        update_job(job_id, status=JobStatus.running)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.dataset_name or f"api_generated_{stamp}"
        dataset_dir = DATASET_ROOT / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        generator = SyntheticImageGenerator()
        synthetic_params = _to_synthetic_params(request.params)
        for i in range(1, request.num_images + 1):
            image_rgb, ground_truth = generator.generate(synthetic_params)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            stem = f"synthetic_{i:03d}"
            cv2.imwrite(str(dataset_dir / f"{stem}.png"), image_bgr)
            YOLODatasetExporter.save_detailed_ground_truth(
                dataset_dir / f"{stem}_groundtruth.txt", ground_truth
            )
            update_job(job_id, status=JobStatus.running, result={"generated": i, "total": request.num_images})

        update_job(
            job_id,
            status=JobStatus.done,
            result={
                "dataset_dir": str(dataset_dir.resolve()),
                "num_images": request.num_images,
            },
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job status
        update_job(job_id, status=JobStatus.error, error=str(exc))


@router.post("/dataset", response_model=JobResponse, status_code=202)
def start_dataset_generation(
    request: GenerateDatasetRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    """Generate num_images synthetic images + groundtruth files as a background job.

    The resulting folder's path (job result.dataset_dir) can be passed
    straight into POST /api/benchmark/batch's dataset_dir.
    """
    job_id = create_job()
    background_tasks.add_task(_execute_dataset_generation, job_id, request)
    return JobResponse(**get_job(job_id))


@router.get("/dataset/{job_id}", response_model=JobResponse)
def get_dataset_generation(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse(**job)


def _execute_yolo_dataset_export(job_id: str, request: ExportYoloDatasetRequest) -> None:
    """Mirrors main_window.py's export_yolo_dataset: generates train/val/test
    splits with YOLO-format .txt labels + dataset.yaml, not the detailed-
    groundtruth-only format /api/generate/dataset writes for benchmarking."""
    try:
        update_job(job_id, status=JobStatus.running)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data") / f"{request.dataset_name}_{stamp}"
        generator = SyntheticImageGenerator()
        synthetic_params = _to_synthetic_params(request.params)

        split_counts = {"train": request.train_images, "val": request.val_images, "test": request.test_images}
        exported = {}
        for split, count in split_counts.items():
            exported_count = 0
            for i in range(count):
                image_rgb, ground_truth = generator.generate(synthetic_params)
                # One image per export_dataset() call -- keeps memory flat for
                # large train_images counts instead of buffering the whole split.
                exported_count += YOLODatasetExporter.export_dataset(
                    [{"image": image_rgb, "ground_truth_data": ground_truth, "name": f"{split}_{i + 1:05d}"}],
                    output_dir,
                    split=split,
                )
            exported[split] = exported_count
            update_job(job_id, status=JobStatus.running, result={"progress": dict(exported)})

        YOLODatasetExporter.create_dataset_yaml(output_dir, dataset_name=request.dataset_name)
        update_job(
            job_id,
            status=JobStatus.done,
            result={"output_dir": str(output_dir.resolve()), "exported": exported},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job status
        update_job(job_id, status=JobStatus.error, error=str(exc))


@router.post("/yolo-dataset", response_model=JobResponse, status_code=202)
def start_yolo_dataset_export(
    request: ExportYoloDatasetRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    """Generate a full YOLO-trainable dataset (images/ + labels/ + dataset.yaml)
    under data/<dataset_name>_<timestamp>/, ready for YOLO training as-is."""
    job_id = create_job()
    background_tasks.add_task(_execute_yolo_dataset_export, job_id, request)
    return JobResponse(**get_job(job_id))


@router.get("/yolo-dataset/{job_id}", response_model=JobResponse)
def get_yolo_dataset_export(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse(**job)
