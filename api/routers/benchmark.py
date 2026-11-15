"""POST /api/benchmark/batch - async batch benchmark over a server-side dataset folder.

Runs Quick/Deep/ML over every *.png (+ matching *_groundtruth.txt) under
dataset_dir, the same contract as run_ml_batch_benchmark.py, but as a
background job with status polling instead of a blocking request (a 200
image run takes 40-90s, too long for a single synchronous HTTP call).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.jobs import create_job, get_job, update_job
from api.model_registry import registry
from api.schemas import BatchBenchmarkRequest, JobResponse, JobStatus
from api.serialization import json_safe
from config.settings import PreprocessingParams
from src.analysis import DeepAnalyzer, QuickAnalyzer, ReportGenerator
from src.analysis.benchmark_batch import build_ml_batch_payload, load_benchmark_images, run_analyzer_over_images
from src.analysis.benchmark_snapshot import save_benchmark_snapshot

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

# Mirrors GUI widget defaults in src/gui/main_window.py (get_analysis_params)
QUICK_PARAMS = PreprocessingParams(method="basic", min_area=20, max_area=10000, blur=5, distance_thresh=0.5)
DEEP_PARAMS = PreprocessingParams(
    method="advanced", min_area=20, max_area=10000, blur=3, distance_thresh=0.4,
    marker_size=5, brightness_threshold=0, min_blur_score=0.0, adaptive_c_value=2,
)
OUTPUT_DIR = Path("benchmark_results")


def _execute_batch_benchmark(
    job_id: str, dataset_dir: str, model_name: Optional[str], confidence_threshold: float
) -> None:
    try:
        update_job(job_id, status=JobStatus.running)
        images_data = load_benchmark_images(Path(dataset_dir))
        analyzer = registry.load(model_name)
        model_path = str(registry.available[registry.active_name])

        quick_results = run_analyzer_over_images(QuickAnalyzer(), images_data, QUICK_PARAMS)
        deep_results = run_analyzer_over_images(DeepAnalyzer(), images_data, DEEP_PARAMS)
        ml_results = run_analyzer_over_images(analyzer, images_data, DEEP_PARAMS, confidence_threshold=confidence_threshold)

        timestamp = datetime.now()
        payload, evidence = build_ml_batch_payload(
            images_data,
            {"quick_analysis": quick_results, "deep_analysis": deep_results, "ml_benchmark": ml_results},
            timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

        OUTPUT_DIR.mkdir(exist_ok=True)
        stamp = timestamp.strftime("%Y%m%d_%H%M%S")
        html_path = OUTPUT_DIR / f"api_benchmark_{len(images_data)}images_{stamp}.html"
        ReportGenerator().generate_benchmark_report(payload, str(html_path))
        snapshot_path = save_benchmark_snapshot(html_path, evidence)

        summary = {
            method: {
                "available": payload[method].get("available"),
                "reason": payload[method].get("reason"),
                "precision": payload[method].get("precision"),
                "recall": payload[method].get("recall"),
                "f1_score": payload[method].get("f1_score"),
                "per_class": payload[method].get("per_class"),
                "matched_attributes": payload[method].get("matched_attributes"),
            }
            for method in ("quick_analysis", "deep_analysis", "ml_benchmark")
        }
        update_job(
            job_id,
            status=JobStatus.done,
            result=json_safe(summary),
            html_report_url=f"/reports/{html_path.name}",
            sidecar_json_url=f"/reports/{snapshot_path.name}",
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job status, not a crash
        update_job(job_id, status=JobStatus.error, error=str(exc))


@router.post("/batch", response_model=JobResponse, status_code=202)
def start_batch_benchmark(request: BatchBenchmarkRequest, background_tasks: BackgroundTasks) -> JobResponse:
    dataset_path = Path(request.dataset_dir)
    if not dataset_path.is_dir():
        raise HTTPException(status_code=400, detail=f"dataset_dir not found: {dataset_path}")
    job_id = create_job()
    background_tasks.add_task(
        _execute_batch_benchmark, job_id, request.dataset_dir, request.model_name, request.confidence_threshold
    )
    return JobResponse(**get_job(job_id))


@router.get("/batch/{job_id}", response_model=JobResponse)
def get_batch_benchmark(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse(**job)
