"""
Microplastic Analyzer API

Run locally with:  ./.venv/bin/python3 run_api.py
Interactive docs:   http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

# Allow `import src...` / `import config...` the same way the GUI entry point does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Library code logs via logging (not print); default to INFO, override with
# LOG_LEVEL=DEBUG to see the detailed segmentation/calibration traces.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.model_registry import registry
from api.routers import analyze, benchmark, calibration, generate
from api.schemas import HealthResponse, ModelInfo

app = FastAPI(
    title="Microplastic Analyzer API",
    version="0.1.0",
    description="REST API wrapping the Quick/Deep/ML analysis pipeline for a web frontend.",
)

# Default to common local dev server ports (Vite/CRA); adjust before deploying off localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"
REPORTS_DIR.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

app.include_router(analyze.router)
app.include_router(benchmark.router)
app.include_router(generate.router)
app.include_router(calibration.router)


@app.on_event("startup")
def _load_default_model() -> None:
    try:
        registry.load()
    except FileNotFoundError:
        # /api/health reports yolo_loaded=False; /api/analyze(method=ml) returns 503 until a model exists.
        pass


@app.get("/api/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    return [ModelInfo(name=name, path=str(path)) for name, path in registry.available.items()]


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        yolo_loaded=registry.analyzer is not None,
        active_model=registry.active_name,
        available_models=[ModelInfo(name=name, path=str(path)) for name, path in registry.available.items()],
    )
