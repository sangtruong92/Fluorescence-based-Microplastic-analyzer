"""Loads YOLO model weights once per process and keeps the active selection.

Avoids reloading a multi-hundred-MB checkpoint on every request; ultralytics/
torch model objects are not guaranteed thread-safe for concurrent inference,
so callers share one instance and the API is expected to run with a single
worker for now (see docs/development_changelog.md "Open items").
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.analysis.ml_benchmark_analyzer import MLBenchmarkAnalyzer

MODELS_DIR = Path(__file__).resolve().parent.parent / "src" / "ml"


def discover_models() -> Dict[str, Path]:
    """Return {folder_name: best.pt path} for every src/ml/*/best.pt found."""
    if not MODELS_DIR.exists():
        return {}
    return {
        child.name: child / "best.pt"
        for child in sorted(MODELS_DIR.iterdir())
        if child.is_dir() and (child / "best.pt").exists()
    }


class ModelRegistry:
    def __init__(self) -> None:
        self._models = discover_models()
        self._active_name: Optional[str] = None
        self._yolo_model = None
        self._analyzer: Optional[MLBenchmarkAnalyzer] = None

    @property
    def available(self) -> Dict[str, Path]:
        return self._models

    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    def load(self, name: Optional[str] = None) -> MLBenchmarkAnalyzer:
        """Load (or reuse) a model by folder name; defaults to yolo26m_v2 if present."""
        if name is None:
            name = "yolo26m_v2" if "yolo26m_v2" in self._models else next(iter(self._models), None)
        if name is None:
            raise FileNotFoundError(f"no YOLO weights found under {MODELS_DIR}")
        if name not in self._models:
            raise FileNotFoundError(f"unknown model {name!r}; available: {list(self._models)}")
        if self._analyzer is not None and self._active_name == name:
            return self._analyzer

        from ultralytics import YOLO  # deferred: optional heavy dependency

        self._yolo_model = YOLO(str(self._models[name]))
        self._analyzer = MLBenchmarkAnalyzer(self._yolo_model)
        self._active_name = name
        return self._analyzer

    @property
    def analyzer(self) -> Optional[MLBenchmarkAnalyzer]:
        return self._analyzer


registry = ModelRegistry()
