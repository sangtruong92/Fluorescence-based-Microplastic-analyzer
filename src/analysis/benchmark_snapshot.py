"""Versioned JSON evidence persisted next to benchmark HTML reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


BENCHMARK_SNAPSHOT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert numpy/path/tuple values into deterministic JSON-compatible data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_benchmark_snapshot(
    html_path: Path,
    snapshot: Dict[str, Any],
    *,
    output_path: Optional[Path] = None,
) -> Path:
    """Write an atomic, HTML-linked benchmark sidecar."""
    html_path = Path(html_path).resolve()
    if not html_path.is_file():
        raise FileNotFoundError(f"benchmark HTML does not exist: {html_path}")
    destination = Path(output_path).resolve() if output_path else html_path.with_suffix(".benchmark.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_snapshot_schema_version": BENCHMARK_SNAPSHOT_SCHEMA_VERSION,
        "report": {
            "path": str(html_path),
            "sha256": sha256_file(html_path),
        },
        **snapshot,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    return destination
