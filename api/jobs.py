"""In-memory job store for long-running batch benchmarks.

Local-testing only: state is lost on process restart and is not shared across
workers. Good enough for "run locally first" per the current requirement;
swap for Redis/DB-backed storage before running multiple server workers.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from api.schemas import JobStatus

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}

# Without this, _jobs grows forever across a long-running server process --
# every finished job (including its full result payload) stays in memory.
_FINISHED_JOB_TTL = timedelta(hours=1)


def _prune_finished_jobs_locked() -> None:
    """Drop done/error jobs older than _FINISHED_JOB_TTL. Caller holds _lock."""
    cutoff = datetime.now(timezone.utc) - _FINISHED_JOB_TTL
    expired = [
        job_id
        for job_id, job in _jobs.items()
        if job["status"] in (JobStatus.done, JobStatus.error)
        and datetime.fromisoformat(job["created_at"]) < cutoff
    ]
    for job_id in expired:
        del _jobs[job_id]


def create_job() -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _prune_finished_jobs_locked()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.pending,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "result": None,
            "html_report_url": None,
            "sidecar_json_url": None,
        }
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
        else:
            logger.warning("update_job called for unknown job_id=%s (fields=%s)", job_id, list(fields))


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None
