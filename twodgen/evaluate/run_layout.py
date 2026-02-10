from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

from twodgen.common.run_metadata import collect_run_metadata


RUN_LAYOUT_SCHEMA_VERSION = "run_layout_v1"
RUN_METADATA_SCHEMA_VERSION = "run_metadata_v1"
METRICS_SUMMARY_SCHEMA_VERSION = "metrics_summary_v1"
FAILURE_BREAKDOWN_SCHEMA_VERSION = "failure_breakdown_v1"
PROJECTION_STATS_SCHEMA_VERSION = "projection_stats_v1"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_dir: Path
    plots_dir: Path
    samples_dir: Path
    run_metadata: Path
    projection_stats: Path
    metrics_summary: Path
    failure_breakdown: Path
    status: Path
    error_trace: Path


def utc_timestamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_token(value: str) -> str:
    keep = []
    for ch in value.strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    text = "".join(keep).strip("_")
    return text or "run"


def config_hash(config_obj: Any) -> str:
    payload = json.dumps(config_obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def make_run_paths(
    *,
    experiment_id: str,
    runs_root: Path = Path("runs"),
    run_name: Optional[str] = None,
) -> RunPaths:
    exp = _safe_token(experiment_id)
    run = _safe_token(run_name) if run_name else utc_timestamp_compact()
    run_dir = Path(runs_root) / exp / run
    return RunPaths(
        root=Path(runs_root),
        run_dir=run_dir,
        plots_dir=run_dir / "plots",
        samples_dir=run_dir / "samples",
        run_metadata=run_dir / "run_metadata.json",
        projection_stats=run_dir / "projection_stats.json",
        metrics_summary=run_dir / "metrics_summary.json",
        failure_breakdown=run_dir / "failure_breakdown.json",
        status=run_dir / "STATUS.json",
        error_trace=run_dir / "error_trace.txt",
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    paths.samples_dir.mkdir(parents=True, exist_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
    _atomic_write_bytes(path, data)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def make_schema_payload(
    *,
    schema_version: str,
    payload: Dict[str, Any],
    experiment_id: Optional[str],
    seed: Optional[int],
    protocol: Optional[str],
    config_hash_value: Optional[str],
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rm = run_metadata if run_metadata is not None else collect_run_metadata()
    git_commit = None
    if isinstance(rm.get("git"), dict):
        git_commit = rm["git"].get("commit")
    merged = dict(payload)
    merged.setdefault("schema_version", schema_version)
    merged.setdefault("git_commit", git_commit)
    merged.setdefault("timestamp", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    merged.setdefault("experiment_id", experiment_id)
    merged.setdefault("config_hash", config_hash_value)
    merged.setdefault("seed", seed)
    merged.setdefault("protocol", protocol)
    return merged


def write_status(
    paths: RunPaths,
    *,
    status: str,
    experiment_id: Optional[str],
    seed: Optional[int],
    protocol: Optional[str],
    config_hash_value: Optional[str],
    note: Optional[str] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {"status": status}
    if note:
        payload["note"] = note
    wrapped = make_schema_payload(
        schema_version=RUN_LAYOUT_SCHEMA_VERSION,
        payload=payload,
        experiment_id=experiment_id,
        seed=seed,
        protocol=protocol,
        config_hash_value=config_hash_value,
        run_metadata=run_metadata,
    )
    atomic_write_json(paths.status, wrapped)


def write_error_trace(paths: RunPaths, exc: BaseException) -> None:
    text = f"{type(exc).__name__}: {exc}\n"
    atomic_write_text(paths.error_trace, text)

