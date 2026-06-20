"""Structured audit log writer."""

from __future__ import annotations

import json
import os
import uuid
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config_loader import load_observability_config
from ..model_version import MODEL_VERSION
from .structured_logging import log_event, model_version_string

PROCESS_SESSION_ID = os.getenv("SESSION_ID", f"session-{uuid.uuid4().hex[:12]}")

AUDIT_LOG_SCHEMA = {
    "session_id": "Session identifier",
    "user_id": "User identifier or anonymous",
    "event_type": "Audit event type",
    "tender_id": "Selected tender identifier when applicable",
    "scenario_id": "Scenario identifier when applicable",
    "reveal_status": "hidden/revealed/revealed_for_backtest",
    "timestamp": "UTC ISO-8601 event creation time",
    "input_hash": "SHA-256 hash of non-sensitive input summary/payload",
    "output_hash": "SHA-256 hash of non-sensitive output summary/payload",
    "model_version": "Active model/artifact version",
    "config_version": "Active config version",
}


def _stable_hash(value: Any) -> str:
    if value in (None, ""):
        return ""
    normalized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(normalized.encode("utf-8")).hexdigest()


def normalize_audit_event(event: dict[str, Any]) -> dict[str, Any]:
    observability = load_observability_config().get("audit", {})
    input_value = event.get("input_payload", event.get("input_summary", ""))
    output_value = event.get("output_payload", event.get("output_summary", ""))
    normalized = {
        "session_id": event.get("session_id", PROCESS_SESSION_ID),
        "user_id": event.get("user_id", os.getenv("USER_ID", observability.get("default_user_id", "anonymous"))),
        "event_type": event.get("event_type", event.get("user_action", "unknown")),
        "user_action": event.get("user_action", event.get("event_type", "unknown")),
        "tender_id": event.get("tender_id", ""),
        "scenario_id": event.get("scenario_id", ""),
        "reveal_status": event.get("reveal_status", "hidden"),
        "module": event.get("module", "app"),
        "input_summary": event.get("input_summary", ""),
        "output_summary": event.get("output_summary", ""),
        "input_hash": event.get("input_hash", _stable_hash(input_value)),
        "output_hash": event.get("output_hash", _stable_hash(output_value)),
        "validation_status": event.get("validation_status", "unknown"),
        "leakage_status": event.get("leakage_status", "unknown"),
        "advisor_guardrail_status": event.get("advisor_guardrail_status", "not_applicable"),
        "config_version": event.get("config_version", MODEL_VERSION.get("config_version", "config-v1")),
        "model_version": event.get("model_version", model_version_string()),
    }
    return {**normalized, **event}


def write_audit_event(event: dict[str, Any], audit_dir: str | Path | None = None) -> Path:
    observability = load_observability_config().get("audit", {})
    directory = Path(audit_dir or str(observability.get("directory", "audit_logs")))
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    file_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_name = str(event.get("event_type", "event")).replace("/", "_")
    path = directory / f"{file_timestamp}_{event_name}_{uuid.uuid4().hex[:8]}.json"
    payload = {"created_at_utc": timestamp, "timestamp": timestamp, **normalize_audit_event(event)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event(
        payload["event_type"],
        module=payload.get("module", "audit"),
        status=payload.get("validation_status", "unknown"),
        message=f"audit event written: {payload['event_type']}",
        tender_id=str(payload.get("tender_id") or "") or None,
        scenario_id=str(payload.get("scenario_id") or "") or None,
        audit_path=str(path),
        reveal_status=payload.get("reveal_status"),
    )
    return path
