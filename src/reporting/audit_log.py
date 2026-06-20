"""Structured audit log writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_LOG_SCHEMA = {
    "timestamp": "UTC ISO-8601 event creation time",
    "user_action": "User-visible action or system event",
    "tender_id": "Selected tender identifier when applicable",
    "module": "Application module",
    "input_summary": "Short non-sensitive input summary",
    "output_summary": "Short non-sensitive output summary",
    "validation_status": "pass/fail/blocked/unknown",
    "leakage_status": "pass/fail/unknown",
    "advisor_guardrail_status": "pass/fail/blocked/not_applicable",
    "config_version": "Active config version or file summary",
    "model_version": "Active model/artifact version",
}


def normalize_audit_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "user_action": event.get("user_action", event.get("event_type", "unknown")),
        "tender_id": event.get("tender_id", ""),
        "module": event.get("module", "app"),
        "input_summary": event.get("input_summary", ""),
        "output_summary": event.get("output_summary", ""),
        "validation_status": event.get("validation_status", "unknown"),
        "leakage_status": event.get("leakage_status", "unknown"),
        "advisor_guardrail_status": event.get("advisor_guardrail_status", "not_applicable"),
        "config_version": event.get("config_version", "config/app_config.yaml"),
        "model_version": event.get("model_version", "retrieval:v1;kmeans:v1;isolation_forest:v1;baseline:v1"),
    }
    return {**normalized, **event}


def write_audit_event(event: dict[str, Any], audit_dir: str | Path = "audit_logs") -> Path:
    directory = Path(audit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_name = str(event.get("event_type", "event")).replace("/", "_")
    path = directory / f"{timestamp}_{event_name}.json"
    payload = {"created_at_utc": timestamp, "timestamp": timestamp, **normalize_audit_event(event)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
