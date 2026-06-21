"""Structured audit log writer."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..config_loader import load_observability_config
from ..constants import ACTUAL_RESULT_FIELDS
from ..model_version import MODEL_VERSION
from ..schema_contracts import load_json_schema, validate_json_schema
from .structured_logging import log_event, model_version_string

PROCESS_SESSION_ID = os.getenv("SESSION_ID", f"session-{uuid.uuid4().hex[:12]}")

AUDIT_LOG_EVENT_JSON_SCHEMA = load_json_schema("audit_log_event.schema.json")
AUDIT_LOG_SCHEMA = {
    field: spec.get("description", spec.get("type", ""))
    for field, spec in AUDIT_LOG_EVENT_JSON_SCHEMA["properties"].items()
}
SENSITIVE_AUDIT_FIELDS = set(ACTUAL_RESULT_FIELDS)
REVEAL_STATUS_ALIASES = {"revealed_for_backtest": "not_applicable", "backtest": "not_applicable"}


def _stable_hash(value: Any) -> str:
    if value in (None, ""):
        return ""
    normalized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(normalized.encode("utf-8")).hexdigest()


def _nullable_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _hash_user_id(user_id: Any) -> str | None:
    if user_id in (None, ""):
        return None
    return _stable_hash(str(user_id))


def _normalize_reveal_status(value: Any) -> str:
    status = str(value or "hidden")
    status = REVEAL_STATUS_ALIASES.get(status, status)
    return status if status in {"hidden", "revealed", "not_applicable"} else "not_applicable"


def _sanitize_details(value: Any, reveal_status: str) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if key in SENSITIVE_AUDIT_FIELDS and reveal_status != "revealed":
                safe[f"{key}_masked"] = True
                safe[f"{key}_hash"] = _stable_hash(item)
                continue
            safe[key] = _sanitize_details(item, reveal_status)
        return safe
    if isinstance(value, list):
        return [_sanitize_details(item, reveal_status) for item in value]
    if isinstance(value, str) and reveal_status != "revealed":
        lowered = value.casefold()
        if any(field.casefold() in lowered for field in SENSITIVE_AUDIT_FIELDS):
            return {"masked": True, "hash": _stable_hash(value)}
    return value


def normalize_audit_event(event: dict[str, Any], timestamp: str | None = None) -> dict[str, Any]:
    observability = load_observability_config().get("audit", {})
    input_value = event.get("input_payload", event.get("input_summary", ""))
    output_value = event.get("output_payload", event.get("output_summary", ""))
    reveal_status = _normalize_reveal_status(event.get("reveal_status", "hidden"))
    user_id = event.get("user_id", os.getenv("USER_ID", observability.get("default_user_id", "anonymous")))
    detail_keys = {
        "user_action",
        "module",
        "input_summary",
        "output_summary",
        "validation_status",
        "leakage_status",
        "advisor_guardrail_status",
        "leakage_audit",
    }
    details = {
        "module": event.get("module", "app"),
        "user_action": event.get("user_action", event.get("event_type", "unknown")),
        "validation_status": event.get("validation_status", "unknown"),
        "leakage_status": event.get("leakage_status", "unknown"),
        "advisor_guardrail_status": event.get("advisor_guardrail_status", "not_applicable"),
    }
    details.update({key: event[key] for key in detail_keys if key in event})
    details.update(event.get("details", {}) if isinstance(event.get("details"), dict) else {})
    return {
        "event_id": str(event.get("event_id") or f"audit-{uuid.uuid4().hex}"),
        "timestamp": timestamp or str(event.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        "event_type": str(event.get("event_type", event.get("user_action", "unknown"))),
        "session_id": str(event.get("session_id", PROCESS_SESSION_ID)),
        "user_id_hash": event.get("user_id_hash", _hash_user_id(user_id)),
        "tender_id": _nullable_string(event.get("tender_id")),
        "scenario_id": _nullable_string(event.get("scenario_id")),
        "reveal_status": reveal_status,
        "input_hash": event.get("input_hash", _stable_hash(input_value)) or None,
        "output_hash": event.get("output_hash", _stable_hash(output_value)) or None,
        "model_version": str(event.get("model_version", model_version_string())),
        "config_version": str(event.get("config_version", MODEL_VERSION.get("config_version", "config-v1"))),
        "details": _sanitize_details(details, reveal_status),
    }


def validate_audit_log_event(event: dict[str, Any]) -> dict[str, Any]:
    errors = validate_json_schema(event, AUDIT_LOG_EVENT_JSON_SCHEMA)
    return {"valid": not errors, "errors": errors}


def write_audit_event(event: dict[str, Any], audit_dir: str | Path | None = None) -> Path:
    observability = load_observability_config().get("audit", {})
    directory = Path(audit_dir or str(observability.get("directory", "audit_logs")))
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    file_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_name = str(event.get("event_type", "event")).replace("/", "_")
    path = directory / f"{file_timestamp}_{event_name}_{uuid.uuid4().hex[:8]}.json"
    payload = normalize_audit_event(event, timestamp=timestamp)
    validation = validate_audit_log_event(payload)
    if not validation["valid"]:
        raise ValueError(f"Invalid audit log event: {validation['errors']}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event(
        payload["event_type"],
        module=payload.get("details", {}).get("module", "audit"),
        status=payload.get("details", {}).get("validation_status", "unknown"),
        message=f"audit event written: {payload['event_type']}",
        tender_id=str(payload.get("tender_id") or "") or None,
        scenario_id=str(payload.get("scenario_id") or "") or None,
        audit_path=str(path),
        reveal_status=payload.get("reveal_status"),
    )
    return path
