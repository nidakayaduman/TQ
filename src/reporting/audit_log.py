"""Structured audit log writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_audit_event(event: dict[str, Any], audit_dir: str | Path = "audit_logs") -> Path:
    directory = Path(audit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_name = str(event.get("event_type", "event")).replace("/", "_")
    path = directory / f"{timestamp}_{event_name}.json"
    payload = {"created_at_utc": timestamp, **event}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

