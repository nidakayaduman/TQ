"""Structured JSON logging helpers for audit-friendly application events."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config_loader import load_observability_config
from ..model_version import MODEL_VERSION

LOGGER_NAME = "tender_iq"
_LOGGER_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": getattr(record, "event_type", record.getMessage()),
            "module": getattr(record, "module_name", record.name),
            "status": getattr(record, "status", "info"),
            "message": record.getMessage(),
            "tender_id": getattr(record, "tender_id", None),
            "scenario_id": getattr(record, "scenario_id", None),
            "config_version": getattr(record, "config_version", MODEL_VERSION.get("config_version")),
            "model_version": getattr(record, "model_version", model_version_string()),
        }
        extra = getattr(record, "event_payload", None)
        if isinstance(extra, dict):
            payload.update({key: value for key, value in extra.items() if value is not None})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def model_version_string() -> str:
    return ";".join(
        str(MODEL_VERSION.get(key, ""))
        for key in [
            "retrieval_model_version",
            "profile_cluster_model_version",
            "isolation_forest_model_version",
            "baseline_model_version",
        ]
    )


def configure_json_logging() -> logging.Logger:
    global _LOGGER_CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if _LOGGER_CONFIGURED or logger.handlers:
        _LOGGER_CONFIGURED = True
        return logger

    config = load_observability_config().get("logging", {})
    level = str(os.getenv("LOG_LEVEL", config.get("level", "INFO"))).upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if bool(config.get("enabled", True)):
        log_dir = Path(str(config.get("directory", "logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / str(config.get("application_log", "app.jsonl")), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_CONFIGURED = True
    return logger


def log_event(
    event_type: str,
    *,
    module: str = "app",
    status: str = "info",
    message: str = "",
    tender_id: str | None = None,
    scenario_id: str | None = None,
    **payload: Any,
) -> None:
    logger = configure_json_logging()
    logger.info(
        message or event_type,
        extra={
            "event_type": event_type,
            "module_name": module,
            "status": status,
            "tender_id": tender_id,
            "scenario_id": scenario_id,
            "event_payload": payload,
        },
    )


def log_exception(
    event_type: str,
    *,
    module: str = "app",
    status: str = "error",
    message: str = "",
    tender_id: str | None = None,
    scenario_id: str | None = None,
    **payload: Any,
) -> None:
    logger = configure_json_logging()
    logger.exception(
        message or event_type,
        extra={
            "event_type": event_type,
            "module_name": module,
            "status": status,
            "tender_id": tender_id,
            "scenario_id": scenario_id,
            "event_payload": payload,
        },
    )


def recent_log_events(limit: int = 5) -> list[dict[str, Any]]:
    config = load_observability_config().get("logging", {})
    path = Path(str(config.get("directory", "logs"))) / str(config.get("application_log", "app.jsonl"))
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
