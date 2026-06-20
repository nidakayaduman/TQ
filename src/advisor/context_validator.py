"""Advisor context validation and reveal-aware sanitization."""

from __future__ import annotations

from typing import Any

from ..constants import ACTUAL_RESULT_FIELDS

REQUIRED_CONTEXT_KEYS = [
    "tender_id",
    "similar_tenders",
    "won_profile_fit_score",
    "cluster_name",
    "isolation_forest",
    "corridor",
    "scenario_score",
    "leakage_audit",
]


def sanitize_advisor_context(context: dict[str, Any]) -> dict[str, Any]:
    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items() if key not in ACTUAL_RESULT_FIELDS}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    safe = scrub(dict(context))
    if not safe.get("revealed", False):
        safe.pop("revealed_actual", None)
        for field in ACTUAL_RESULT_FIELDS:
            safe.pop(field, None)
    return safe


def validate_advisor_context(context: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_CONTEXT_KEYS if key not in context]
    hidden_present = []
    if not context.get("revealed", False):
        hidden_present = [field for field in ACTUAL_RESULT_FIELDS if field in context]
        if "revealed_actual" in context:
            hidden_present.append("revealed_actual")
    return {
        "context_valid": not missing and not hidden_present,
        "missing_keys": missing,
        "hidden_actual_fields_present": hidden_present,
        "context_validation_status": "pass" if not missing and not hidden_present else "fail",
    }
