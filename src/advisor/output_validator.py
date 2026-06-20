"""Advisor output schema and safety validation."""

from __future__ import annotations

from typing import Any

from ..constants import DISCLAIMER
from .forbidden_claim_detector import detect_forbidden_claims

REQUIRED_ADVISOR_FIELDS = [
    "summary",
    "profile_fit_explanation",
    "price_corridor_explanation",
    "margin_explanation",
    "risk_explanation",
    "confidence_explanation",
    "similar_tenders_summary",
    "manual_review_required",
    "forbidden_claims_detected",
    "disclaimer",
]


def validate_advisor_output(output: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_ADVISOR_FIELDS if field not in output]
    combined_text = " ".join(str(value) for value in output.values())
    forbidden = detect_forbidden_claims(combined_text)
    disclaimer_ok = output.get("disclaimer") == DISCLAIMER
    valid = not missing and not forbidden["forbidden_claims_detected"] and disclaimer_ok
    return {
        "valid": valid,
        "missing_fields": missing,
        "forbidden": forbidden,
        "disclaimer_ok": disclaimer_ok,
        "advisor_validation_status": "pass" if valid else "fail",
    }

