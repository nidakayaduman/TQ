"""Validate that advisor output is grounded in deterministic fields."""

from __future__ import annotations

from typing import Any


def validate_grounding(output: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(value) for value in output.values()).casefold()
    unsupported: list[str] = []
    if "rakip kesin" in text or "competitor will" in text:
        unsupported.append("Unsupported competitor behavior claim.")
    if "actual_won_unit_price" in text and not context.get("revealed", False):
        unsupported.append("Hidden actual result mentioned before reveal.")
    return {
        "grounded": not unsupported,
        "unsupported_claims": unsupported,
        "grounding_validation_status": "pass" if not unsupported else "fail",
    }

