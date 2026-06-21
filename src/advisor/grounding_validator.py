"""Validate that advisor output is grounded in deterministic fields."""

from __future__ import annotations

from typing import Any


def _evidence_ids(context: dict[str, Any]) -> set[str]:
    items = context.get("evidence_items", [])
    if not isinstance(items, list):
        return set()
    return {str(item.get("evidence_id")) for item in items if isinstance(item, dict) and item.get("evidence_id")}


def validate_grounding(output: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(value) for value in output.values()).casefold()
    unsupported: list[str] = []
    available_ids = _evidence_ids(context)
    used = output.get("evidence_used", [])
    used_ids = set()
    if isinstance(used, list):
        for item in used:
            if isinstance(item, dict) and item.get("evidence_id"):
                used_ids.add(str(item["evidence_id"]))
            elif isinstance(item, str):
                used_ids.add(item)
    if available_ids:
        if not used_ids:
            unsupported.append("Evidence ID bulunmuyor.")
        missing_ids = sorted(used_ids - available_ids)
        if missing_ids:
            unsupported.append(f"Context içinde olmayan evidence ID kullanıldı: {', '.join(missing_ids)}")
    if "rakip kesin" in text or "competitor will" in text:
        unsupported.append("Unsupported competitor behavior claim.")
    if "actual_won_unit_price" in text and not context.get("revealed", False):
        unsupported.append("Hidden actual result mentioned before reveal.")
    grounding_score = 1.0
    if available_ids:
        grounding_score = len(used_ids & available_ids) / max(len(used_ids), 1)
    if unsupported:
        grounding_score = min(grounding_score, 0.49)
    return {
        "grounded": not unsupported,
        "unsupported_claims": unsupported,
        "grounding_score": float(grounding_score),
        "grounding_validation_status": "pass" if not unsupported else "fail",
    }
