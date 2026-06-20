"""Validate that advisor output stays supported by known model context."""

from __future__ import annotations

from typing import Any

from .grounding_validator import validate_grounding


def validate_supported_claims(output: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    grounding = validate_grounding(output, context)
    text = " ".join(str(value) for value in output.values()).casefold()
    unsupported = list(grounding["unsupported_claims"])
    if "rakip" in text and "tahmin" in text:
        unsupported.append("Veri dışı rakip tahmini.")
    if "uydurdum" in text or "varsaydım" in text:
        unsupported.append("Veri dışı varsayım.")
    if not context.get("revealed", False) and ("gerçek kazanılmış fiyat" in text or "gerçek karlılık" in text):
        unsupported.append("Reveal öncesi gerçek sonuç yorumu.")
    return {
        "supported": not unsupported,
        "unsupported_claims": sorted(set(unsupported)),
        "support_validation_status": "pass" if not unsupported else "fail",
    }
