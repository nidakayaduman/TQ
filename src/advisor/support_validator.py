"""Validate that advisor output stays supported by known model context."""

from __future__ import annotations

from typing import Any

from .grounding_validator import validate_grounding


def validate_supported_claims(output: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    grounding = validate_grounding(output, context)
    text = " ".join(str(value) for value in output.values()).casefold()
    unsupported = list(grounding["unsupported_claims"])
    safe_competitor_limit = any(
        phrase in text
        for phrase in [
            "rakip fiyatları tahmin edilmez",
            "rakip fiyatı tahmin edilmez",
            "rakip bazlı kazanma tahmini yapılmaz",
            "rakip tahmini yapılmaz",
            "rakip davranışı tahmin edilmez",
        ]
    )
    explicit_competitor_prediction = any(
        phrase in text
        for phrase in [
            "rakiplerin",
            "rakipler",
            "rakibin",
            "rakip kurum",
            "rakip firma",
        ]
    ) and any(
        phrase in text
        for phrase in [
            "tahmin ediyorum",
            "tahmin ediyor",
            "tahmin eder",
            "tahmin edilebilir",
            "fiyat vereceğini",
            "fiyat verir",
        ]
    )
    if explicit_competitor_prediction or ("rakip" in text and "tahmin" in text and not safe_competitor_limit):
        unsupported.append("Veri dışı rakip tahmini.")
    if "uydurdum" in text or "varsaydım" in text:
        unsupported.append("Veri dışı varsayım.")
    safe_reveal_limit = any(
        phrase in text
        for phrase in [
            "gizli gerçek fiyat",
            "gerçek kazanılmış fiyat kullanılmaz",
            "gerçek kazanılmış fiyat kullanılmadı",
            "gerçek karlılık kullanılmaz",
            "gerçek karlılık kullanılmadı",
            "reveal edilmemiş gerçek sonuçlar kullanılmaz",
        ]
    )
    if not context.get("revealed", False) and ("gerçek kazanılmış fiyat" in text or "gerçek karlılık" in text) and not safe_reveal_limit:
        unsupported.append("Reveal öncesi gerçek sonuç yorumu.")
    return {
        "supported": not unsupported,
        "unsupported_claims": sorted(set(unsupported)),
        "support_validation_status": "pass" if not unsupported else "fail",
    }
