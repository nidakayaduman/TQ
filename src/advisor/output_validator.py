"""Advisor output schema and safety validation."""

from __future__ import annotations

from typing import Any

from ..constants import DISCLAIMER
from .forbidden_claim_detector import detect_forbidden_claims

REQUIRED_ADVISOR_FIELDS = [
    "decision_summary",
    "data_situation",
    "recommended_action",
    "pwin_interpretation",
    "pricing_interpretation",
    "margin_risk",
    "learner_signals",
    "supporting_evidence",
    "risks",
    "next_actions",
    "manual_review_required",
    "forbidden_claims_detected",
    "disclaimer",
]

SAFE_FALLBACK_OUTPUT = {
    "decision_summary": "Danışman yanıtı güvenlik kontrolünden geçmediği için deterministik güvenli yanıt kullanılmalı.",
    "data_situation": "Yorum yalnızca mevcut model çıktılarıyla sınırlıdır.",
    "recommended_action": "Emsal, fiyat koridoru, profil uyumu ve risk bayraklarını manuel kontrol edin.",
    "pwin_interpretation": "Bu skor olasılık değil, geçmiş kazanılmış profile uyum göstergesidir.",
    "pricing_interpretation": "Veri dışı fiyat üretilmez.",
    "margin_risk": "Kural ihlali varsa manuel inceleme gerekir.",
    "learner_signals": {},
    "supporting_evidence": [],
    "risks": ["Güvenlik kontrolü nedeniyle serbest metin yanıtı engellendi."],
    "next_actions": ["Yapılandırılmış rapor ekranındaki metrikleri inceleyin."],
    "manual_review_required": True,
    "forbidden_claims_detected": True,
    "disclaimer": DISCLAIMER,
}


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
