"""Advisor output schema and safety validation."""

from __future__ import annotations

from typing import Any

from ..constants import DISCLAIMER
from .forbidden_claim_detector import detect_forbidden_claims

REQUIRED_ADVISOR_FIELDS = [
    "executive_summary",
    "recommended_action",
    "scenario_rationale",
    "evidence_used",
    "risk_warnings",
    "human_checks_required",
    "forbidden_claims_check",
    "confidence_rationale",
    "limitations",
    "decision_summary",
    "data_situation",
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
    "executive_summary": "Danışman yanıtı güvenlik kontrolünden geçmediği için deterministik güvenli yanıt kullanılmalı.",
    "data_situation": "Yorum yalnızca mevcut model çıktılarıyla sınırlıdır.",
    "recommended_action": "Emsal, fiyat koridoru, profil uyumu ve risk bayraklarını manuel kontrol edin.",
    "scenario_rationale": "Yalnızca mevcut senaryo skorları ve risk bayrakları yorumlanır.",
    "evidence_used": ["E_PROFILE_001", "E_PRICE_001", "E_RISK_001"],
    "risk_warnings": ["Güvenlik kontrolü nedeniyle serbest metin yanıtı engellendi."],
    "human_checks_required": ["Yapılandırılmış rapor ekranındaki metrikleri inceleyin."],
    "forbidden_claims_check": False,
    "confidence_rationale": "Yanıt güvenli fallback ile üretildi.",
    "limitations": "Gerçek kazanma olasılığı, rakip davranışı veya reveal edilmemiş gerçek sonuç verilmez.",
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
    combined_text = " ".join(str(value) for key, value in output.items() if key != "disclaimer")
    forbidden = detect_forbidden_claims(combined_text)
    disclaimer_ok = output.get("disclaimer") == DISCLAIMER
    valid = not missing and not forbidden["forbidden_claims_detected"] and disclaimer_ok
    schema_valid = not missing and isinstance(output.get("evidence_used"), list)
    return {
        "valid": valid,
        "schema_valid": schema_valid,
        "missing_fields": missing,
        "forbidden": forbidden,
        "forbidden_claims_detected": forbidden["forbidden_claims_detected"],
        "disclaimer_ok": disclaimer_ok,
        "advisor_validation_status": "pass" if valid else "fail",
        "llm_validation_status": "pass" if valid else "fail",
        "grounding_score": 1.0 if valid else 0.0,
        "prompt_injection_detected": False,
        "fallback_used": False,
    }
