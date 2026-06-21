"""Advisor output schema and safety validation."""

from __future__ import annotations

from typing import Any

from ..constants import ACTUAL_RESULT_FIELDS
from ..schema_contracts import load_json_schema, validate_json_schema
from .forbidden_claim_detector import detect_forbidden_claims
from .grounding_validator import validate_grounding

REQUIRED_ADVISOR_FIELDS = [
    "executive_summary",
    "recommended_action",
    "scenario_rationale",
    "evidence_used",
    "risk_warnings",
    "human_checks_required",
    "confidence_rationale",
    "limitations",
    "forbidden_claims_check",
]

ADVISOR_OUTPUT_JSON_SCHEMA = load_json_schema("advisor_output.schema.json")

SAFE_FALLBACK_OUTPUT = {
    "executive_summary": "Danışman yanıtı güvenlik kontrolünden geçmediği için deterministik güvenli yanıt kullanılmalı.",
    "recommended_action": "Emsal, fiyat koridoru, profil uyumu ve risk bayraklarını manuel kontrol edin.",
    "scenario_rationale": "Yalnızca mevcut senaryo skorları ve risk bayrakları yorumlanır.",
    "evidence_used": [
        {"evidence_id": "E_PROFILE_001", "claim": "Profil uyumu kontrol edildi."},
        {"evidence_id": "E_PRICE_001", "claim": "Fiyat koridoru kontrol edildi."},
        {"evidence_id": "E_RISK_001", "claim": "Risk bayrakları kontrol edildi."},
    ],
    "risk_warnings": ["Güvenlik kontrolü nedeniyle serbest metin yanıtı engellendi."],
    "human_checks_required": ["Yapılandırılmış rapor ekranındaki metrikleri inceleyin."],
    "confidence_rationale": "Yanıt güvenli fallback ile üretildi.",
    "limitations": [
        "Bu çıktı gerçek kazanma olasılığı değildir.",
        "Rakip fiyatları tahmin edilmez.",
        "Reveal edilmemiş gerçek sonuçlar kullanılmaz.",
    ],
    "forbidden_claims_check": {
        "claims_true_win_chance": False,
        "claims_guaranteed_win": False,
    },
}


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def advisor_semantic_text(output: dict[str, Any]) -> str:
    """Return only user-facing advisor content for claim detection."""
    return _flatten_text({key: value for key, value in output.items() if key != "forbidden_claims_check"})


def _forbidden_claim_flags(output: dict[str, Any]) -> dict[str, bool]:
    value = output.get("forbidden_claims_check")
    if not isinstance(value, dict):
        return {"claims_true_win_chance": True, "claims_guaranteed_win": True}
    return {
        "claims_true_win_chance": bool(value.get("claims_true_win_chance")),
        "claims_guaranteed_win": bool(value.get("claims_guaranteed_win")),
    }


def _hidden_actual_mentions(output: dict[str, Any], context: dict[str, Any] | None) -> list[str]:
    if context and context.get("revealed", False):
        return []
    text = _flatten_text(output).casefold()
    return sorted(field for field in ACTUAL_RESULT_FIELDS if field.casefold() in text)


def validate_advisor_output(output: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    missing = [field for field in REQUIRED_ADVISOR_FIELDS if field not in output]
    schema_errors = validate_json_schema(output, ADVISOR_OUTPUT_JSON_SCHEMA)
    forbidden = detect_forbidden_claims(advisor_semantic_text(output))
    flags = _forbidden_claim_flags(output)
    forbidden_by_flag = any(flags.values())
    grounding = validate_grounding(output, context or {})
    hidden_actual_fields = _hidden_actual_mentions(output, context)
    schema_valid = not missing and not schema_errors
    valid = (
        schema_valid
        and not forbidden["forbidden_claims_detected"]
        and not forbidden_by_flag
        and grounding["grounded"]
        and not hidden_actual_fields
    )
    return {
        "valid": valid,
        "schema_valid": schema_valid,
        "missing_fields": missing,
        "schema_errors": schema_errors,
        "forbidden": forbidden,
        "forbidden_claims_check": flags,
        "forbidden_claims_detected": forbidden["forbidden_claims_detected"] or forbidden_by_flag,
        "hidden_actual_fields_used": hidden_actual_fields,
        "grounding": grounding,
        "advisor_validation_status": "pass" if valid else "fail",
        "llm_validation_status": "pass" if valid else "fail",
        "grounding_score": grounding["grounding_score"] if valid else min(grounding["grounding_score"], 0.49),
        "prompt_injection_detected": False,
        "fallback_used": False,
    }
