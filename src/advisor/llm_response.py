"""Normalize raw LLM responses into the advisor output contract."""

from __future__ import annotations

import json
import re
from typing import Any

from .forbidden_claim_detector import detect_forbidden_claims

FORBIDDEN_CLAIMS_CHECK_FIELD = "forbidden_claims_check"
RISK_LABELS = {
    "low_similarity": "Emsal ihale benzerliği düşük; geçmiş kazanılmış örneklerle yakınlık zayıf olduğu için model güveni azalır.",
    "wide_price_band": "Fiyat koridoru geniş; olası fiyat aralığı büyüdüğü için karar desteği daha dikkatli yorumlanmalıdır.",
    "medium_model_disagreement": "Farklı fiyat modelleri arasında belirgin ayrışma var; fiyat önerisi manuel kontrol edilmelidir.",
    "high_model_disagreement": "Fiyat modelleri arasında yüksek ayrışma var; fiyat kararı güçlü manuel inceleme gerektirir.",
    "low_margin": "Beklenen karlılık düşük; maliyet ve marj varsayımları kontrol edilmelidir.",
}
ADVISOR_TEXT_FIELDS = [
    "executive_summary",
    "recommended_action",
    "scenario_rationale",
    "confidence_rationale",
    "summary",
    "decision_summary",
    "answer",
    "response",
    "rationale",
    "analysis",
]


def _extract_json_string_fields(content: str) -> dict[str, str] | None:
    extracted: dict[str, str] = {}
    for field in ADVISOR_TEXT_FIELDS:
        match = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"', content, flags=re.DOTALL)
        if match:
            try:
                extracted[field] = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                extracted[field] = match.group(1)
    return extracted or None


def normalize_llm_payload(content: str) -> dict[str, Any] | None:
    cleaned = str(content or "").strip()
    cleaned = cleaned.removeprefix("\ufeff").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.casefold().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return _extract_json_string_fields(candidate)
    if isinstance(parsed, dict) and "choices" in parsed:
        try:
            nested_content = parsed["choices"][0]["message"]["content"]
            return normalize_llm_payload(str(nested_content))
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def safe_text(value: Any, max_length: int = 1200) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\u3400-\u9fff]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:max_length]


def clean_risk_warning(value: Any) -> str:
    text = safe_text(value, 360)
    return RISK_LABELS.get(text, text)


def default_evidence(context: dict[str, Any], claim_prefix: str = "Model çıktısı yorumlandı.") -> list[dict[str, str]]:
    evidence_items = context.get("evidence_items", [])
    if not isinstance(evidence_items, list):
        evidence_items = []
    defaults: list[dict[str, str]] = []
    for item in evidence_items[:3]:
        if isinstance(item, dict) and item.get("evidence_id"):
            defaults.append(
                {
                    "evidence_id": str(item["evidence_id"]),
                    "claim": safe_text(item.get("content") or claim_prefix, 240),
                }
            )
    return defaults or [
        {"evidence_id": "E_PROFILE_001", "claim": "Profil uyumu kontrol edildi."},
        {"evidence_id": "E_PRICE_001", "claim": "Fiyat koridoru kontrol edildi."},
        {"evidence_id": "E_RISK_001", "claim": "Risk bayrakları kontrol edildi."},
    ]


def normalize_advisor_payload_schema(payload: dict[str, Any], context: dict[str, Any], question: str) -> dict[str, Any]:
    """Coerce model JSON into the strict advisor schema before safety validation."""
    allowed_ids = {item["evidence_id"] for item in default_evidence(context)}
    summary = (
        payload.get("executive_summary")
        or payload.get("summary")
        or payload.get("decision_summary")
        or payload.get("answer")
        or payload.get("response")
        or ""
    )
    rationale = payload.get("scenario_rationale") or payload.get("rationale") or payload.get("analysis") or summary
    action = payload.get("recommended_action") or payload.get("action") or "Model çıktıları karar komitesi tarafından manuel kontrol edilmeli."
    evidence_used = payload.get("evidence_used")
    normalized_evidence: list[dict[str, str]] = []
    if isinstance(evidence_used, list):
        for item in evidence_used:
            if isinstance(item, dict):
                evidence_id = str(item.get("evidence_id", "")).strip()
                claim = safe_text(item.get("claim") or item.get("content") or item.get("text"), 360)
            else:
                evidence_id = str(item).strip()
                claim = "Model bağlamındaki kanıt kullanıldı."
            if evidence_id in allowed_ids:
                normalized_evidence.append({"evidence_id": evidence_id, "claim": claim or "Model bağlamındaki kanıt kullanıldı."})
    if not normalized_evidence:
        normalized_evidence = default_evidence(context)

    risk_warnings = payload.get("risk_warnings") or payload.get("risks") or payload.get("risk_flags") or []
    if not isinstance(risk_warnings, list):
        risk_warnings = [risk_warnings]
    risk_warnings = [clean_risk_warning(item) for item in risk_warnings if str(item).strip()][:4]
    if not risk_warnings:
        risk_flags = context.get("risk_flags", [])
        risk_warnings = [clean_risk_warning(item) for item in risk_flags[:4]] if isinstance(risk_flags, list) else []
    if not risk_warnings:
        risk_warnings = ["Belirgin ek risk uyarısı yok; yine de maliyet ve teslim varsayımları kontrol edilmeli."]

    human_checks = payload.get("human_checks_required") or payload.get("manual_checks") or payload.get("human_checks") or []
    if not isinstance(human_checks, list):
        human_checks = [human_checks]
    human_checks = [safe_text(item, 360) for item in human_checks if str(item).strip()][:4]
    if not human_checks:
        human_checks = ["Fiyat, maliyet, teslim ve stok varsayımları teklif komitesi tarafından kontrol edilmeli."]

    limitations = payload.get("limitations") or []
    if not isinstance(limitations, list):
        limitations = [limitations]
    limitations = [safe_text(item, 360) for item in limitations if str(item).strip()][:4]
    required_limits = [
        "Bu çıktı gerçek kazanma olasılığı değildir.",
        "Kaybedilmiş ihale verisi olmadığı için kazanma/kaybetme sınıflandırması yapılmaz.",
        "Rakip fiyatları tahmin edilmez; sadece mevcut veriyle karar desteği sağlanır.",
    ]
    for item in required_limits:
        if item not in limitations:
            limitations.append(item)

    forbidden_flags = payload.get(FORBIDDEN_CLAIMS_CHECK_FIELD)
    if not isinstance(forbidden_flags, dict):
        forbidden_flags = {}

    return {
        "executive_summary": safe_text(summary or rationale or "OpenRouter yanıtı yapılandırılmış danışman formatına dönüştürüldü.", 1200),
        "recommended_action": safe_text(action, 1200),
        "scenario_rationale": safe_text(rationale or summary, 1800),
        "evidence_used": normalized_evidence,
        "risk_warnings": risk_warnings,
        "human_checks_required": human_checks,
        "confidence_rationale": safe_text(
            payload.get("confidence_rationale")
            or payload.get("confidence")
            or f"Yanıt, seçili ihale bağlamı ve kullanıcı sorusu temel alınarak yorumlandı: {question}",
            1200,
        ),
        "limitations": limitations[:5],
        FORBIDDEN_CLAIMS_CHECK_FIELD: {
            "claims_true_win_chance": bool(forbidden_flags.get("claims_true_win_chance", False)),
            "claims_guaranteed_win": bool(forbidden_flags.get("claims_guaranteed_win", False)),
        },
    }


def payload_from_free_text(content: str, context: dict[str, Any], question: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    parsed = normalize_llm_payload(text)
    if parsed:
        return normalize_advisor_payload_schema(parsed, context, question)
    if text.lstrip().startswith("{"):
        return None
    if detect_forbidden_claims(text)["forbidden_claims_detected"]:
        return None
    return normalize_advisor_payload_schema(
        {
            "executive_summary": safe_text(text, 1200),
            "recommended_action": "Bu serbest metin yanıtı güvenli şemaya dönüştürüldü; teklif kararı öncesi metrikler manuel kontrol edilmeli.",
            "scenario_rationale": safe_text(text, 1800),
        },
        context,
        question,
    )
