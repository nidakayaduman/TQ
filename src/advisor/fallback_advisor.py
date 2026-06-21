"""Deterministic safe advisor."""

from __future__ import annotations

from typing import Any


def build_fallback_advisor(context: dict[str, Any]) -> dict[str, Any]:
    profile_score = float(context.get("won_profile_fit_score", 0))
    price_score = float(context.get("price_band_fit_score", 0))
    margin_score = float(context.get("margin_score", 0))
    confidence_score = float(context.get("model_confidence_score", 0))
    risk_flags = list(context.get("risk_flags", []))
    similar_count = int(context.get("similar_tender_count", 0))
    manual_review = confidence_score < 50 or profile_score < 45 or bool(risk_flags)
    corridor = context.get("corridor", {})
    low = corridor.get("predicted_low_price")
    mid = corridor.get("predicted_mid_price")
    high = corridor.get("predicted_high_price")
    evidence_items = context.get("evidence_items") or [
        {"evidence_id": "E_PROFILE_001", "type": "profile_fit", "content": f"Profil uyumu {profile_score:.1f}/100."},
        {"evidence_id": "E_PRICE_001", "type": "price_band", "content": f"Fiyat koridoru düşük {low}, orta {mid}, yüksek {high}."},
        {"evidence_id": "E_RISK_001", "type": "risk", "content": f"Risk bayrakları: {', '.join(risk_flags) if risk_flags else 'yok'}."},
        {"evidence_id": "E_CONF_001", "type": "confidence", "content": f"Model güveni {confidence_score:.1f}/100."},
    ]
    evidence_used = [
        {
            "evidence_id": str(item.get("evidence_id")),
            "claim": str(item.get("content") or item.get("claim") or item.get("type") or "Kanıt kullanıldı."),
        }
        for item in evidence_items
        if isinstance(item, dict) and item.get("evidence_id")
    ]
    summary = (
        f"Seçili ihale {profile_score:.1f}/100 profil uyumu ve {confidence_score:.1f}/100 veri güveniyle "
        f"orta düzey karar desteği üretiyor. Fiyat bandı uyumu {price_score:.1f}/100; "
        f"{'manuel inceleme önerilir' if manual_review else 'mevcut çıktılar teklif çalışması için kullanılabilir'}."
    )
    action = (
        "Fiyatı dengeli koridor çevresinde tutup risk bayraklarını manuel kontrol edin."
        if manual_review
        else "Dengeli fiyat seviyesini ana referans alıp teklif senaryolarını karlılık hedefiyle karşılaştırın."
    )
    risks = risk_flags[:4] if risk_flags else ["Belirgin risk bayrağı yok."]
    checks = [
        "Düşük, dengeli ve yüksek fiyat senaryolarını karlılık hedefiyle karşılaştırın.",
        "Benzer ihale listesindeki ürün grubu, bölge ve miktar eşleşmelerini kontrol edin.",
        "Sıra dışılık sinyali varsa fiyat ve teslim koşullarını manuel inceleyin.",
    ]
    return {
        "executive_summary": summary,
        "recommended_action": action,
        "scenario_rationale": (
            f"Senaryo yorumu profil uyumu {profile_score:.1f}, fiyat bandı uyumu {price_score:.1f}, "
            f"karlılık skoru {margin_score:.1f} ve model güveni {confidence_score:.1f} bileşenlerine dayanır."
        ),
        "evidence_used": evidence_used[:4],
        "risk_warnings": risks,
        "human_checks_required": checks,
        "confidence_rationale": f"Model güveni {confidence_score:.1f}/100; benzer ihale sayısı {similar_count}.",
        "limitations": [
            "Bu çıktı gerçek kazanma olasılığı değildir.",
            "Kaybedilmiş ihale verisi olmadığı için kazanma/kaybetme sınıflandırması yapılmaz.",
            "Rakip fiyatları tahmin edilmez; sadece mevcut veriyle karar desteği sağlanır.",
        ],
        "forbidden_claims_check": {
            "claims_true_win_chance": False,
            "claims_guaranteed_win": False,
        },
    }
