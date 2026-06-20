"""Deterministic safe advisor."""

from __future__ import annotations

from typing import Any

from ..constants import DISCLAIMER


def build_fallback_advisor(context: dict[str, Any]) -> dict[str, Any]:
    profile_score = float(context.get("won_profile_fit_score", 0))
    price_score = float(context.get("price_band_fit_score", 0))
    margin_score = float(context.get("margin_score", 0))
    confidence_score = float(context.get("model_confidence_score", 0))
    risk_flags = list(context.get("risk_flags", []))
    cluster_name = context.get("cluster_name", "Bilinmeyen profil")
    similar_count = int(context.get("similar_tender_count", 0))
    manual_review = confidence_score < 50 or profile_score < 45 or bool(risk_flags)
    return {
        "summary": (
            f"Kazanılmış profil uyum skoru {profile_score:.1f}/100. "
            f"Senaryo fiyat bandı uyumu {price_score:.1f}/100, marj skoru {margin_score:.1f}/100."
        ),
        "profile_fit_explanation": (
            f"İhale geçmiş kazanılmış kayıtlar içinde '{cluster_name}' kümesine yakın değerlendirildi."
        ),
        "price_corridor_explanation": (
            "Fiyat koridoru benzer kazanılmış ihalelerin tarihsel fiyat dağılımından türetilmiştir."
        ),
        "margin_explanation": "Marj skoru önerilen fiyat ve tahmini birim maliyet ilişkisinden hesaplanır.",
        "risk_explanation": "Risk bayrakları: " + (", ".join(risk_flags) if risk_flags else "kritik bayrak yok."),
        "confidence_explanation": (
            f"Model güven skoru {confidence_score:.1f}/100; benzer ihale sayısı {similar_count}."
        ),
        "similar_tenders_summary": f"Yorum {similar_count} benzer kazanılmış ihale üzerinden oluşturuldu.",
        "manual_review_required": manual_review,
        "forbidden_claims_detected": False,
        "disclaimer": DISCLAIMER,
    }

