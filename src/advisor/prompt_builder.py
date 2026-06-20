"""Grounded advisor prompt builder."""

from __future__ import annotations

import json
from typing import Any

from ..constants import DISCLAIMER


def build_advisor_prompt(context: dict[str, Any]) -> str:
    safe_context = dict(context)
    question = safe_context.pop("user_question", "") or "Genel yönetici yorumu üret."
    return f"""Aşağıdaki JSON, bir ihale karar destek uygulamasının hesapladığı tüm ana çıktılarıdır.
Görevin hesap yapmak değil, bu çıktıları yöneticiye doğru bağlamla yorumlamaktır.

Kritik bağlam:
- Veri sadece geçmişte kazanılmış ihalelerden oluşur; kaybedilen ihale yoktur.
- Bu nedenle gerçek kazan/kaybet olasılığı, supervised classification veya rakip bazlı kazanma tahmini yapılamaz.
- Ana gösterge gerçek kazanma olasılığı değildir; "bu yeni ihale geçmişte kazandığımız işlere, fiyat bandımıza ve başarı profillerimize ne kadar benziyor?" sorusunun emsal bazlı karar destek göstergesidir.
- Kazanılmış verilerden şunlar yapılabildi: benzer ihale retrieval, normalize fiyat koridoru, Linear Regression Baseline, Random Forest / Ağaç Tabanlı Baseline, Cost Plus Margin referansı, Isolation Forest kazanım profili yakınlığı, K-Means başarı profili eşleşmesi ve senaryo skoru.
- Verilmeyen bilgiyi uydurma, sayısal değerleri değiştirme, sadece MODEL_CONTEXT_JSON içeriğine dayan.
- {DISCLAIMER}

Yanıtı Türkçe ve geçerli JSON olarak ver. Markdown kullanma. Şema:
{{
  "executive_summary": "2-3 cümlelik yönetici özeti",
  "decision_summary": "2-3 cümlelik yönetici özeti",
  "data_situation": "Veri kapsamını, kayıp veri olmadığını ve bunun modelleme sınırını açıkla",
  "recommended_action": "Teklif / manuel inceleme / fiyat revizyonu gibi net öneri",
  "scenario_rationale": "senaryo skorunu ve ana bileşenleri açıkla",
  "evidence_used": ["MODEL_CONTEXT_JSON içindeki evidence_items listesinden evidence_id değerleri"],
  "risk_warnings": ["en fazla 4 risk uyarısı"],
  "human_checks_required": ["en fazla 4 insan kontrol maddesi"],
  "forbidden_claims_check": false,
  "confidence_rationale": "model güveninin neden yüksek/orta/düşük olduğunu açıkla",
  "limitations": "gerçek kazanma olasılığı, rakip davranışı ve reveal edilmemiş gerçek sonuç verilmediğini kısa açıkla",
  "pwin_interpretation": "profil uyum göstergesini ve ana sürükleyicileri açıkla; pwin veya gerçek olasılık gibi anlatma",
  "pricing_interpretation": "düşük/orta/yüksek fiyat ve model uyumunu yorumla",
  "margin_risk": "maliyet ve karlılık açısından risk yorumu",
  "learner_signals": {{
    "isolation_forest": "one-class skorunun anlamı",
    "kmeans": "başarı profili yorumun",
    "regression_models": "Linear ve Random Forest baseline sinyalleri"
  }},
  "supporting_evidence": ["en fazla 4 kısa kanıt maddesi"],
  "risks": ["en fazla 4 risk maddesi"],
  "next_actions": ["en fazla 4 uygulanabilir sonraki adım"],
  "manual_review_required": true,
  "forbidden_claims_detected": false,
  "disclaimer": "{DISCLAIMER}"
}}

Kullanıcı sorusu varsa özellikle onu cevapla: {question}

MODEL_CONTEXT_JSON:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}
"""
