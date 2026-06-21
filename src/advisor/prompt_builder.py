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
  "recommended_action": "Teklif / manuel inceleme / fiyat revizyonu gibi net öneri",
  "scenario_rationale": "senaryo skorunu ve ana bileşenleri açıkla",
  "evidence_used": [
    {{"evidence_id": "MODEL_CONTEXT_JSON içindeki evidence_items listesinden evidence_id", "claim": "bu kanıta dayanan kısa iddia"}}
  ],
  "risk_warnings": ["en fazla 4 risk uyarısı"],
  "human_checks_required": ["en fazla 4 insan kontrol maddesi"],
  "confidence_rationale": "model güveninin neden yüksek/orta/düşük olduğunu açıkla",
  "limitations": [
    "Bu çıktı gerçek kazanma olasılığı değildir.",
    "Kaybedilmiş ihale verisi olmadığı için kazanma/kaybetme sınıflandırması yapılmaz.",
    "Rakip fiyatları tahmin edilmez; sadece mevcut veriyle karar desteği sağlanır."
  ],
  "forbidden_claims_check": {{
    "claims_true_win_probability": false,
    "claims_guaranteed_win": false
  }}
}}

Kullanıcı sorusu varsa özellikle onu cevapla: {question}

MODEL_CONTEXT_JSON:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}
"""
