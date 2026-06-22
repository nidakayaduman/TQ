"""Grounded advisor prompt builder."""

from __future__ import annotations

import json
from typing import Any

from ..constants import DISCLAIMER


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


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

Yanıt stili:
- Önce konuyu temel seviyede açıkla: bu soru ne anlama gelir, kullanıcı hangi kavramı okumalı?
- Sonra seçili ihalenin sonucunu iş diliyle söyle: hangi profile/kümeye yakın, bu ne kadar güçlü, manuel inceleme gerekir mi?
- Ardından teknik mekanizmayı kısa ama net anlat: retrieval, K-Means, Isolation Forest, fiyat koridoru, senaryo skoru ve ağırlıklar nasıl katkı verir?
- En sonda karar desteği yorumunu ver: teklif ekibi hangi varsayımları kontrol etmeli?
- Kullanıcı "hangi profile benziyor" veya "profil" diye sorarsa mutlaka K-Means profil grubu, profil uyum skoru, benzer ihale kalitesi ve Isolation Forest normal/sıra dışı sinyalini ayrı ayrı açıkla.
- Kullanıcı "fiyat koridoru" diye sorarsa önce düşük/orta/yüksek bandın ne demek olduğunu anlat, sonra seçili fiyatın bandın neresinde durduğunu ve geniş bant/model ayrılığı riskini yorumla.
- Kullanıcı "neden skor" veya "senaryo" diye sorarsa önce skorun otomatik karar olmadığını söyle, sonra bileşenleri ve ağırlıkları açıkla.
- Risk kodlarını ham teknik etiket olarak yazma; low_similarity, wide_price_band, medium_model_disagreement gibi kodları Türkçe açıklamaya çevir.
- Yanıt doğrudan teknik formülle başlamasın; temel açıklama ile başlasın, teknik ayrıntı ikinci bölümde gelsin.

Yanıtı Türkçe ve geçerli JSON olarak ver. Markdown kullanma. Şema:
{{
  "executive_summary": "Önce temel açıklama, sonra seçili ihaleye özel kısa iş yorumu içeren 3-5 cümlelik özet",
  "recommended_action": "Teklif / manuel inceleme / fiyat revizyonu gibi net öneri",
  "scenario_rationale": "Önce kavramı temelden açıkla, sonra teknik metrikleri ve skor bileşenlerini açıkla",
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
    "claims_true_win_chance": false,
    "claims_guaranteed_win": false
  }}
}}

Kullanıcı sorusu varsa özellikle onu cevapla: {question}

MODEL_CONTEXT_JSON:
{json.dumps(safe_context, ensure_ascii=False, indent=2, default=_json_default)}
"""
