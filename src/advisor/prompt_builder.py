"""Grounded advisor prompt builder."""

from __future__ import annotations

import json
from typing import Any

from ..constants import DISCLAIMER


def build_advisor_prompt(context: dict[str, Any]) -> str:
    safe_context = dict(context)
    return f"""Türkçe yanıt ver. Sadece aşağıdaki yapılandırılmış model çıktısını yorumla.

Zorunlu uyarı:
{DISCLAIMER}

Kurallar:
- Gizli veya reveal edilmemiş gerçek sonuçları kullanma.
- Rakip davranışı, kesin sonuç veya unsupported finansal etki uydurma.
- Çıktıyı JSON olarak şu alanlarla döndür: summary, profile_fit_explanation, price_corridor_explanation, margin_explanation, risk_explanation, confidence_explanation, similar_tenders_summary, manual_review_required, forbidden_claims_detected, disclaimer.

MODEL_OUTPUT_JSON:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}
"""

