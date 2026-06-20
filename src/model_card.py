"""Model card generation."""

from __future__ import annotations

from .constants import DISCLAIMER
from .model_version import MODEL_VERSION


def generate_model_card(metrics_summary: dict[str, object] | None = None) -> str:
    metrics_text = metrics_summary or {}
    return f"""# Tender IQ Agentic Bid Advisor Model Card

## Amaç
Bu prototip yalnızca geçmişte kazanılmış ihale verilerini kullanarak yeni bir ihalenin geçmiş kazanılmış profillere uyumunu, fiyat bandı hizasını, karlılık oranı/risk dengesini ve güven düzeyini gösterir.

## Zorunlu Uyarı
{DISCLAIMER}

## Veri ve Sınırlar
Veri kaybedilmiş veya no-bid ihaleleri içermez. Bu nedenle supervised kazan/kaybet sınıflandırması, gerçek sonuç tahmini veya rakip davranışı iddiası yapılmaz.

## Skor Mantığı
Senaryo puanı; kazanılmış profil uyumu, fiyat bandı uyumu, karlılık oranı skoru, model güveni ve risk cezası bileşenlerinden config ile hesaplanır.

## Test Metodolojisi
Ana yöntem pseudo-live temporal backtesting yaklaşımıdır. Test yılındaki her ihale, o tarihte yeni gelmiş gibi simüle edilir ve gerçek sonuç alanları reveal öncesi maskelenir.

## Metrikler
{metrics_text}

## Model ve Config Versiyonları
{MODEL_VERSION}

## Ne Zaman Kullanılmamalı
Kaybedilmiş/no-bid veri olmadan gerçek kazan/kaybet kararı, kesin sonuç beklentisi veya competitor davranışı tahmini için kullanılmamalıdır.

## Gelecek Veri İhtiyacı
Gerçek supervised kazan/kaybet modellemesi için güvenilir kazanılmış, kaybedilmiş ve no-bid kayıtları gerekir.
"""
