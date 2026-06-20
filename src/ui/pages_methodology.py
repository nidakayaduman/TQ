"""Methodology page content."""

from __future__ import annotations

import streamlit as st

from .components import page_header


def render_methodology_page() -> None:
    page_header("Metodoloji", "Sadece kazanılmış ihale verisiyle pseudo-live değerlendirme.")
    st.markdown(
        """
        Sistem yalnızca geçmişte kazanılmış ihale kayıtlarını kullanır. Kaybedilmiş veya no-bid veri olmadığı için
        klasik kazan/kaybet sınıflandırması yapılmaz.

        Hesaplanan ana gösterge **Kazanılmış Profil Uyum Skoru**dur. Bu skor; geçmiş kazanılmış ihalelere benzerlik,
        fiyat bandı uyumu, marj/risk dengesi ve model güvenini birlikte yorumlar.

        Ana değerlendirme yöntemi **pseudo-live temporal backtesting** yaklaşımıdır. Test yılındaki her ihale,
        o tarihte yeni gelmiş gibi ele alınır. Gerçek sonuç alanları reveal adımına kadar retrieval, scorer,
        optimizer ve advisor katmanlarından maskelenir.

        Metrikler; fiyat bandı hizası, actual won scenario rank percentile, profil uyumu, retrieval kalitesi,
        segment kırılımları ve advisor güvenliği üzerinden izlenir.

        Gerçek supervised kazan/kaybet modellemesi için güvenilir kazanılmış, kaybedilmiş ve no-bid ihale verisi gerekir.
        """
    )

