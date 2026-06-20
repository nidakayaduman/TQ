from __future__ import annotations

import ast
from pathlib import Path


APP_TEXT = Path("app.py").read_text(encoding="utf-8")


def test_no_raw_html_method_card_renderer():
    assert '<div class="method-card">' not in APP_TEXT
    assert "def method_card(" not in APP_TEXT


def test_no_html_sent_to_plain_streamlit_renderers():
    tree = ast.parse(APP_TEXT)
    blocked_calls = {"write", "text", "code"}
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "st"
            and func.attr in blocked_calls
        ):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "<div" in arg.value:
                offenders.append((node.lineno, func.attr))
    assert offenders == []


def test_turkish_pages_and_chatbot_exist():
    for label in [
        "Ana Sayfa",
        "Veri Yükleme ve Kalite Kontrol",
        "Metodoloji",
        "Test İhalesi Simülatörü",
        "Senaryo Analizi",
        "Gerçek Sonuçla Karşılaştır",
        "Backtest Sonuçları",
        "Benzer İhaleler",
        "AI Danışman",
        "Raporlar ve Audit",
    ]:
        assert label in APP_TEXT
    assert "st.chat_input" in APP_TEXT
    assert "st.chat_message" in APP_TEXT


def test_methodology_terms_are_present():
    for term in [
        "TF-IDF",
        "Cosine similarity",
        "K-Means",
        "Isolation Forest",
        "Fiyat Koridoru",
        "Backtest",
        "Sızıntı Kontrolü",
    ]:
        assert term in APP_TEXT

