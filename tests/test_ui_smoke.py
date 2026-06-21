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
        "Veri Seti ve Kalite Kontrol",
        "Metodoloji",
        "Test için İhale Seç",
        "Emsal İhale Analizi",
        "Profil Uyum Analizi",
        "Fiyat Koridoru ve Model Karşılaştırması",
        "Teklif Senaryoları",
        "Gerçek Sonuçla Karşılaştır",
        "Backtest Sonuçları",
        "AI Danışman",
        "Raporlar ve Kontroller",
    ]:
        assert label in APP_TEXT
    for removed_label in ["Test İhalesi Simülatörü", "Senaryo Analizi", "Benzer İhaleler", "Raporlar ve Audit"]:
        assert removed_label not in APP_TEXT
    assert "st.chat_input" in APP_TEXT
    assert "st.chat_message" in APP_TEXT
    for advisor_section in [
        "Kısa Özet",
        "Önerilen Aksiyon",
        "Senaryo Gerekçesi",
        "Kullanılan Kanıtlar",
        "Risk Uyarıları",
        "Manuel Kontrol Gerekenler",
        "Güven Gerekçesi",
        "Sınırlar",
    ]:
        assert advisor_section in APP_TEXT


def test_advisor_ui_does_not_render_raw_json_schema():
    assert "advisor_output.schema.json" not in APP_TEXT
    assert "TenderIQAdvisorOutput" not in APP_TEXT
    assert '"forbidden_claims_check"' not in APP_TEXT


def test_openrouter_models_are_explicit_and_selectable():
    for text in [
        "OpenRouter Model Seçimi",
        "st.selectbox(",
        '"number": "1"',
        "OpenRouter Auto",
        "openrouter/auto",
        '"number": "2"',
        "Google Gemini 2.5 Flash",
        "google/gemini-2.5-flash",
        '"number": "3"',
        "OpenAI GPT-4o Mini",
        "openai/gpt-4o-mini",
        "selected_openrouter_model_id()",
    ]:
        assert text in APP_TEXT


def test_advisor_answer_source_is_visible():
    for text in [
        "chat-source",
        "Hazır bağlam mesajı",
        "OpenRouter LLM -",
        "Güvenli fallback -",
        "answer_source",
        "advisor_llm_status",
        "set_advisor_llm_status",
    ]:
        assert text in APP_TEXT


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
