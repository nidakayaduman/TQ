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
        "read_local_openrouter_secret",
        "secrets.toml",
        "key=\"selected_openrouter_model\"",
        "format_func=lambda model_id",
        '"number": "1"',
        "Nex AGI Nex-N2-Pro",
        "nex-agi/nex-n2-pro:free",
        '"number": "2"',
        "OpenAI gpt-oss-120b",
        "openai/gpt-oss-120b:free",
        '"number": "3"',
        "NVIDIA Nemotron 3 Super 120B A12B",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter_model_attempt_order",
        "openrouter_model_label",
        "advisor-model-chain",
        "Seçili primary model",
        "Otomatik backup",
        "Son kullanılan model",
        "Primary model hata verirse sıradaki model otomatik denenir.",
        "selected_openrouter_model_id()",
    ]:
        assert text in APP_TEXT
    assert "google/gemma-4-31b-it:free" not in APP_TEXT


def test_advisor_prompt_explains_metrics_from_basics():
    for text in [
        "temelden açıkla: her önemli metrik",
        "ne ölçer",
        "hangi veriden/modelden gelir",
        "değer yüksek/düşük ise nasıl okunur",
        "iş kararı için ne anlama gelir",
        "Mixed-type clustering",
        "profil grubu benzerliği verir",
        "Isolation Forest sıra dışılık/manual review sinyali verir",
    ]:
        assert text in APP_TEXT


def test_advisor_answer_source_is_visible():
    for text in [
        "chat-source",
        "Bağlam hazır",
        "OpenRouter LLM",
        "OpenRouter LLM -",
        "Güvenli sistem yanıtı",
        "fallback_reason",
        "answer_source",
        "advisor_llm_status",
        "set_advisor_llm_status",
    ]:
        assert text in APP_TEXT
    assert "Güvenli fallback -" not in APP_TEXT
    assert "ADVISOR_CHAT_UI_VERSION" in APP_TEXT


def test_methodology_terms_are_present():
    for term in [
        "Yerel metin embedding",
        "Embedding yakınlığı",
        "Mixed-Type Clustering",
        "Isolation Forest",
        "Fiyat Koridoru",
        "Backtest",
        "Sızıntı Kontrolü",
    ]:
        assert term in APP_TEXT


def test_profile_page_contains_mixed_type_and_isolation_diagnostics():
    for text in [
        "Bu sayfa üç farklı profil sinyalini ayrı okur",
        "Profil skoru bileşenleri",
        "KNN emsal benzerliği",
        "Isolation tipiklik skoru",
        "Mixed-type cluster skoru",
        "Mixed-Type Cluster Analizi",
        "Mixed-Type Cluster Kalitesi",
        "Silhouette Score",
        "Cluster sıkılığı",
        "Cluster boyut aralığı",
        "Sıra Dışılık Kontrolü (Isolation Forest)",
        "Anomaly score",
        "Threshold",
        "Manual review flag",
    ]:
        assert text in APP_TEXT


def test_profile_models_are_not_described_as_price_predictors():
    for text in [
        "Mixed-type clustering fiyat tahmini",
        "Isolation Forest fiyat tahmini",
        "mixed-type clustering fiyatı doğru tahmin",
        "Isolation Forest fiyatı doğru tahmin",
    ]:
        assert text not in APP_TEXT


def test_removed_visual_noise_and_score_section_are_absent():
    assert "Skorlar nasıl okunur?" not in APP_TEXT
    for icon in ["📊", "🔎", "🧭", "📍", "📦", "💬", "🧱", "🔒", "🧰", "🎯", "💹", "🛡️", "⚠️", "✅"]:
        assert icon not in APP_TEXT
    assert 'icon="•"' not in APP_TEXT


def test_methodology_key_explanations_are_visible_cards_not_expanders():
    for label in [
        "Neden accuracy, precision, recall veya ROC-AUC ana başarı metriği değil?",
        "Top-K retrieval ve eşleşme metrikleri",
        "Sıra dışı durum örnekleri",
        "Fiyat koridoru nasıl oluşuyor?",
        "Başarıyı hangi metriklerle ölçüyoruz?",
    ]:
        assert label in APP_TEXT
        assert f'with st.expander("{label}' not in APP_TEXT
    for metric in [
        "Mixed-Type Silhouette Score",
        "Mixed-Type Cluster Sıkılığı",
        "Cluster Size Distribution",
        "Assignment Confidence",
        "Isolation Forest Inlier Rate",
        "Isolation Forest Anomaly Rate",
        "Synthetic Outlier Manual Review Rate",
    ]:
        assert metric in APP_TEXT


def test_scenario_page_labels_price_strategy_scope():
    assert "Öne Çıkan Fiyat Senaryoları" in APP_TEXT
    assert "Mixed-type clustering ve Isolation Forest profil tanılama sinyalleri ayrı olarak Profil Uyum Analizi sayfasında değerlendirilir." in APP_TEXT
    assert "Mixed-type clustering ve Isolation Forest burada fiyat tahmini olarak kullanılmaz" in APP_TEXT
    assert "SCENARIO_RENDER_CACHE_VERSION" in APP_TEXT
    assert "def select_strategy_cards" in APP_TEXT
    assert "valid_candidates if not valid_candidates.empty else candidates" in APP_TEXT


def test_test_tender_manual_inputs_persist_as_adjusted_context():
    for text in [
        "manual_tender_overrides",
        "editable_tender_defaults",
        "apply_editable_tender_values",
        "manual_adjusted_tender",
        "Bu alanlar manuel override olarak saklanır",
        "simülasyon sonrası emsal, profil, fiyat ve senaryo sayfalarına taşınır",
        "estimated_unit_cost_try",
    ]:
        assert text in APP_TEXT


def test_removed_reports_and_openrouter_noise_are_absent():
    for text in [
        "Sistem Kontrolleri",
        "Audit Durumu",
        "Son Log Olayları",
        "OpenRouter bağlantısı",
        "OpenRouter API key environment üzerinden bulundu.",
        "advisor_openrouter_model_label",
    ]:
        assert text not in APP_TEXT


def test_reveal_and_backtest_include_profile_diagnostics():
    for text in [
        "Profil Tanılama Metrikleri",
        "Mixed-Type / Isolation Forest",
        "Mixed-type atama güveni",
        "Manual review",
        "Ürün grubu anomaly oranı",
        "Mixed-Type Cluster Metrikleri",
        "Sıra Dışılık Kontrolü",
    ]:
        assert text in APP_TEXT


def test_backtest_distinguishes_selected_tender_from_full_test_set():
    for text in [
        "Seçili İhale Backtest Detayı",
        "Bu oran toplu MAPE değildir",
    ]:
        assert text not in APP_TEXT
    for text in [
        "Backtest Geneli: Fiyat Koridoru Metrikleri",
        "Bu metrikler seçili ihale için değil, test yılındaki tüm ihalelerin ortalamasıdır",
        "01 Benzerlik Tabanlı Koridor: düşük=p25, orta=predicted_mid_price/Top-K medyan, yüksek=p75",
        "Backtest hazırlanıyor",
        "Sonuçlar tamamlanmadan rapor bölümleri gösterilmez.",
    ]:
        assert text in APP_TEXT


def test_backtest_profile_diagnostics_cache_is_versioned():
    for text in [
        "BACKTEST_PROFILE_DIAGNOSTICS_CACHE_VERSION",
        "PROFILE_DIAGNOSTIC_COLUMNS",
        "backtest_has_profile_diagnostics",
        "load_backtest_results",
        "cached_backtest.clear()",
    ]:
        assert text in APP_TEXT
