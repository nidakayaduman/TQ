# Tender IQ Agentic Bid Advisor

Streamlit tabanlı production-minded MVP. Uygulama yalnızca tarihsel **kazanılmış** ihale verisiyle çalışır ve yeni bir ihale için **Kazanılmış Profil Uyum Skoru** üretir.

> Bu skor gerçek kazanma olasılığı değildir. Geçmişte kazanılmış ihalelere benzerlik, fiyat bandı uyumu, karlılık/risk dengesi ve model güvenini gösterir. Gerçek kazanma olasılığı için güvenilir kazanılmış ve kaybedilmiş ihale verisi gerekir.

## Ne Yapar?

- CSV kazanılmış ihale verisini yükler ve şema/kalite kontrolü yapar.
- Pseudo-live temporal split uygular: train, validation, test.
- Test ihalesinde gerçek sonuç alanlarını reveal öncesi maskeler.
- Benzer kazanılmış ihaleleri bulur.
- Fiyat koridoru, profil uyumu, karlılık oranı, risk ve güven skorlarını hesaplar.
- Aday senaryolar üretir ve config-driven senaryo skoru verir.
- Gerçek Sonuçla Karşılaştır ekranında seçilen senaryoyu gerçek kazanılmış fiyat ve karlılık oranıyla karşılaştırır.
- Backtest, segment metrikleri, baseline karşılaştırması ve audit/export çıktıları üretir.
- OpenRouter üzerinden seçilebilir LLM modelleriyle AI Danışman yorumu üretir; LLM yoksa deterministic fallback advisor güvenli şekilde devreye girer.

## Veri Sınırı

Veri setinde kaybedilmiş veya no-bid kayıt yoktur. Bu nedenle uygulama supervised kazan/kaybet sınıflandırması, kesin sonuç iddiası veya rakip davranışı tahmini yapmaz.

## Çalıştırma

```bash
make setup
make run
```

## Test

```bash
make test
python -m compileall app.py src tests
```

## Docker

```bash
make docker-build
make docker-run
```

Docker image Python 3.11 tabanlıdır, non-root kullanıcıyla çalışır, Streamlit 8501 portunu açar ve healthcheck içerir.

## Ana Modüller

- `src/schema.py`: şema normalizasyonu ve validasyon
- `src/feature_masking.py`: reveal öncesi actual result masking
- `src/leakage_audit.py`: leakage audit
- `src/retrieval.py`: benzer kazanılmış ihale retrieval
- `src/price_corridor.py`: fiyat bandı ve band uyumu
- `src/clustering.py`: Isolation Forest ve KMeans profil uyumu
- `src/optimizer/`: senaryo üretimi, validasyon ve skor
- `src/evaluation/`: backtest, metrics, baseline, segment/stress test
- `src/advisor/`: guardrail, output validation, fallback advisor
- `src/reporting/`: CSV, audit ve model artifact export

## Konfigürasyon

Skor ağırlıkları `config/app_config.yaml`, hard constraints `config/hard_constraints.yaml`, soft penalties `config/soft_penalties.yaml` içindedir. Logging, audit ve artifact dizinleri `config/observability.yaml` üzerinden yönetilir.

## Production Engineering Notları

- Structured JSON loglar `logs/app.jsonl` dosyasına ve stdout'a yazılır.
- Audit event'leri `audit_logs/` altında session, kullanıcı, event tipi, reveal durumu, input/output hash ve model/config versiyon bilgileriyle saklanır.
- Backtest artifact bilgileri `model_artifacts/{run_id}/` altında config snapshot, split manifest, metrics ve mümkün olduğunda pickle model dosyalarıyla tutulur.
- UI ham traceback veya teknik JSON göstermez; hata detayları loglara yazılır, kullanıcıya Türkçe iş mesajı gösterilir.
- `LLM_PROVIDER=openrouter` varsayılandır. OpenRouter API anahtarı yerel `.streamlit/secrets.toml` içinde tutulur; anahtar yoksa veya LLM guardrail doğrulaması geçmezse AI Danışman deterministik fallback advisor ile çalışmaya devam eder.

## AI Danışman LLM Modelleri

AI Danışman sayfasında OpenRouter modeli UI üzerinden seçilir ve sonraki sohbet yanıtlarında request body içindeki `model` alanı bu seçimle güncellenir:

1. `nvidia/nemotron-3-super-120b-a12b:free` - NVIDIA Nemotron 3 Super 120B A12B
2. `google/gemma-4-31b-it:free` - Google Gemma 4 31B IT
3. `openrouter/owl-alpha` - OpenRouter Owl Alpha

## Gizli Anahtarlar

Gerçek API anahtarları repo'ya commit edilmemelidir. Yerel kullanım için `.streamlit/secrets.toml` kullanın. Boş örnek key dosyaları repoda tutulmaz.

Eğer gerçek bir API anahtarı daha önce paylaşılmış, loglanmış veya yanlışlıkla commit edilmişse ilgili sağlayıcı panelinden anahtarı revoke/rotate edin ve yeni anahtarı sadece yerel secret dosyasında tutun.
