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
- Deterministic fallback advisor ile güvenli, grounded yorum üretir.

## Veri Sınırı

Veri setinde kaybedilmiş veya no-bid kayıt yoktur. Bu nedenle uygulama supervised kazan/kaybet sınıflandırması, kesin sonuç iddiası veya rakip davranışı tahmini yapmaz.

## Çalıştırma

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
python -m pytest
```

## Docker

```bash
docker build -t tender-iq .
docker run -p 8501:8501 tender-iq
```

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
- `src/reporting/`: CSV/HTML/audit export

## Konfigürasyon

Skor ağırlıkları `config/app_config.yaml`, hard constraints `config/hard_constraints.yaml`, soft penalties `config/soft_penalties.yaml` içindedir.

## Gizli Anahtarlar

Gerçek API anahtarları repo'ya commit edilmemelidir. Yerel kullanım için `.streamlit/secrets.toml` veya `.env` kullanın. Örnek dosyalar:

- `.env.example`
- `.streamlit/secrets.example.toml`
