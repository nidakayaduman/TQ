# İhale Karar Yardımcısı

Streamlit tabanlı demo uygulama. Amaç, yeni bir ihale için geçmişte kazanılmış
benzer ihalelere bakarak anlaşılır bir fiyat aralığı, marj görünümü ve ihale
puanı vermektir.

Uygulama X İlaç Şirketi için hazırlanmış sentetik demo verisini kullanır:

- `data/x_ilac_synthetic_tenders_2021_2025.csv`

Bu veri gerçek şirket ihale verisi değildir. Demo ve sunum amacıyla
oluşturulmuş sentetik veridir.

## Uygulama Ne Yapar?

- Benzer kazanılmış ihaleleri bulur.
- Geçmiş fiyatları Mayıs 2026 seviyesine taşır.
- Top-k benzer ihaleler, Linear Regression ve XGBoost ile model destekli fiyat aralığı gösterir.
- Girilen maliyete göre marj hesaplar.
- İhaleyi basit bir 0-100 puanla özetler.

## Veri Varsayımı

Veri sadece kazanılmış ihaleleri içerir. Kaybedilmiş ihale olmadığı için uygulama
kazanma ihtimali hesaplamaz, sonuç tahmini yapmaz ve sınıflandırma modeli
kullanmaz.

## Fiyat Normalizasyonu

Ana fiyat kıyaslaması artık `inflation_adjusted_unit_price_2026_try` alanını
kullanır. Bu alan fiyatları Mayıs 2026 TL seviyesine taşır.

Tam yıl 2026 enflasyonu henüz bilinmediği için 2026 yıl sonu tahmini
kullanılmaz. Bunun yerine en güncel gerçekleşmiş CPI seviyesi olan Mayıs 2026
kullanılır.

Kullanılan CPI katsayıları:

| Yıl | Mayıs 2026 katsayısı |
| --- | --- |
| 2021 | 5.9647 |
| 2022 | 3.6309 |
| 2023 | 2.2037 |
| 2024 | 1.5263 |
| 2025 | 1.1661 |
| 2026 | 1.0000 |

Formüller:

```text
inflation_adjusted_unit_price_2026_try =
winning_unit_price_try * cpi_factor_to_2026

inflation_adjusted_contract_value_2026_try =
contract_value_try * cpi_factor_to_2026
```

Eski 2025 normalize alanları izlenebilirlik için veri dosyasında korunur.

## Model Destekli Fiyat Koridoru

Uygulama yeni ihale için önce `overall_similarity_score` üretir ve en benzer
50 kazanılmış ihaleyi seçer. Bu skor; metin benzerliği, ürün, ürün grubu, bölge,
ihale usulü, miktar, teslim süresi ve tahmini rakip sayısını birlikte kullanır.

Fiyat tahmini iki modelle desteklenir:

- Linear Regression: açıklanabilir temel fiyat tahmini
- XGBoost Regression: doğrusal olmayan ilişkilere duyarlı fiyat tahmini

Modellerin hedef değişkeni:

```text
inflation_adjusted_unit_price_2026_try
```

Nihai düşük / orta / yüksek fiyat; top-k fiyat dağılımı, Linear Regression
tahmini, XGBoost tahmini ve 5-fold backtest hata payları birleştirilerek
hesaplanır. Bu hâlâ kazanma ihtimali modeli değildir; sadece kazanılmış ihale
hafızasına dayalı fiyat benchmark ve tahmin desteğidir.

## Başarı Metrikleri

Uygulama model kontrolü için 5-fold backtest metriklerini gösterir:

- MAE: Ortalama TL hata
- MAPE: Ortalama yüzde hata
- Coverage: Gerçek fiyatın model hata payı koridoruna düşme oranı

## Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

Eğer `streamlit` komutu bulunmazsa:

```bash
python3 -m streamlit run app.py
```
