# Price Corridor Report

**MVP positioning:** Tender Memory & Price Benchmarking Platform
**Source file:** `/Users/kayaduman/Downloads/polifarma_synthetic_tender_dataset_2021_2025/polifarma_synthetic_tenders_2021_2025.csv`
**Dataset rows:** 700

## Business Explanation

This is NOT a prediction model. It does not estimate win probability, recommend go/no-go decisions, predict tender outcomes, or infer bid success.

This is a historical benchmarking engine. For a new tender query, it retrieves similar historical won tenders and derives a reasonable price benchmark range from those examples.

The primary price corridor uses `inflation_adjusted_unit_price_2025_try` so older won prices are compared in 2025 TRY terms. `winning_unit_price_try` is included only as a reference view of nominal historical prices.

## Method

- Reuses the existing TF-IDF + cosine similarity retrieval engine.
- Query fields: `product_name`, `product_group`, `region`, and `procedure_type`.
- Historical searchable fields: `tender_title`, `product_name`, `product_group`, `buyer_institution`, `region`, and `procedure_type`.
- Retrieves top 30 candidates by text similarity and re-ranks to the top 10 with the existing hybrid score.
- Computes price, margin, and discount benchmarks only from the top 10 retrieved historical won tenders.
- Quantity is not used.

## Confidence Logic

| Level | Rule |
| --- | --- |
| High | 10 similar tenders found and average similarity score >= 0.75 |
| Medium | 7-9 similar tenders found, or average similarity score between 0.55 and 0.75 |
| Low | Fewer than 7 similar tenders found, or average similarity score below 0.55 |

## Overall Summary

- Test cases: 5
- Confidence counts: {'Medium': 2, 'High': 3}
- Average similarity across cases: 0.765166
- Streamlit MVP assessment: Suitable for integration into the Streamlit MVP as an explainable historical benchmarking module. It does not predict tender outcomes; it retrieves similar historical won tenders and summarizes achieved prices, margins, and discounts.

## Test Case Results

### Test Case 1

**Query:** `%0.9 NaCl 500 ml` / `IV Solution` / `Marmara` / `Açık İhale`

**Confidence:** Medium (n=10, avg similarity=0.648206)

#### Historical Price Corridor

| Metric | Inflation adjusted unit price 2025 TRY |
| --- | --- |
| min | 24.21 |
| p25 | 24.48 |
| median_p50 | 25.31 |
| p75 | 25.46 |
| max | 27.85 |

#### Winning Unit Price TRY Reference

| Metric | Winning unit price TRY |
| --- | --- |
| min | 6.92 |
| p25 | 10.4 |
| median_p50 | 19.01 |
| p75 | 23.72 |
| max | 25.31 |

#### Historical Margin Benchmark

| Metric | Value |
| --- | --- |
| average_gross_margin_pct | 18.12 |
| median_gross_margin_pct | 19.76 |
| p25_gross_margin_pct | 10.76 |
| p75_gross_margin_pct | 23.88 |

#### Historical Discount Benchmark

| Metric | Value |
| --- | --- |
| average_discount_to_estimated_cost_pct | 15.87 |
| median_discount_to_estimated_cost_pct | 17.35 |

#### Top 10 Similar Tenders Used

| Rank | Tender ID | Year | Product | Group | Region | Procedure | Adj. unit price 2025 TRY | Winning unit price TRY | Gross margin % | Discount % | TF-IDF | Final similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2022-0276 | 2022 | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 F) | 24.48 | 10.31 | 10.63 | 17.4 | 0.542117 | 0.67527 |
| 2 | SYN-POL-2024-0435 | 2024 | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 F) | 26.72 | 22.26 | 29.05 | 21.06 | 0.532084 | 0.66925 |
| 3 | SYN-POL-2021-0009 | 2021 | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 B) | 25.44 | 6.92 | 22.38 | 11.06 | 0.529606 | 0.667764 |
| 4 | SYN-POL-2023-0310 | 2023 | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 B) | 27.85 | 17.65 | 24.38 | 17.31 | 0.513061 | 0.657837 |
| 5 | SYN-POL-2022-0168 | 2022 | %0.9 NaCl 500 ml | IV Solution | İç Anadolu | Açık İhale | 25.31 | 10.66 | 26.37 | 18.97 | 0.579949 | 0.647969 |
| 6 | SYN-POL-2025-0617 | 2025 | %0.9 NaCl 500 ml | IV Solution | Ege | Açık İhale | 24.48 | 24.48 | 18.83 | 20.75 | 0.567021 | 0.640213 |
| 7 | SYN-POL-2021-0046 | 2021 | %0.9 NaCl 500 ml | IV Solution | Karadeniz | Açık İhale | 25.46 | 6.93 | 9.08 | 17.95 | 0.563244 | 0.637946 |
| 8 | SYN-POL-2025-0665 | 2025 | %0.9 NaCl 500 ml | IV Solution | Güneydoğu Anadolu | Açık İhale | 24.21 | 24.21 | 8.61 | 15.98 | 0.552485 | 0.631491 |
| 9 | SYN-POL-2025-0576 | 2025 | %0.9 NaCl 500 ml | IV Solution | Ege | Açık İhale | 25.31 | 25.31 | 20.69 | 5.05 | 0.551132 | 0.630679 |
| 10 | SYN-POL-2024-0503 | 2024 | %0.9 NaCl 500 ml | IV Solution | İç Anadolu | Açık İhale | 24.46 | 20.37 | 11.15 | 13.14 | 0.539401 | 0.623641 |

### Test Case 2

**Query:** `%5 Dekstroz 500 ml` / `IV Solution` / `Ege` / `Açık İhale`

**Confidence:** Medium (n=10, avg similarity=0.72809)

#### Historical Price Corridor

| Metric | Inflation adjusted unit price 2025 TRY |
| --- | --- |
| min | 24.69 |
| p25 | 26.39 |
| median_p50 | 28.3 |
| p75 | 30.5 |
| max | 34.81 |

#### Winning Unit Price TRY Reference

| Metric | Winning unit price TRY |
| --- | --- |
| min | 6.72 |
| p25 | 9.39 |
| median_p50 | 16.69 |
| p75 | 25.75 |
| max | 29.63 |

#### Historical Margin Benchmark

| Metric | Value |
| --- | --- |
| average_gross_margin_pct | 12.78 |
| median_gross_margin_pct | 11.25 |
| p25_gross_margin_pct | 9.82 |
| p75_gross_margin_pct | 14.37 |

#### Historical Discount Benchmark

| Metric | Value |
| --- | --- |
| average_discount_to_estimated_cost_pct | 10.57 |
| median_discount_to_estimated_cost_pct | 11.41 |

#### Top 10 Similar Tenders Used

| Rank | Tender ID | Year | Product | Group | Region | Procedure | Adj. unit price 2025 TRY | Winning unit price TRY | Gross margin % | Discount % | TF-IDF | Final similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2021-0028 | 2021 | %5 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 24.69 | 6.72 | 11.52 | 13.42 | 0.718253 | 0.830952 |
| 2 | SYN-POL-2021-0011 | 2021 | %5 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 27.59 | 7.51 | 14.76 | 14.03 | 0.712573 | 0.827544 |
| 3 | SYN-POL-2023-0290 | 2023 | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 34.81 | 22.05 | 20.55 | 7.74 | 0.625296 | 0.750178 |
| 4 | SYN-POL-2021-0073 | 2021 | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 32.47 | 8.84 | 9.51 | 12.01 | 0.608293 | 0.739976 |
| 5 | SYN-POL-2024-0469 | 2024 | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 30.79 | 25.64 | 6.82 | 7.71 | 0.596907 | 0.733144 |
| 6 | SYN-POL-2025-0624 | 2025 | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 29.01 | 29.01 | 10.59 | 10.81 | 0.590214 | 0.729128 |
| 7 | SYN-POL-2025-0644 | 2025 | %5 Dekstroz 500 ml | IV Solution | Ege | Pazarlık (MD 21 F) | 25.78 | 25.78 | 13.21 | 13.18 | 0.533532 | 0.670119 |
| 8 | SYN-POL-2025-0669 | 2025 | %5 Dekstroz 500 ml | IV Solution | Ege | Pazarlık (MD 21 B) | 29.63 | 29.63 | 20.3 | 3.55 | 0.532469 | 0.669481 |
| 9 | SYN-POL-2022-0253 | 2022 | %5 Dekstroz 500 ml | IV Solution | Ege | Çerçeve Anlaşma | 26.87 | 11.32 | 10.99 | 15.74 | 0.527791 | 0.666675 |
| 10 | SYN-POL-2022-0204 | 2022 | %5 Dekstroz 500 ml | IV Solution | Ege | Çerçeve Anlaşma | 26.23 | 11.05 | 9.57 | 7.55 | 0.52283 | 0.663698 |

### Test Case 3

**Query:** `Levofloksasin IV 500 mg/100 ml` / `Injectable` / `Akdeniz` / `Pazarlık (MD 21 B)`

**Confidence:** High (n=10, avg similarity=0.81391)

#### Historical Price Corridor

| Metric | Inflation adjusted unit price 2025 TRY |
| --- | --- |
| min | 107.61 |
| p25 | 115.89 |
| median_p50 | 118.68 |
| p75 | 122.45 |
| max | 125.42 |

#### Winning Unit Price TRY Reference

| Metric | Winning unit price TRY |
| --- | --- |
| min | 34.13 |
| p25 | 52.02 |
| median_p50 | 71.59 |
| p75 | 95.86 |
| max | 99.35 |

#### Historical Margin Benchmark

| Metric | Value |
| --- | --- |
| average_gross_margin_pct | 22.51 |
| median_gross_margin_pct | 21.23 |
| p25_gross_margin_pct | 20.07 |
| p75_gross_margin_pct | 26.62 |

#### Historical Discount Benchmark

| Metric | Value |
| --- | --- |
| average_discount_to_estimated_cost_pct | 13.13 |
| median_discount_to_estimated_cost_pct | 13.75 |

#### Top 10 Similar Tenders Used

| Rank | Tender ID | Year | Product | Group | Region | Procedure | Adj. unit price 2025 TRY | Winning unit price TRY | Gross margin % | Discount % | TF-IDF | Final similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2024-0499 | 2024 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 B) | 115.67 | 96.33 | 19.97 | 13.69 | 0.822703 | 0.893622 |
| 2 | SYN-POL-2021-0013 | 2021 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 B) | 125.42 | 34.13 | 28.67 | 13.81 | 0.808824 | 0.885294 |
| 3 | SYN-POL-2022-0247 | 2022 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 F) | 118.98 | 50.12 | 21.64 | 11.62 | 0.847993 | 0.858796 |
| 4 | SYN-POL-2022-0267 | 2022 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 F) | 123.5 | 52.02 | 28.42 | 14.21 | 0.815019 | 0.839011 |
| 5 | SYN-POL-2022-0196 | 2022 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 F) | 123.51 | 52.03 | 26.95 | 7.32 | 0.787254 | 0.822352 |
| 6 | SYN-POL-2023-0328 | 2023 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Açık İhale | 118.37 | 74.99 | 18.79 | 7.48 | 0.727138 | 0.786283 |
| 7 | SYN-POL-2023-0322 | 2023 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Açık İhale | 107.61 | 68.18 | 13.85 | 12.44 | 0.721801 | 0.783081 |
| 8 | SYN-POL-2024-0433 | 2024 | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Açık İhale | 119.3 | 99.35 | 20.37 | 13.95 | 0.68814 | 0.762884 |
| 9 | SYN-POL-2024-0537 | 2024 | Levofloksasin IV 500 mg/100 ml | Injectable | Marmara | Pazarlık (MD 21 B) | 113.42 | 94.45 | 25.62 | 15.5 | 0.758996 | 0.755398 |
| 10 | SYN-POL-2024-0549 | 2024 | Levofloksasin IV 500 mg/100 ml | Injectable | Ege | Pazarlık (MD 21 B) | 116.56 | 97.07 | 20.82 | 21.28 | 0.753964 | 0.752378 |

### Test Case 4

**Query:** `Sodyum Bikarbonat Ampul 10 ml` / `Injectable` / `İç Anadolu` / `Açık İhale`

**Confidence:** High (n=10, avg similarity=0.809435)

#### Historical Price Corridor

| Metric | Inflation adjusted unit price 2025 TRY |
| --- | --- |
| min | 18.9 |
| p25 | 19.98 |
| median_p50 | 20.27 |
| p75 | 20.91 |
| max | 21.71 |

#### Winning Unit Price TRY Reference

| Metric | Winning unit price TRY |
| --- | --- |
| min | 5.14 |
| p25 | 6.49 |
| median_p50 | 10.99 |
| p75 | 15.86 |
| max | 19.95 |

#### Historical Margin Benchmark

| Metric | Value |
| --- | --- |
| average_gross_margin_pct | 20.62 |
| median_gross_margin_pct | 20.53 |
| p25_gross_margin_pct | 18.47 |
| p75_gross_margin_pct | 23.21 |

#### Historical Discount Benchmark

| Metric | Value |
| --- | --- |
| average_discount_to_estimated_cost_pct | 15.68 |
| median_discount_to_estimated_cost_pct | 17.36 |

#### Top 10 Similar Tenders Used

| Rank | Tender ID | Year | Product | Group | Region | Procedure | Adj. unit price 2025 TRY | Winning unit price TRY | Gross margin % | Discount % | TF-IDF | Final similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2023-0385 | 2023 | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Açık İhale | 21.08 | 13.36 | 29.01 | 8.96 | 0.831405 | 0.898843 |
| 2 | SYN-POL-2021-0055 | 2021 | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Çerçeve Anlaşma | 20.29 | 5.52 | 23.94 | 16.15 | 0.760621 | 0.806373 |
| 3 | SYN-POL-2022-0152 | 2022 | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Pazarlık (MD 21 F) | 21.71 | 9.15 | 27.28 | 5.79 | 0.758287 | 0.804972 |
| 4 | SYN-POL-2025-0636 | 2025 | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Pazarlık (MD 21 B) | 19.95 | 19.95 | 20.96 | 18.86 | 0.756815 | 0.804089 |
| 5 | SYN-POL-2021-0043 | 2021 | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 21.28 | 5.79 | 20.1 | 11.52 | 0.837521 | 0.802513 |
| 6 | SYN-POL-2022-0255 | 2022 | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 20.41 | 8.6 | 21.04 | 13.63 | 0.837142 | 0.802285 |
| 7 | SYN-POL-2021-0034 | 2021 | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Çerçeve Anlaşma | 18.9 | 5.14 | 18.97 | 18.7 | 0.750577 | 0.800346 |
| 8 | SYN-POL-2023-0313 | 2023 | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 20.25 | 12.83 | 18.3 | 20.79 | 0.832673 | 0.799604 |
| 9 | SYN-POL-2025-0700 | 2025 | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 19.68 | 19.68 | 10.01 | 23.81 | 0.829259 | 0.797555 |
| 10 | SYN-POL-2024-0476 | 2024 | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 20.05 | 16.7 | 16.56 | 18.57 | 0.796288 | 0.777773 |

### Test Case 5

**Query:** `Periton Diyaliz Solüsyonu 2000 ml` / `Special Solution` / `Ege` / `Açık İhale`

**Confidence:** High (n=10, avg similarity=0.826187)

#### Historical Price Corridor

| Metric | Inflation adjusted unit price 2025 TRY |
| --- | --- |
| min | 117.84 |
| p25 | 130.81 |
| median_p50 | 143.1 |
| p75 | 144.26 |
| max | 156.61 |

#### Winning Unit Price TRY Reference

| Metric | Winning unit price TRY |
| --- | --- |
| min | 35.44 |
| p25 | 55.23 |
| median_p50 | 79.31 |
| p75 | 91.25 |
| max | 151.45 |

#### Historical Margin Benchmark

| Metric | Value |
| --- | --- |
| average_gross_margin_pct | 20.85 |
| median_gross_margin_pct | 21.52 |
| p25_gross_margin_pct | 16.39 |
| p75_gross_margin_pct | 25.11 |

#### Historical Discount Benchmark

| Metric | Value |
| --- | --- |
| average_discount_to_estimated_cost_pct | 11.36 |
| median_discount_to_estimated_cost_pct | 10.52 |

#### Top 10 Similar Tenders Used

| Rank | Tender ID | Year | Product | Group | Region | Procedure | Adj. unit price 2025 TRY | Winning unit price TRY | Gross margin % | Discount % | TF-IDF | Final similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2021-0007 | 2021 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Açık İhale | 143.38 | 39.02 | 25.17 | 13.02 | 0.837799 | 0.902679 |
| 2 | SYN-POL-2022-0143 | 2022 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Açık İhale | 126.88 | 53.45 | 14.14 | 8.53 | 0.836409 | 0.901845 |
| 3 | SYN-POL-2023-0374 | 2023 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Açık İhale | 144.42 | 91.5 | 21.24 | 11.82 | 0.830463 | 0.898278 |
| 4 | SYN-POL-2022-0181 | 2022 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Pazarlık (MD 21 F) | 143.77 | 60.56 | 24.92 | 16.82 | 0.76356 | 0.808136 |
| 5 | SYN-POL-2023-0318 | 2023 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Çerçeve Anlaşma | 132.51 | 83.96 | 16.52 | 5.33 | 0.761959 | 0.807175 |
| 6 | SYN-POL-2023-0281 | 2023 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Pazarlık (MD 21 F) | 156.61 | 99.22 | 27.5 | 25.07 | 0.760601 | 0.806361 |
| 7 | SYN-POL-2025-0697 | 2025 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Doğrudan Temin | 151.45 | 151.45 | 29.67 | 9.21 | 0.747276 | 0.798366 |
| 8 | SYN-POL-2023-0373 | 2023 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | İç Anadolu | Açık İhale | 117.84 | 74.66 | 11.23 | 5.44 | 0.802739 | 0.781643 |
| 9 | SYN-POL-2021-0096 | 2021 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | İç Anadolu | Açık İhale | 130.24 | 35.44 | 16.35 | 3.0 | 0.799908 | 0.779945 |
| 10 | SYN-POL-2023-0346 | 2023 | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Akdeniz | Açık İhale | 142.82 | 90.48 | 21.8 | 15.31 | 0.795735 | 0.777441 |

## Integration Readiness

The output is suitable for Streamlit MVP integration as a deterministic price benchmarking module. A Streamlit view can show the top 10 tender memory table, the historical price corridor, margin benchmark, discount benchmark, and confidence level for the user's query.

No Streamlit app changes, margin simulation, win probability logic, outcome prediction, or backtesting were built in this step.
