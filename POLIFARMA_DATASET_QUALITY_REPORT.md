# Polifarma Synthetic Dataset Quality Report

**Source file:** `/Users/kayaduman/Downloads/polifarma_synthetic_tender_dataset_2021_2025/polifarma_synthetic_tenders_2021_2025.csv`
**Rows:** 700
**Columns:** 33

## Summary

- Missing values: 0 total across all columns.
- Nominal price trend: Pass: nominal winning unit price medians increase monotonically from 2021 to 2025 overall and for every product with complete year coverage.
- Overall median `winning_unit_price_try` rises from 8.55 TRY in 2021 to 30.20 TRY in 2025.
- Overall median `inflation_adjusted_unit_price_2025_try` changes from 31.45 TRY in 2021 to 30.20 TRY in 2025.

## Distributions

### Year Distribution
| Year | Rows |
| --- | --- |
| 2021 | 140 |
| 2022 | 140 |
| 2023 | 140 |
| 2024 | 140 |
| 2025 | 140 |

### Product Group Distribution
| Product group | Rows |
| --- | --- |
| IV Solution | 373 |
| Injectable | 245 |
| Special Solution | 82 |

### Product Name Distribution
| Product name | Rows |
| --- | --- |
| %0.9 NaCl 100 ml | 22 |
| %0.9 NaCl 1000 ml | 36 |
| %0.9 NaCl 250 ml | 39 |
| %0.9 NaCl 500 ml | 42 |
| %10 Dekstroz 500 ml | 33 |
| %5 Dekstroz 250 ml | 40 |
| %5 Dekstroz 500 ml | 27 |
| Hemodiyaliz Solüsyonu 5000 ml | 41 |
| Levofloksasin IV 500 mg/100 ml | 34 |
| Mannitol %20 150 ml | 40 |
| Metronidazol IV 5 mg/ml 100 ml | 31 |
| Parasetamol IV 10 mg/ml 100 ml | 28 |
| Periton Diyaliz Solüsyonu 2000 ml | 41 |
| Potasyum Klorür Ampul 10 ml | 37 |
| Ringer Laktat 1000 ml | 26 |
| Ringer Laktat 500 ml | 36 |
| Siprofloksasin IV 200 mg/100 ml | 35 |
| Sodyum Bikarbonat Ampul 10 ml | 40 |
| Steril Su 100 ml | 38 |
| Steril Su 500 ml | 34 |

### Region Distribution
| Region | Rows |
| --- | --- |
| Akdeniz | 102 |
| Doğu Anadolu | 90 |
| Ege | 111 |
| Güneydoğu Anadolu | 99 |
| Karadeniz | 104 |
| Marmara | 87 |
| İç Anadolu | 107 |

## Numeric Checks

- `quantity`: min 13393.00, median 100114.00, max 988939.00
- `gross_margin_pct`: min 4.10, median 20.14, max 34.83

### Winning Unit Price TRY by Year
| Year | Min | Median | Max |
| --- | --- | --- | --- |
| 2021 | 2.79 | 8.55 | 48.0 |
| 2022 | 4.23 | 13.61 | 70.51 |
| 2023 | 6.0 | 21.8 | 107.62 |
| 2024 | 8.08 | 29.66 | 150.04 |
| 2025 | 9.14 | 30.2 | 178.18 |

### Inflation Adjusted Unit Price 2025 TRY by Year
| Year | Min | Median | Max |
| --- | --- | --- | --- |
| 2021 | 10.25 | 31.45 | 176.37 |
| 2022 | 10.05 | 32.31 | 167.39 |
| 2023 | 9.47 | 34.41 | 169.87 |
| 2024 | 9.7 | 35.61 | 180.17 |
| 2025 | 9.14 | 30.2 | 178.18 |

## Price Increase Logic Check

- Overall nominal median non-decreasing from 2021 to 2025: `True`
- Product-group nominal medians with decreases: 0
- Product-name nominal medians with decreases: 0

### Nominal Median by Product Group
| Product group | 2021 | 2022 | 2023 | 2024 | 2025 | Non-decreasing |
| --- | --- | --- | --- | --- | --- | --- |
| IV Solution | 6.89 | 10.46 | 16.74 | 21.65 | 25.02 | True |
| Injectable | 21.04 | 33.49 | 38.38 | 60.25 | 74.73 | True |
| Special Solution | 42.0 | 60.78 | 91.84 | 130.2 | 142.83 | True |

## Missing Values by Column

| Column | Missing values |
| --- | --- |
| record_type | 0 |
| company_name | 0 |
| tender_id | 0 |
| year | 0 |
| tender_date | 0 |
| contract_date | 0 |
| buyer_institution | 0 |
| city | 0 |
| region | 0 |
| procurement_type | 0 |
| procedure_type | 0 |
| tender_title | 0 |
| product_group | 0 |
| product_name | 0 |
| unit | 0 |
| quantity | 0 |
| delivery_months | 0 |
| competitor_count_estimate | 0 |
| estimated_unit_cost_try | 0 |
| estimated_total_cost_try | 0 |
| winning_unit_price_try | 0 |
| contract_value_try | 0 |
| internal_unit_cost_try | 0 |
| gross_margin_pct | 0 |
| gross_profit_try | 0 |
| discount_to_estimated_cost_pct | 0 |
| cpi_factor_to_2025 | 0 |
| sector_price_factor_to_2025 | 0 |
| inflation_adjusted_unit_price_2025_try | 0 |
| inflation_adjusted_contract_value_2025_try | 0 |
| result | 0 |
| strategic_fit_score | 0 |
| data_note | 0 |

## Notes

- This report validates the provided synthetic CSV only; it does not modify the Streamlit app and does not implement similarity search or price corridor logic.
- The JSON report contains the full product-level nominal median trend checks.
