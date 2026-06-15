# TF-IDF Similar Tender Retrieval Report

**Source file:** `/Users/kayaduman/Downloads/polifarma_synthetic_tender_dataset_2021_2025/polifarma_synthetic_tenders_2021_2025.csv`
**Dataset rows:** 700

## Method

- Historical searchable text uses `tender_title`, `product_name`, `product_group`, `buyer_institution`, `region`, and `procedure_type`.
- Query text uses `product_name`, `product_group`, `region`, and `procedure_type`.
- The first pass retrieves the top 30 candidates with `sklearn` `TfidfVectorizer` and `cosine_similarity`.
- The second pass re-ranks candidates with the requested business rules.
- Quantity is not used in this version.

## Hybrid Score

| Component | Weight | Score behavior |
| --- | --- | --- |
| TF-IDF cosine similarity | 60% | Raw cosine similarity from 0 to 1 |
| Product group exact match | 15% | 1 for exact match, otherwise 0 |
| Product name exact or partial match | 10% | 1 for exact/substring match, otherwise token-overlap ratio |
| Region exact match | 10% | 1 for exact match, otherwise 0 |
| Procedure type exact match | 5% | 1 for exact match, otherwise 0 |

## Quality Summary

- All test cases pass MVP quality bar: `True`
- Average top-10 product group matches: 10.0
- Average top-10 exact or partial product-name matches: 10.0
- Average top-10 region matches: 6.8
- Average top-10 procedure-type matches: 5.6
- Streamlit MVP assessment: Good enough for the Streamlit MVP: retrieval consistently prioritizes matching product group and exact or closely related product names, while region and procedure type influence ranking without dominating it.

Region and procedure type are visible in the ranking, but the product context dominates as intended.

## Test Case Results

### Test Case 1

**Query:** `%0.9 NaCl 500 ml` / `IV Solution` / `Marmara` / `Açık İhale`

- Product group matches in top 10: 10
- Exact product-name matches in top 10: 10
- Exact or partial product-name matches in top 10: 10
- Region matches in top 10: 4
- Procedure-type matches in top 10: 6

| Rank | Tender ID | Year | Tender title | Product | Group | Region | Procedure | Winning unit price TRY | Inflation adjusted unit price 2025 TRY | TF-IDF | Group score | Product score | Region score | Procedure score | Final score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2022-0276 | 2022 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 F) | 10.31 | 24.48 | 0.542117 | 1.0 | 1.0 | 1.0 | 0.0 | 0.67527 |
| 2 | SYN-POL-2024-0435 | 2024 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 F) | 22.26 | 26.72 | 0.532084 | 1.0 | 1.0 | 1.0 | 0.0 | 0.66925 |
| 3 | SYN-POL-2021-0009 | 2021 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 B) | 6.92 | 25.44 | 0.529606 | 1.0 | 1.0 | 1.0 | 0.0 | 0.667764 |
| 4 | SYN-POL-2023-0310 | 2023 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Marmara | Pazarlık (MD 21 B) | 17.65 | 27.85 | 0.513061 | 1.0 | 1.0 | 1.0 | 0.0 | 0.657837 |
| 5 | SYN-POL-2022-0168 | 2022 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | İç Anadolu | Açık İhale | 10.66 | 25.31 | 0.579949 | 1.0 | 1.0 | 0.0 | 1.0 | 0.647969 |
| 6 | SYN-POL-2025-0617 | 2025 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Ege | Açık İhale | 24.48 | 24.48 | 0.567021 | 1.0 | 1.0 | 0.0 | 1.0 | 0.640213 |
| 7 | SYN-POL-2021-0046 | 2021 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Karadeniz | Açık İhale | 6.93 | 25.46 | 0.563244 | 1.0 | 1.0 | 0.0 | 1.0 | 0.637946 |
| 8 | SYN-POL-2025-0665 | 2025 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Güneydoğu Anadolu | Açık İhale | 24.21 | 24.21 | 0.552485 | 1.0 | 1.0 | 0.0 | 1.0 | 0.631491 |
| 9 | SYN-POL-2025-0576 | 2025 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | Ege | Açık İhale | 25.31 | 25.31 | 0.551132 | 1.0 | 1.0 | 0.0 | 1.0 | 0.630679 |
| 10 | SYN-POL-2024-0503 | 2024 | %0.9 NaCl 500 ml Alımı | %0.9 NaCl 500 ml | IV Solution | İç Anadolu | Açık İhale | 20.37 | 24.46 | 0.539401 | 1.0 | 1.0 | 0.0 | 1.0 | 0.623641 |

### Test Case 2

**Query:** `%5 Dekstroz 500 ml` / `IV Solution` / `Ege` / `Açık İhale`

- Product group matches in top 10: 10
- Exact product-name matches in top 10: 6
- Exact or partial product-name matches in top 10: 10
- Region matches in top 10: 10
- Procedure-type matches in top 10: 6

| Rank | Tender ID | Year | Tender title | Product | Group | Region | Procedure | Winning unit price TRY | Inflation adjusted unit price 2025 TRY | TF-IDF | Group score | Product score | Region score | Procedure score | Final score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2021-0028 | 2021 | %5 Dekstroz 500 ml Alımı | %5 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 6.72 | 24.69 | 0.718253 | 1.0 | 1.0 | 1.0 | 1.0 | 0.830952 |
| 2 | SYN-POL-2021-0011 | 2021 | %5 Dekstroz 500 ml Alımı | %5 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 7.51 | 27.59 | 0.712573 | 1.0 | 1.0 | 1.0 | 1.0 | 0.827544 |
| 3 | SYN-POL-2023-0290 | 2023 | %10 Dekstroz 500 ml Alımı | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 22.05 | 34.81 | 0.625296 | 1.0 | 0.75 | 1.0 | 1.0 | 0.750178 |
| 4 | SYN-POL-2021-0073 | 2021 | %10 Dekstroz 500 ml Alımı | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 8.84 | 32.47 | 0.608293 | 1.0 | 0.75 | 1.0 | 1.0 | 0.739976 |
| 5 | SYN-POL-2024-0469 | 2024 | %10 Dekstroz 500 ml Alımı | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 25.64 | 30.79 | 0.596907 | 1.0 | 0.75 | 1.0 | 1.0 | 0.733144 |
| 6 | SYN-POL-2025-0624 | 2025 | %10 Dekstroz 500 ml Alımı | %10 Dekstroz 500 ml | IV Solution | Ege | Açık İhale | 29.01 | 29.01 | 0.590214 | 1.0 | 0.75 | 1.0 | 1.0 | 0.729128 |
| 7 | SYN-POL-2025-0644 | 2025 | %5 Dekstroz 500 ml Alımı | %5 Dekstroz 500 ml | IV Solution | Ege | Pazarlık (MD 21 F) | 25.78 | 25.78 | 0.533532 | 1.0 | 1.0 | 1.0 | 0.0 | 0.670119 |
| 8 | SYN-POL-2025-0669 | 2025 | %5 Dekstroz 500 ml Alımı | %5 Dekstroz 500 ml | IV Solution | Ege | Pazarlık (MD 21 B) | 29.63 | 29.63 | 0.532469 | 1.0 | 1.0 | 1.0 | 0.0 | 0.669481 |
| 9 | SYN-POL-2022-0253 | 2022 | %5 Dekstroz 500 ml Alımı | %5 Dekstroz 500 ml | IV Solution | Ege | Çerçeve Anlaşma | 11.32 | 26.87 | 0.527791 | 1.0 | 1.0 | 1.0 | 0.0 | 0.666675 |
| 10 | SYN-POL-2022-0204 | 2022 | %5 Dekstroz 500 ml Alımı | %5 Dekstroz 500 ml | IV Solution | Ege | Çerçeve Anlaşma | 11.05 | 26.23 | 0.52283 | 1.0 | 1.0 | 1.0 | 0.0 | 0.663698 |

### Test Case 3

**Query:** `Levofloksasin IV 500 mg/100 ml` / `Injectable` / `Akdeniz` / `Pazarlık (MD 21 B)`

- Product group matches in top 10: 10
- Exact product-name matches in top 10: 10
- Exact or partial product-name matches in top 10: 10
- Region matches in top 10: 8
- Procedure-type matches in top 10: 4

| Rank | Tender ID | Year | Tender title | Product | Group | Region | Procedure | Winning unit price TRY | Inflation adjusted unit price 2025 TRY | TF-IDF | Group score | Product score | Region score | Procedure score | Final score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2024-0499 | 2024 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 B) | 96.33 | 115.67 | 0.822703 | 1.0 | 1.0 | 1.0 | 1.0 | 0.893622 |
| 2 | SYN-POL-2021-0013 | 2021 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 B) | 34.13 | 125.42 | 0.808824 | 1.0 | 1.0 | 1.0 | 1.0 | 0.885294 |
| 3 | SYN-POL-2022-0247 | 2022 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 F) | 50.12 | 118.98 | 0.847993 | 1.0 | 1.0 | 1.0 | 0.0 | 0.858796 |
| 4 | SYN-POL-2022-0267 | 2022 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 F) | 52.02 | 123.5 | 0.815019 | 1.0 | 1.0 | 1.0 | 0.0 | 0.839011 |
| 5 | SYN-POL-2022-0196 | 2022 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Pazarlık (MD 21 F) | 52.03 | 123.51 | 0.787254 | 1.0 | 1.0 | 1.0 | 0.0 | 0.822352 |
| 6 | SYN-POL-2023-0328 | 2023 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Açık İhale | 74.99 | 118.37 | 0.727138 | 1.0 | 1.0 | 1.0 | 0.0 | 0.786283 |
| 7 | SYN-POL-2023-0322 | 2023 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Açık İhale | 68.18 | 107.61 | 0.721801 | 1.0 | 1.0 | 1.0 | 0.0 | 0.783081 |
| 8 | SYN-POL-2024-0433 | 2024 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Akdeniz | Açık İhale | 99.35 | 119.3 | 0.68814 | 1.0 | 1.0 | 1.0 | 0.0 | 0.762884 |
| 9 | SYN-POL-2024-0537 | 2024 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Marmara | Pazarlık (MD 21 B) | 94.45 | 113.42 | 0.758996 | 1.0 | 1.0 | 0.0 | 1.0 | 0.755398 |
| 10 | SYN-POL-2024-0549 | 2024 | Levofloksasin IV 500 mg/100 ml Alımı | Levofloksasin IV 500 mg/100 ml | Injectable | Ege | Pazarlık (MD 21 B) | 97.07 | 116.56 | 0.753964 | 1.0 | 1.0 | 0.0 | 1.0 | 0.752378 |

### Test Case 4

**Query:** `Sodyum Bikarbonat Ampul 10 ml` / `Injectable` / `İç Anadolu` / `Açık İhale`

- Product group matches in top 10: 10
- Exact product-name matches in top 10: 10
- Exact or partial product-name matches in top 10: 10
- Region matches in top 10: 5
- Procedure-type matches in top 10: 6

| Rank | Tender ID | Year | Tender title | Product | Group | Region | Procedure | Winning unit price TRY | Inflation adjusted unit price 2025 TRY | TF-IDF | Group score | Product score | Region score | Procedure score | Final score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2023-0385 | 2023 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Açık İhale | 13.36 | 21.08 | 0.831405 | 1.0 | 1.0 | 1.0 | 1.0 | 0.898843 |
| 2 | SYN-POL-2021-0055 | 2021 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Çerçeve Anlaşma | 5.52 | 20.29 | 0.760621 | 1.0 | 1.0 | 1.0 | 0.0 | 0.806373 |
| 3 | SYN-POL-2022-0152 | 2022 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Pazarlık (MD 21 F) | 9.15 | 21.71 | 0.758287 | 1.0 | 1.0 | 1.0 | 0.0 | 0.804972 |
| 4 | SYN-POL-2025-0636 | 2025 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Pazarlık (MD 21 B) | 19.95 | 19.95 | 0.756815 | 1.0 | 1.0 | 1.0 | 0.0 | 0.804089 |
| 5 | SYN-POL-2021-0043 | 2021 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 5.79 | 21.28 | 0.837521 | 1.0 | 1.0 | 0.0 | 1.0 | 0.802513 |
| 6 | SYN-POL-2022-0255 | 2022 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 8.6 | 20.41 | 0.837142 | 1.0 | 1.0 | 0.0 | 1.0 | 0.802285 |
| 7 | SYN-POL-2021-0034 | 2021 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | İç Anadolu | Çerçeve Anlaşma | 5.14 | 18.9 | 0.750577 | 1.0 | 1.0 | 1.0 | 0.0 | 0.800346 |
| 8 | SYN-POL-2023-0313 | 2023 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 12.83 | 20.25 | 0.832673 | 1.0 | 1.0 | 0.0 | 1.0 | 0.799604 |
| 9 | SYN-POL-2025-0700 | 2025 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 19.68 | 19.68 | 0.829259 | 1.0 | 1.0 | 0.0 | 1.0 | 0.797555 |
| 10 | SYN-POL-2024-0476 | 2024 | Sodyum Bikarbonat Ampul 10 ml Alımı | Sodyum Bikarbonat Ampul 10 ml | Injectable | Güneydoğu Anadolu | Açık İhale | 16.7 | 20.05 | 0.796288 | 1.0 | 1.0 | 0.0 | 1.0 | 0.777773 |

### Test Case 5

**Query:** `Periton Diyaliz Solüsyonu 2000 ml` / `Special Solution` / `Ege` / `Açık İhale`

- Product group matches in top 10: 10
- Exact product-name matches in top 10: 10
- Exact or partial product-name matches in top 10: 10
- Region matches in top 10: 7
- Procedure-type matches in top 10: 6

| Rank | Tender ID | Year | Tender title | Product | Group | Region | Procedure | Winning unit price TRY | Inflation adjusted unit price 2025 TRY | TF-IDF | Group score | Product score | Region score | Procedure score | Final score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SYN-POL-2021-0007 | 2021 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Açık İhale | 39.02 | 143.38 | 0.837799 | 1.0 | 1.0 | 1.0 | 1.0 | 0.902679 |
| 2 | SYN-POL-2022-0143 | 2022 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Açık İhale | 53.45 | 126.88 | 0.836409 | 1.0 | 1.0 | 1.0 | 1.0 | 0.901845 |
| 3 | SYN-POL-2023-0374 | 2023 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Açık İhale | 91.5 | 144.42 | 0.830463 | 1.0 | 1.0 | 1.0 | 1.0 | 0.898278 |
| 4 | SYN-POL-2022-0181 | 2022 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Pazarlık (MD 21 F) | 60.56 | 143.77 | 0.76356 | 1.0 | 1.0 | 1.0 | 0.0 | 0.808136 |
| 5 | SYN-POL-2023-0318 | 2023 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Çerçeve Anlaşma | 83.96 | 132.51 | 0.761959 | 1.0 | 1.0 | 1.0 | 0.0 | 0.807175 |
| 6 | SYN-POL-2023-0281 | 2023 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Pazarlık (MD 21 F) | 99.22 | 156.61 | 0.760601 | 1.0 | 1.0 | 1.0 | 0.0 | 0.806361 |
| 7 | SYN-POL-2025-0697 | 2025 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Ege | Doğrudan Temin | 151.45 | 151.45 | 0.747276 | 1.0 | 1.0 | 1.0 | 0.0 | 0.798366 |
| 8 | SYN-POL-2023-0373 | 2023 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | İç Anadolu | Açık İhale | 74.66 | 117.84 | 0.802739 | 1.0 | 1.0 | 0.0 | 1.0 | 0.781643 |
| 9 | SYN-POL-2021-0096 | 2021 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | İç Anadolu | Açık İhale | 35.44 | 130.24 | 0.799908 | 1.0 | 1.0 | 0.0 | 1.0 | 0.779945 |
| 10 | SYN-POL-2023-0346 | 2023 | Periton Diyaliz Solüsyonu 2000 ml Alımı | Periton Diyaliz Solüsyonu 2000 ml | Special Solution | Akdeniz | Açık İhale | 90.48 | 142.82 | 0.795735 | 1.0 | 1.0 | 0.0 | 1.0 | 0.777441 |

## MVP Positioning

This is an MVP-grade, explainable similarity engine. It is lightweight and suitable for Streamlit Community Cloud because it uses only a small in-memory TF-IDF matrix and deterministic business-rule re-ranking.

Future enhancement options:

- multilingual sentence embeddings
- FAISS/vector database
- hybrid semantic + structured retrieval
- learned ranking using real client feedback

No Streamlit app changes, price corridor, margin simulation, FAISS, sentence-transformers, or external APIs were used.
