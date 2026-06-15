# Tender Intelligence Decision Support

Streamlit MVP for pharmaceutical tender memory, historical price benchmarking,
margin simulation, and tender attractiveness scoring.

The app uses the synthetic Polifarma won-tender dataset:

- `data/polifarma_synthetic_tenders_2021_2025.csv`

## Purpose

The platform answers:

- What similar tenders have we won before?
- What price range was historically successful?
- What margin levels were historically achieved?
- How attractive does this tender look based on historical patterns?

## MVP Modules

1. Similar Tender Retrieval Engine
   - TF-IDF searchable tender text
   - cosine similarity
   - business-rule re-ranking
   - top 10 historical won tenders

2. Historical Price Benchmark Engine
   - primary benchmark: `inflation_adjusted_unit_price_2025_try`
   - nominal reference: `winning_unit_price_try`
   - min, P25, median, P75, max, average, standard deviation
   - conservative, balanced, and aggressive price scenarios

3. Margin Simulation Engine
   - user-entered estimated unit cost
   - conservative, balanced, and aggressive margin scenarios
   - historical gross-margin benchmark from the top 10 similar tenders

4. Tender Attractiveness Scoring
   - similarity score
   - balanced-margin score
   - historical strategic fit
   - competition risk
   - delivery risk

## Important Data Assumption

The dataset contains only historical won tenders. The MVP is therefore a
historical benchmarking and decision-support tool, not an award-likelihood
model or supervised tender-outcome classifier.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not on your PATH:

```bash
python3 -m streamlit run app.py
```

## Local Verification

The current implementation was verified locally with:

```bash
python3 -m py_compile app.py
python3 -m streamlit run app.py --server.port 8501 --server.headless true
```

Then the app was opened at `http://localhost:8501`, `Analiz Et` was clicked,
and the required sections rendered:

- Top 10 Similar Historical Won Tenders
- Historical Price Corridor
- Margin Simulation
- Tender Attractiveness Score
- Business Explanation
