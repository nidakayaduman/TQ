# Tender Intelligence Platform

Streamlit demo for a **Public Tender Intelligence** MVP.

The demo uses completed public EKAP tender records to show:

- tender memory
- similar completed tender retrieval
- Contract Value Corridor
- Estimated Cost Benchmark
- Historical Fit Score
- backtesting-style workflow metrics

It does **not** model win probability, go/no-go decisions, or true unit price
when quantity/unit are unavailable.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Final Demo Dataset

The app reads:

- `data/final_demo_company.json`

Selected company:

`S.S.RİZE 3 NOLU MOTORLU TAŞIYICILAR KOOP`

Reason: highest usable single-company EKAP history in the corrected unique
tender ranking from the completed-tender scan.

Current final demo summary:

- unique completed tenders: 12
- records with contract value: 12
- records with estimated cost: 3
- unit price calculated: false

See `FINAL_DEMO_POSITIONING.md` for business positioning.

## Data Files

Final app data:

- `data/final_demo_company.json`

EKAP traceability data:

- `data/ekap_company_discovery.json`
- `data/ekap_company_ranking_unique_tenders.json`
- `data/ekap_company_tender_records.json`
- `data/ekap_company_tender_records.csv`
- `data/ekap_source_audit.json`

Preserved KAP asset:

- `data/company_tender_records.json`
- `data/company_tender_records.csv`
- `data/source_audit.json`

## Scripts

EKAP sidecar pipeline:

```bash
python scripts/export_ekap_dataset.py --max-completed-tenders 500
```

KAP pipeline:

```bash
python scripts/export_dataset.py
```

The EKAP pipeline uses public completed-tender records with status
`Sonuç İlanı Yayımlanmış`. Item rows are not counted as separate tenders in the
corrected company ranking.

## Limitations

This is a public EKAP demo dataset. It proves that the MVP can ingest completed
public tender records, build tender memory, retrieve similar completed tenders,
and benchmark contract values against estimated costs.

It does not prove win probability, go/no-go performance, internal margin, or
competitor behavior. Those require client-owned data:

- won/lost tender history
- quantity/unit
- internal cost
- margin
- competitor offers
