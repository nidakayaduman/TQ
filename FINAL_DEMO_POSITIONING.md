# Final Demo Positioning

## Selected Company

**S.S.RİZE 3 NOLU MOTORLU TAŞIYICILAR KOOP**

This company is selected for the Streamlit MVP demo because it has the highest usable single-company EKAP history in the corrected unique-tender ranking from the collected completed-tender scan.

The selection is based on public EKAP completed tender records, not sector relevance.

Key dataset facts:

- Unique completed tenders: 12
- Contract/item rows: 12
- Unique contracts: 12
- Field completeness score: 0.8906
- Corrected final ranking score: 0.96718
- Dataset file: `data/final_demo_company.json`

## What The Demo Proves

The MVP can ingest completed public tender records and turn them into a reusable tender memory.

It can demonstrate:

- Tender memory from official completed EKAP records
- Similar completed tender retrieval
- Contract Value Corridor
- Estimated Cost Benchmark
- Estimated cost vs contract value comparison
- Historical Fit Score
- Backtesting-style metrics over completed public tender records

This is a **Public Tender Intelligence Demo**. It is not positioned as a pharma-specific dataset.

## What The Demo Does Not Prove

This demo does not prove win probability or award likelihood.

It does not prove go/no-go decision quality.

It does not calculate true unit price when quantity and unit are unavailable.

It does not infer missing quantities, units, costs, margins, competitor prices, or internal commercial assumptions.

## Official EKAP Fields Used

The final demo dataset uses official public EKAP fields only:

- `tender_id`
- `tender_name`
- `buyer_institution`
- `procurement_type`
- `procedure_type`
- `location`
- `tender_date`
- `winning_company`
- `contract_value_try`
- `estimated_cost_try`
- `contract_date`
- `item_description`

`item_description` is populated from the official item name when available; otherwise it uses the official tender name.

No unit price is calculated because quantity and unit are unavailable in the collected records.

## What Client Data Would Unlock Next

Client-owned data would allow the MVP to move from public-memory demonstration to decision support:

- Won/lost tender history
- Quantity/unit
- Internal cost
- Margin
- Competitor offers

With those fields, the platform could support deeper pricing simulations, margin analysis, competitor benchmarking, and eventually go/no-go modeling. Those are outside the scope of this public EKAP demo.
