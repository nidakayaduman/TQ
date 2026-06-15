"""Export the EKAP completed-tender sidecar dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from discover_companies import (
    DATA_DIR,
    DISCOVERY_OUTPUT,
    RANKING_UNIQUE_TENDERS_OUTPUT,
    discover_companies,
    write_discovery_outputs,
)
from normalize_ekap_records import NORMALIZED_OUTPUT, normalize_ekap_records
from scrape_ekap import SCRAPE_OUTPUT, scrape_selected_company_records


EKAP_JSON = DATA_DIR / "ekap_company_tender_records.json"
EKAP_CSV = DATA_DIR / "ekap_company_tender_records.csv"
EKAP_AUDIT = DATA_DIR / "ekap_source_audit.json"


CSV_FIELDS = [
    "tender_id",
    "tender_name",
    "buyer_institution",
    "procurement_type",
    "procedure_type",
    "location",
    "tender_date",
    "result_status",
    "winning_company",
    "contract_value_try",
    "estimated_cost_try",
    "contract_date",
    "highest_bid_try",
    "lowest_bid_try",
    "termination_status",
    "transfer_status",
    "item_details_available",
    "item_name",
    "quantity",
    "unit",
    "item_contract_value_try",
    "item_unit_price_try",
    "source_url",
    "source_type",
    "notes",
    "extraction_confidence",
]


def _write_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CSV_FIELDS})


def _build_audit(discovery: dict, unique_ranking: list[dict], final_payload: dict) -> dict:
    records = discovery.get("records", [])
    selected = final_payload.get("selected_company") or {}
    selected_name = selected.get("company")
    top_10 = unique_ranking[:10]
    return {
        "source_workstream": "EKAP v2 completed tender sidecar investigation",
        "preserved_kap_dataset": [
            "data/company_tender_records.json",
            "data/company_tender_records.csv",
            "data/source_audit.json",
        ],
        "completed_result_records_checked": discovery.get("completed_tenders_checked", 0),
        "contract_records_extracted": len(records),
        "records_with_contract_value": sum(1 for record in records if record.get("contract_value_try") is not None),
        "records_with_estimated_cost": sum(1 for record in records if record.get("estimated_cost_try") not in (None, 0)),
        "records_with_winning_company": sum(1 for record in records if record.get("winning_company")),
        "records_with_item_details": sum(1 for record in records if record.get("item_details_available")),
        "companies_evaluated_unique_tender_ranking": unique_ranking,
        "top_10_companies_by_unique_tender_score": top_10,
        "unique_tender_validation": {
            "ranking_basis": "final_score = 70% normalized unique tender count + 30% tender-level field completeness",
            "item_rows_are_not_counted_as_tenders": True,
        },
        "selected_company": selected_name,
        "reason_for_selection": (
            "Selected from the completed-tender sample by final_score = "
            "70% normalized unique tender count + 30% tender-level field completeness. "
            "Item rows are not counted as separate tenders."
            if selected_name
            else None
        ),
        "blocked_pages": discovery.get("blocked_pages", []),
        "missing_fields": {
            "quantity": "Not consistently exposed in the completed tender contract summary; set to null unless an accessible item endpoint provides it.",
            "unit": "Not consistently exposed in the completed tender contract summary; set to null unless an accessible item endpoint provides it.",
            "item_unit_price_try": "Calculated only when quantity and item/contract value are both available.",
        },
        "limitations": [
            "Ranking is based on the configured completed-tender sample size, not the entire EKAP corpus.",
            "Unique tender ranking counts multiple item rows from one tender as one tender.",
            "Some tenders are exception-scope procurements where official estimated cost is zero or absent.",
            "Clickable item/part details often expose part names and part-level amounts, but not quantity/unit in the summary response.",
            "No synthetic values are mixed into official EKAP records.",
        ],
    }


def export_ekap_dataset(max_completed_tenders: int = 150, page_size: int = 50, delay_seconds: float = 0.4) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    discovery_result = discover_companies(max_completed_tenders=max_completed_tenders, page_size=page_size, delay_seconds=delay_seconds)
    write_discovery_outputs(discovery_result)

    selected_company = (
        discovery_result.company_ranking_unique_tenders[0]["company"]
        if discovery_result.company_ranking_unique_tenders
        else None
    )
    scrape = scrape_selected_company_records(selected_company, DISCOVERY_OUTPUT) if selected_company else {"selected_company": None, "records": [], "audit": {}}
    SCRAPE_OUTPUT.write_text(json.dumps(scrape, ensure_ascii=False, indent=2), encoding="utf-8")

    normalized = normalize_ekap_records(SCRAPE_OUTPUT, RANKING_UNIQUE_TENDERS_OUTPUT)
    NORMALIZED_OUTPUT.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    final_payload = {
        "selected_company": normalized["selected_company"],
        "records": normalized["records"],
        "synthetic_fields_for_demo": normalized["synthetic_fields_for_demo"],
    }
    EKAP_JSON.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(final_payload["records"], EKAP_CSV)

    discovery = json.loads(DISCOVERY_OUTPUT.read_text(encoding="utf-8"))
    unique_ranking = json.loads(RANKING_UNIQUE_TENDERS_OUTPUT.read_text(encoding="utf-8"))
    audit = _build_audit(discovery, unique_ranking, final_payload)
    EKAP_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"final": final_payload, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-completed-tenders", type=int, default=150)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--delay-seconds", type=float, default=0.4)
    args = parser.parse_args()
    payload = export_ekap_dataset(args.max_completed_tenders, args.page_size, args.delay_seconds)
    selected = payload["final"].get("selected_company") or {}
    print(f"EKAP selected company: {selected.get('company')}")
    print(f"EKAP records: {len(payload['final']['records'])}")
    print(f"Wrote {RANKING_UNIQUE_TENDERS_OUTPUT}, {EKAP_JSON}, {EKAP_CSV}, {EKAP_AUDIT}")


if __name__ == "__main__":
    main()
