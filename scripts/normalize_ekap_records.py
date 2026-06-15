"""Normalize EKAP completed-tender records into the MVP sidecar schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from discover_companies import DATA_DIR, RANKING_UNIQUE_TENDERS_OUTPUT, normalize_company_name
from scrape_ekap import SCRAPE_OUTPUT


NORMALIZED_OUTPUT = DATA_DIR / "ekap_normalized_records.json"


def _unit_price(record: dict) -> float | None:
    quantity = record.get("quantity")
    total = record.get("item_contract_value_try") or record.get("contract_value_try")
    if quantity in (None, 0, "") or total is None:
        return None
    try:
        return round(float(total) / float(quantity), 4)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def normalize_ekap_records(scrape_path: Path = SCRAPE_OUTPUT, ranking_path: Path = RANKING_UNIQUE_TENDERS_OUTPUT) -> dict:
    scrape = json.loads(scrape_path.read_text(encoding="utf-8"))
    ranking = json.loads(ranking_path.read_text(encoding="utf-8")) if ranking_path.exists() else []
    selected_company = normalize_company_name(scrape.get("selected_company"))

    records = []
    for record in scrape.get("records", []):
        normalized = {
            "tender_id": record.get("tender_id"),
            "tender_name": record.get("tender_name"),
            "buyer_institution": record.get("buyer_institution"),
            "procurement_type": record.get("procurement_type"),
            "procedure_type": record.get("procedure_type"),
            "location": record.get("location"),
            "tender_date": record.get("tender_date"),
            "result_status": record.get("result_status"),
            "winning_company": normalize_company_name(record.get("winning_company")),
            "contract_value_try": record.get("contract_value_try"),
            "estimated_cost_try": record.get("estimated_cost_try"),
            "contract_date": record.get("contract_date"),
            "highest_bid_try": record.get("highest_bid_try"),
            "lowest_bid_try": record.get("lowest_bid_try"),
            "termination_status": record.get("termination_status"),
            "transfer_status": record.get("transfer_status"),
            "item_details_available": record.get("item_details_available"),
            "item_name": record.get("item_name"),
            "quantity": record.get("quantity"),
            "unit": record.get("unit"),
            "item_contract_value_try": record.get("item_contract_value_try"),
            "item_unit_price_try": record.get("item_unit_price_try") or _unit_price(record),
            "source_url": record.get("source_url"),
            "source_type": record.get("source_type"),
            "notes": record.get("notes"),
            "extraction_confidence": record.get("extraction_confidence"),
        }
        records.append(normalized)

    return {
        "selected_company": next((row for row in ranking if row.get("selected")), {"company": selected_company}),
        "records": records,
        "synthetic_fields_for_demo": [],
        "audit": {
            "normalized_record_count": len(records),
            "unit_price_calculations": sum(1 for record in records if record.get("item_unit_price_try") is not None),
            "official_only": True,
            "synthetic_values_in_official_records": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SCRAPE_OUTPUT)
    parser.add_argument("--ranking", type=Path, default=RANKING_UNIQUE_TENDERS_OUTPUT)
    parser.add_argument("--output", type=Path, default=NORMALIZED_OUTPUT)
    args = parser.parse_args()
    payload = normalize_ekap_records(args.input, args.ranking)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
