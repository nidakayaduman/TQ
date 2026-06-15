"""Scrape EKAP completed-tender detail records for POC and selected supplier."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

from discover_companies import (
    DATA_DIR,
    DISCOVERY_OUTPUT,
    extract_contract_records,
    fetch_tender_detail,
    search_completed_tenders,
)


POC_OUTPUT = DATA_DIR / "ekap_poc_records.json"
SCRAPE_OUTPUT = DATA_DIR / "ekap_scrape_results.json"


def scrape_poc_records(max_records: int = 5, delay_seconds: float = 0.4, seed_records: list[dict] | None = None) -> dict:
    session = requests.Session()
    records: list[dict] = []
    checked_rows: list[dict] = []
    blocked_pages: list[dict] = []

    if seed_records:
        eligible_seed_records = [
            record
            for record in seed_records
            if record.get("contract_value_try") is not None and record.get("estimated_cost_try") is not None
        ]
        eligible_seed_records.sort(key=lambda record: "ONKOFAR" in (record.get("winning_company") or ""), reverse=True)
        for record in eligible_seed_records:
            records.append(record)
            if len(records) >= max_records:
                return {
                    "purpose": "Proof of concept for EKAP completed tender contract extraction",
                    "onkofar_records_found": sum(1 for item in records if "ONKOFAR" in (item.get("winning_company") or "")),
                    "records": records,
                    "audit": {
                        "target_record_count": max_records,
                        "actual_record_count": len(records),
                        "checked_rows": [{"source": "seed_discovery_records", "records_available": len(seed_records)}],
                        "contract_value_populated": sum(1 for item in records if item.get("contract_value_try") is not None),
                        "estimated_cost_populated": sum(1 for item in records if item.get("estimated_cost_try") is not None),
                        "blocked_pages": blocked_pages,
                    },
                }

    # ONKOFAR validates the user-provided example path, but does not drive final selection.
    try:
        onkofar_rows = (search_completed_tenders(session, take=10, search_text="ONKOFAR").get("list") or [])
    except requests.RequestException as exc:
        onkofar_rows = []
        blocked_pages.append({"search_text": "ONKOFAR", "reason": str(exc)})

    candidate_rows = list(onkofar_rows)
    if len(candidate_rows) < max_records:
        try:
            broad_rows = search_completed_tenders(session, take=50).get("list") or []
            candidate_rows.extend([row for row in broad_rows if row.get("id") not in {item.get("id") for item in candidate_rows}])
        except requests.RequestException as exc:
            blocked_pages.append({"search_text": "__completed_broad__", "reason": str(exc)})

    for row in candidate_rows:
        if len(records) >= max_records:
            break
        try:
            detail = fetch_tender_detail(session, row)
            extracted = extract_contract_records(row, detail)
            checked_rows.append({"tender_id": row.get("ikn"), "records_extracted": len(extracted)})
            for record in extracted:
                if record.get("contract_value_try") is not None and record.get("estimated_cost_try") is not None:
                    records.append(record)
                    if len(records) >= max_records:
                        break
        except requests.RequestException as exc:
            blocked_pages.append({"tender_id": row.get("ikn"), "reason": str(exc)})
        time.sleep(delay_seconds)

    return {
        "purpose": "Proof of concept for EKAP completed tender contract extraction",
        "onkofar_records_found": sum(1 for record in records if "ONKOFAR" in (record.get("winning_company") or "")),
        "records": records,
        "audit": {
            "target_record_count": max_records,
            "actual_record_count": len(records),
            "checked_rows": checked_rows,
            "contract_value_populated": sum(1 for record in records if record.get("contract_value_try") is not None),
            "estimated_cost_populated": sum(1 for record in records if record.get("estimated_cost_try") is not None),
            "blocked_pages": blocked_pages,
        },
    }


def scrape_selected_company_records(selected_company: str, discovery_path: Path = DISCOVERY_OUTPUT) -> dict:
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in discovery.get("records", [])
        if (record.get("winning_company") or "").upper() == selected_company.upper()
    ]
    return {
        "selected_company": selected_company,
        "records": records,
        "audit": {
            "source_discovery": str(discovery_path),
            "records_for_selected_company": len(records),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-output", type=Path, default=POC_OUTPUT)
    parser.add_argument("--selected-company")
    parser.add_argument("--discovery-input", type=Path, default=DISCOVERY_OUTPUT)
    parser.add_argument("--output", type=Path, default=SCRAPE_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=0.4)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    args.poc_output.parent.mkdir(parents=True, exist_ok=True)
    poc = scrape_poc_records(delay_seconds=args.delay_seconds)
    args.poc_output.write_text(json.dumps(poc, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote POC records: %s (%s records)", args.poc_output, len(poc["records"]))

    if args.selected_company:
        payload = scrape_selected_company_records(args.selected_company, args.discovery_input)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote selected-company scrape: %s", args.output)


if __name__ == "__main__":
    main()
