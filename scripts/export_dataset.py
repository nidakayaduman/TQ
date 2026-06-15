"""Run the public-source pipeline and export JSON, CSV, and source audit."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from discover_sources import CANDIDATES, DATA_DIR, DISCOVERED_PATH, discover_sources  # noqa: E402
from normalize_records import NORMALIZED_RECORDS_PATH, normalize_records  # noqa: E402
from scrape_tender_records import RAW_RECORDS_PATH, scrape_tender_records  # noqa: E402


FINAL_JSON_PATH = DATA_DIR / "company_tender_records.json"
FINAL_CSV_PATH = DATA_DIR / "company_tender_records.csv"
SOURCE_AUDIT_PATH = DATA_DIR / "source_audit.json"


def select_company(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["winning_company"]] = counts.get(record["winning_company"], 0) + 1
    if not counts:
        raise RuntimeError("No usable won tender records found.")
    selected = max(counts.items(), key=lambda item: item[1])
    return {
        "company_name": selected[0],
        "sector": "Pharmaceuticals / serum / medical supply",
        "reason_for_selection": (
            "Selected because it had the highest number of usable public won-tender "
            "records among the checked pharma, serum, and medical-supply candidates."
        ),
        "usable_record_count": selected[1],
        "sources_checked": ["KAP public disclosure search/detail pages"],
    }


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_source_audit(discovered: dict, scraped: dict, normalized: dict, selected_company: dict) -> dict:
    scraped_counts = scraped["audit"].get("records_extracted_per_company", {})
    skipped_by_company: dict[str, list[str]] = {}
    for item in discovered["audit"].get("skipped_urls", []):
        skipped_by_company.setdefault(item.get("candidate_company", "Unknown"), []).append(item.get("reason", ""))

    candidate_summary = []
    for candidate in CANDIDATES:
        usable_count = scraped_counts.get(candidate.canonical_name, 0)
        reasons = skipped_by_company.get(candidate.canonical_name, [])
        if candidate.canonical_name == selected_company["company_name"]:
            status = "selected"
            notes = "Highest usable public won-tender record count in this run."
        elif any("429" in reason for reason in reasons):
            status = "rate_limited_or_partial"
            notes = "KAP returned HTTP 429 for at least one query before deeper collection could complete."
        elif usable_count:
            status = "usable_not_selected"
            notes = "Usable records found, but count was below the selected company."
        else:
            status = "no_usable_kap_records_found_in_run"
            notes = "No usable public won-tender KAP records were extracted in this run."
        candidate_summary.append(
            {
                "company": candidate.canonical_name,
                "usable_record_count": usable_count,
                "status": status,
                "notes": notes,
            }
        )

    return {
        "collection_run": {
            "pipeline": "Tender Intelligence MVP public-source collection",
            "selected_company": selected_company["company_name"],
            "selected_company_usable_records": selected_company["usable_record_count"],
        },
        "discovery": discovered["audit"],
        "candidate_evaluation_summary": candidate_summary,
        "scraping": scraped["audit"],
        "normalization": normalized["audit"],
        "compliance": {
            "captcha_login_paywall_bypass": "No bypass attempted.",
            "polite_headers_and_delays": "Requests use an identifying User-Agent and configurable delay between requests.",
            "source_priority_handling": (
                "KAP was used as the cleanest official public source for this MVP run. "
                "EKAP/DMO direct scraping was not pursued after KAP provided structured official records."
            ),
        },
        "data_gaps": [
            "Lost tender records are not available in this public-source dataset.",
            "Most disclosures do not include quantity, unit, product SKU, or unit price.",
            "Contract values are usually disclosed as total TRY amounts, often plus VAT.",
            "Buyer institution detail is limited to the tender authority text present in KAP.",
            "Synthetic cost, margin, and inflation fields are demo-only and not source facts.",
        ],
    }


def export_dataset(max_pages_per_query: int = 15, delay_seconds: float = 0.6) -> dict:
    DATA_DIR.mkdir(exist_ok=True)

    discovered = discover_sources(
        max_pages_per_query=max_pages_per_query,
        delay_seconds=delay_seconds,
    )
    DISCOVERED_PATH.write_text(json.dumps(discovered, ensure_ascii=False, indent=2), encoding="utf-8")

    scraped = scrape_tender_records(DISCOVERED_PATH, delay_seconds=delay_seconds)
    RAW_RECORDS_PATH.write_text(json.dumps(scraped, ensure_ascii=False, indent=2), encoding="utf-8")

    normalized = normalize_records(RAW_RECORDS_PATH)
    NORMALIZED_RECORDS_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_company = select_company(normalized["records"])
    selected_records = [
        record for record in normalized["records"] if record["winning_company"] == selected_company["company_name"]
    ]
    selected_ids = {record["tender_id"] for record in selected_records}
    selected_synthetic = [
        item for item in normalized["synthetic_fields_for_demo"] if item["tender_id"] in selected_ids
    ]

    final_payload = {
        "selected_company": selected_company,
        "records": selected_records,
        "synthetic_fields_for_demo": selected_synthetic,
    }
    FINAL_JSON_PATH.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(selected_records, FINAL_CSV_PATH)

    source_audit = build_source_audit(discovered, scraped, normalized, selected_company)
    SOURCE_AUDIT_PATH.write_text(json.dumps(source_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "final": final_payload,
        "source_audit": source_audit,
        "paths": {
            "json": str(FINAL_JSON_PATH),
            "csv": str(FINAL_CSV_PATH),
            "audit": str(SOURCE_AUDIT_PATH),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages-per-query", type=int, default=15)
    parser.add_argument("--delay-seconds", type=float, default=0.6)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    payload = export_dataset(args.max_pages_per_query, args.delay_seconds)
    final = payload["final"]
    logging.info("Selected company: %s", final["selected_company"]["company_name"])
    logging.info("Usable records: %s", final["selected_company"]["usable_record_count"])
    logging.info("Wrote %s", payload["paths"])


if __name__ == "__main__":
    main()
