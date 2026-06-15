"""Normalize raw tender records into the demo dataset schema."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_RECORDS_PATH = DATA_DIR / "raw_tender_records.json"
NORMALIZED_RECORDS_PATH = DATA_DIR / "normalized_records.json"


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def parse_date(value: str | None) -> str | None:
    value = clean_text(value)
    if not value or value == "-":
        return None
    try:
        dt = date_parser.parse(value, dayfirst=True)
    except (ValueError, TypeError):
        return None
    return dt.date().isoformat()


def parse_try_amount(value: str | None) -> float | None:
    value = clean_text(value)
    if not value:
        return None
    match = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?)", value)
    if not match:
        return None
    number = match.group(1).replace(".", "").replace(",", ".")
    try:
        return round(float(number), 2)
    except ValueError:
        return None


def normalize_company_name(value: str | None, fallback: str) -> str:
    text = clean_text(value)
    folded = text.casefold()
    if "turk ilaç" in folded or "türk ilaç" in folded or "turk ilac" in folded:
        return "Türk İlaç ve Serum Sanayi A.Ş."
    if "oncosem" in folded:
        return "Oncosem Onkolojik Sistemler Sanayi ve Ticaret A.Ş."
    if "deva" in folded:
        return "Deva Holding A.Ş."
    if "gen ilaç" in folded or "gen ilac" in folded:
        return "Gen İlaç ve Sağlık Ürünleri San. ve Tic. A.Ş."
    return fallback


def infer_product_group(text: str) -> str:
    folded = text.casefold()
    if "serum" in folded:
        return "Serum / IV solution"
    if "sağlık market" in folded or "saglik market" in folded:
        return "Medical supply / Sağlık Market"
    if "onkoloji" in folded or "kemoterapi" in folded:
        return "Oncology medical supply"
    if "ilaç" in folded or "ilac" in folded:
        return "Pharmaceuticals"
    return "Medical supply"


def infer_brand(text: str) -> str:
    for brand in ("TURKFLEKS", "TURK FLEKS", "TURKFLEX"):
        if brand.casefold() in text.casefold():
            return "TURKFLEKS"
    return ""


def extraction_confidence(source: str, amount: float | None, company: str) -> str:
    if source == "KAP" and amount is not None and company:
        return "high"
    if amount is not None and company:
        return "medium"
    return "low"


def make_tender_id(source_url: str) -> str:
    match = re.search(r"/Bildirim/(\d+)", source_url)
    if match:
        return f"KAP-{match.group(1)}"
    return f"SRC-{abs(hash(source_url))}"


def normalize_record(raw: dict[str, Any]) -> dict:
    fields = raw["raw_fields"]
    source_date = parse_date(raw.get("source_date_text"))
    tender_result_date = parse_date(fields.get("İhalenin Sonuçlandığı Tarih"))
    tender_name = clean_text(fields.get("İhale Konusu")) or clean_text(raw.get("disclosure_title"))
    explanation = clean_text(fields.get("Açıklamalar"))
    authority = clean_text(fields.get("İhaleyi Açan Taraf"))
    value_raw = fields.get("İhale Bedelinden Ortaklık Payına Düşen Kısım") or fields.get("İhale Bedeli")
    amount = parse_try_amount(value_raw)
    company = normalize_company_name(raw.get("company_label"), raw["candidate_company"])
    note_parts = []
    if value_raw:
        note_parts.append(f"Raw tender value text: {value_raw}")
    if fields.get("Grup Olarak İhaleye Girilmesi Halinde Diğer Taraflar"):
        note_parts.append(f"Group tender party: {fields['Grup Olarak İhaleye Girilmesi Halinde Diğer Taraflar']}")
    if explanation:
        note_parts.append(explanation)
    notes = " | ".join(note_parts)
    all_text = f"{tender_name} {authority} {explanation}"

    return {
        "tender_id": make_tender_id(raw["source_url"]),
        "source": raw["source"],
        "source_url": raw["source_url"],
        "source_date": source_date,
        "announcement_date": source_date,
        "tender_result_date": tender_result_date,
        "buyer_institution": authority,
        "tender_authority": authority,
        "tender_name": tender_name,
        "product_group": infer_product_group(all_text),
        "product_name": "",
        "brand_name": infer_brand(all_text),
        "region": "Turkey",
        "quantity": None,
        "unit": "",
        "contract_value_try": amount,
        "contract_value_vat_excluded_try": amount if value_raw and "KDV" in value_raw.upper() else None,
        "winning_company": company,
        "result": "won",
        "notes": notes,
        "extraction_confidence": extraction_confidence(raw["source"], amount, company),
    }


def dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for record in records:
        key = (
            clean_text(record.get("tender_name")).casefold(),
            record.get("tender_result_date"),
            record.get("contract_value_try"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def synthetic_fields(records: list[dict]) -> list[dict]:
    current_year = 2026
    synthetic: list[dict] = []
    for record in records:
        result_date = record.get("tender_result_date") or record.get("source_date")
        inflation_factor = None
        if result_date:
            try:
                year = datetime.fromisoformat(result_date).year
                inflation_factor = round(1.18 ** max(current_year - year, 0), 4)
            except ValueError:
                inflation_factor = None
        total = record.get("contract_value_try")
        quantity = record.get("quantity")
        estimated_unit_price = round(total / quantity, 4) if total and quantity else None
        inflation_adjusted = (
            round(estimated_unit_price * inflation_factor, 4)
            if estimated_unit_price is not None and inflation_factor is not None
            else None
        )
        synthetic.append(
            {
                "tender_id": record["tender_id"],
                "estimated_unit_price": estimated_unit_price,
                "estimated_unit_cost": None,
                "gross_margin_pct": None,
                "inflation_factor_to_2026": inflation_factor,
                "inflation_adjusted_unit_price": inflation_adjusted,
                "synthetic_data_note": "Generated for demo only; not from source.",
            }
        )
    return synthetic


def normalize_records(raw_path: Path = RAW_RECORDS_PATH) -> dict:
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    records = [normalize_record(item) for item in raw_payload["records"]]
    records = dedupe_records(records)
    records.sort(key=lambda item: (item.get("tender_result_date") or "", item["tender_id"]), reverse=True)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["winning_company"]] = counts.get(record["winning_company"], 0) + 1

    return {
        "records": records,
        "synthetic_fields_for_demo": synthetic_fields(records),
        "audit": {
            "normalized_record_count": len(records),
            "deduped_record_count_by_company": counts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_RECORDS_PATH)
    parser.add_argument("--output", type=Path, default=NORMALIZED_RECORDS_PATH)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    payload = normalize_records(args.input)
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Normalized %s records", payload["audit"]["normalized_record_count"])
    logging.info("Record counts by company: %s", payload["audit"]["deduped_record_count_by_company"])
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
