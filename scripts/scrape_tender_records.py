"""Scrape structured tender-result records from discovered public URLs."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from discover_sources import DATA_DIR, DISCOVERED_PATH, REQUEST_HEADERS  # noqa: E402


RAW_RECORDS_PATH = DATA_DIR / "raw_tender_records.json"


FIELD_MAP = {
    "İhale Konusu": "tender_name",
    "İhaleyi Açan Taraf": "tender_authority",
    "İhaleye Teklif Verme Tarihi": "tender_bid_date",
    "İhalenin Sonuçlandığı Tarih": "tender_result_date",
    "İhale Sonucu": "tender_result_raw",
    "İhale Bedeli": "tender_value_raw",
    "İhale Bedelinden Ortaklık Payına Düşen Kısım": "company_share_value_raw",
    "Grup Olarak İhaleye Girilmesi Halinde Diğer Taraflar": "group_parties",
    "Grup Olarak İhaleye Girilmesi Halinde Ortaklığın Payı": "group_company_share_pct",
    "Açıklamalar": "explanation_heading",
    "": "",
}


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def extract_source_date(page_text: str, fallback: str | None) -> str | None:
    match = re.search(r"Gönderim Tarihi\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})", page_text)
    if match:
        return match.group(1)
    return fallback


def parse_detail_fields(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    fields: dict[str, str] = {}

    for row in soup.select("tr.data-input-row"):
        title_node = row.select_one(".taxonomy-field-title .content-tr")
        value_node = row.select_one(
            ".taxonomy-context-value.content-tr, "
            ".taxonomy-context-value-summernote.content-tr, "
            "td.taxonomy-context-value.col-order-class-3"
        )
        title = clean_text(title_node.get_text(" ", strip=True) if title_node else "")
        value = clean_text(value_node.get_text(" ", strip=True) if value_node else "")
        if not title and "text-block-value" in row.get("class", []):
            title = "Açıklamalar"
        if title and value:
            fields[title] = value

    explanation_node = soup.select_one(".text-block-value")
    if explanation_node:
        fields["Açıklamalar"] = clean_text(explanation_node.get_text(" ", strip=True))

    return fields


def parse_company_and_title(html: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(html, "lxml")
    page_text = soup.get_text(" ", strip=True)
    title = "İhale Süreci / Sonucu" if "İhale Süreci / Sonucu" in page_text else None
    company = None
    match = re.search(r"İhale Süreci / Sonucu\s+([A-ZÇĞİÖŞÜ0-9 .,&'-]+?)\s+Gönderim Tarihi", page_text)
    if match:
        company = clean_text(match.group(1))
    return company, title


def is_won_result(value: str) -> bool:
    folded = clean_text(value).casefold()
    if not folded:
        return False
    if "devam" in folded or "beklen" in folded:
        return False
    return "kazan" in folded or "uhdesinde kal" in folded


def scrape_record(session: requests.Session, source: dict[str, Any], delay_seconds: float) -> tuple[dict | None, dict | None]:
    url = source["url"]
    try:
        response = session.get(url, headers=REQUEST_HEADERS, timeout=30)
        if response.status_code != 200:
            raise requests.HTTPError(f"unexpected HTTP status {response.status_code} for {url}")
        response.raise_for_status()
        html = response.text
        time.sleep(delay_seconds)
    except requests.RequestException as exc:
        return None, {"url": url, "reason": f"request_failed: {exc}"}

    fields = parse_detail_fields(html)
    page_text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    company_from_page, title_from_page = parse_company_and_title(html)
    tender_result = fields.get("İhale Sonucu", "")

    if not is_won_result(tender_result):
        return None, {"url": url, "reason": f"not_a_won_tender_result: {tender_result or 'missing_result'}"}

    if not fields.get("İhale Bedeli") and not fields.get("İhale Konusu"):
        return None, {"url": url, "reason": "missing_core_tender_fields"}

    record = {
        "source_url": url,
        "source": source.get("source", "KAP"),
        "source_type": source.get("source_type"),
        "source_date_text": extract_source_date(page_text, source.get("source_date_text")),
        "candidate_company": source["candidate_company"],
        "candidate_sector": source["candidate_sector"],
        "company_label": company_from_page or source.get("company_label"),
        "disclosure_title": title_from_page or source.get("title"),
        "raw_fields": fields,
        "raw_page_excerpt": page_text[:1000],
    }
    return record, None


def scrape_tender_records(
    discovered_path: Path = DISCOVERED_PATH,
    delay_seconds: float = 0.6,
) -> dict:
    discovered = json.loads(discovered_path.read_text(encoding="utf-8"))
    session = requests.Session()
    records: list[dict] = []
    skipped: list[dict] = []

    seen_urls: set[str] = set()
    for source in discovered["results"]:
        if source["url"] in seen_urls:
            continue
        seen_urls.add(source["url"])
        record, skip = scrape_record(session, source, delay_seconds)
        if record:
            records.append(record)
        if skip:
            skipped.append(skip)

    by_company: dict[str, int] = {}
    for record in records:
        by_company[record["candidate_company"]] = by_company.get(record["candidate_company"], 0) + 1

    audit = {
        "urls_checked": len(seen_urls),
        "records_extracted_per_company": by_company,
        "skipped_urls": skipped,
    }
    return {"records": records, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DISCOVERED_PATH)
    parser.add_argument("--output", type=Path, default=RAW_RECORDS_PATH)
    parser.add_argument("--delay-seconds", type=float, default=0.6)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    payload = scrape_tender_records(args.input, args.delay_seconds)
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Checked %s detail URLs", payload["audit"]["urls_checked"])
    logging.info("Extracted records by company: %s", payload["audit"]["records_extracted_per_company"])
    logging.info("Skipped %s URLs", len(payload["audit"]["skipped_urls"]))
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
