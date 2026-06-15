"""Discover public tender-result source URLs for the MVP dataset."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.kap.org.tr"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DISCOVERED_PATH = DATA_DIR / "discovered_sources.json"

REQUEST_HEADERS = {
    "User-Agent": "TenderIntelligenceMVP/0.1 public-source-demo",
    "Accept": "text/html",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.7",
}


@dataclass(frozen=True)
class Candidate:
    canonical_name: str
    sector: str
    aliases: tuple[str, ...]
    queries: tuple[str, ...]


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        canonical_name="Türk İlaç ve Serum Sanayi A.Ş.",
        sector="Pharmaceuticals / serum / medical supply",
        aliases=("TRILC", "TURK İLAÇ", "TÜRK İLAÇ", "TURK ILAC", "SERUM"),
        queries=(
            "TRILC ihale",
            "TURK İLAÇ VE SERUM ihale",
            "Türk İlaç Sağlık Market Alım İhalesi",
            "ihale sonucu serum",
        ),
    ),
    Candidate(
        canonical_name="Polifarma İlaç San. ve Tic. A.Ş.",
        sector="Pharmaceuticals / hospital products",
        aliases=("POLİFARMA", "POLIFARMA"),
        queries=("Polifarma ihale", "Polifarma kazandı ihale", "Polifarma DMO ihale"),
    ),
    Candidate(
        canonical_name="Vem İlaç San. ve Tic. A.Ş.",
        sector="Pharmaceuticals",
        aliases=("VEM İLAÇ", "VEM ILAC"),
        queries=("Vem İlaç ihale", "Vem İlaç kazandı ihale", "Vem İlaç DMO"),
    ),
    Candidate(
        canonical_name="Deva Holding A.Ş.",
        sector="Pharmaceuticals",
        aliases=("DEVA HOLDİNG", "DEVA HOLDING", "DEVA"),
        queries=("DEVA ihale", "DEVA HOLDİNG ihale", "DEVA ilaç ihale sonucu"),
    ),
    Candidate(
        canonical_name="Atabay Kimya Sanayi ve Ticaret A.Ş.",
        sector="Pharmaceuticals",
        aliases=("ATABAY",),
        queries=("Atabay ihale", "Atabay ilaç ihale", "Atabay kazandı ihale"),
    ),
    Candidate(
        canonical_name="Santa Farma İlaç Sanayii A.Ş.",
        sector="Pharmaceuticals",
        aliases=("SANTA FARMA",),
        queries=("Santa Farma ihale", "Santa Farma kazandı ihale", "Santa Farma DMO"),
    ),
    Candidate(
        canonical_name="Oncosem Onkolojik Sistemler Sanayi ve Ticaret A.Ş.",
        sector="Oncology / medical supply systems",
        aliases=("ONCOSEM", "ONCSM"),
        queries=("ONCOSEM ihale", "ONCSM ihale", "Oncosem ihale sonucu"),
    ),
    Candidate(
        canonical_name="Gen İlaç ve Sağlık Ürünleri San. ve Tic. A.Ş.",
        sector="Pharmaceuticals / health products",
        aliases=("GEN İLAÇ", "GEN ILAC", "GENIL"),
        queries=("GENIL ihale", "Gen İlaç ihale", "Gen İlaç ihale sonucu"),
    ),
)


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def candidate_matches(candidate: Candidate, text: str) -> bool:
    folded = normalize_text(text)
    return any(normalize_text(alias) in folded for alias in candidate.aliases)


def fetch_html(session: requests.Session, url: str, delay_seconds: float) -> str:
    logging.debug("Fetching %s", url)
    response = session.get(url, headers=REQUEST_HEADERS, timeout=30)
    if response.status_code != 200:
        raise requests.HTTPError(f"unexpected HTTP status {response.status_code} for {url}")
    response.raise_for_status()
    time.sleep(delay_seconds)
    return response.text


def kap_search_url(query: str, page: int) -> str:
    return f"{BASE_URL}/tr/search/{quote(query)}/{page}"


def parse_kap_search_results(
    html: str,
    source_url: str,
    candidate: Candidate,
) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []
    for link in soup.select("a[href^='/tr/Bildirim/']"):
        title = link.contents[0].strip() if link.contents else link.get_text(" ", strip=True)
        company_tag = link.select_one("span.block")
        company = company_tag.get_text(" ", strip=True) if company_tag else ""
        card_text = link.find_parent("div", class_="lg:flex")
        surrounding_text = card_text.get_text(" ", strip=True) if card_text else link.get_text(" ", strip=True)

        if "İhale Süreci / Sonucu" not in title:
            continue
        if not candidate_matches(candidate, f"{company} {surrounding_text}"):
            continue

        date_text = None
        if card_text:
            parts = card_text.get_text(" ", strip=True).split("Gönderim Tarihi")
            if len(parts) > 1:
                date_text = parts[-1].strip()[:19]

        href = link.get("href", "")
        results.append(
            {
                "candidate_company": candidate.canonical_name,
                "candidate_sector": candidate.sector,
                "source": "KAP",
                "source_type": "official_disclosure_search_result",
                "title": title,
                "company_label": company,
                "url": urljoin(BASE_URL, href),
                "source_search_url": source_url,
                "source_date_text": date_text,
                "snippet": surrounding_text[:500],
            }
        )
    return results


def discover_sources(
    candidates: Iterable[Candidate] = CANDIDATES,
    max_pages_per_query: int = 15,
    delay_seconds: float = 0.6,
) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    session = requests.Session()
    urls_checked = 0
    skipped_urls: list[dict] = []
    all_results: list[dict] = []
    candidates = tuple(candidates)
    query_state: dict[tuple[str, str], dict] = {}
    seen_urls_by_company: dict[str, set[str]] = {candidate.canonical_name: set() for candidate in candidates}
    for candidate in candidates:
        for query in candidate.queries:
            query_state[(candidate.canonical_name, query)] = {
                "candidate": candidate,
                "query": query,
                "empty_pages": 0,
                "done": False,
            }

    rate_limited = False
    for page in range(1, max_pages_per_query + 1):
        if rate_limited:
            break
        for state in query_state.values():
            if state["done"]:
                continue
            candidate = state["candidate"]
            query = state["query"]
            url = kap_search_url(query, page)
            try:
                html = fetch_html(session, url, delay_seconds)
                urls_checked += 1
            except requests.RequestException as exc:
                skipped_urls.append(
                    {
                        "url": url,
                        "candidate_company": candidate.canonical_name,
                        "reason": f"request_failed: {exc}",
                    }
                )
                if "429" in str(exc):
                    rate_limited = True
                    break
                state["done"] = True
                continue

            results = parse_kap_search_results(html, url, candidate)
            seen_candidate_urls = seen_urls_by_company[candidate.canonical_name]
            new_results = [item for item in results if item["url"] not in seen_candidate_urls]
            for item in new_results:
                seen_candidate_urls.add(item["url"])
            all_results.extend(new_results)

            if not results:
                state["empty_pages"] += 1
            else:
                state["empty_pages"] = 0
            if state["empty_pages"] >= 2:
                state["done"] = True

    by_company: dict[str, int] = {}
    for item in all_results:
        by_company[item["candidate_company"]] = by_company.get(item["candidate_company"], 0) + 1

    audit = {
        "urls_checked": urls_checked,
        "candidate_counts_from_discovery": by_company,
        "skipped_urls": skipped_urls,
        "source_notes": [
            {
                "source": "KAP",
                "status": "used",
                "note": "Public KAP search and disclosure detail pages were accessible without login or captcha during collection.",
            },
            {
                "source": "EKAP / KIK",
                "status": "not_scraped",
                "note": "Not used for the first demo run because KAP disclosures already provided official structured tender-result fields. No captcha, login, or anti-bot bypass was attempted.",
            },
            {
                "source": "DMO / Sağlık Market",
                "status": "referenced_via_kap",
                "note": "DMO Sağlık Market appears as the tender authority in KAP records; direct DMO scraping was not required for the cleaner MVP dataset.",
            },
        ],
    }
    return {"results": all_results, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages-per-query", type=int, default=15)
    parser.add_argument("--delay-seconds", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=DISCOVERED_PATH)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    payload = discover_sources(
        max_pages_per_query=args.max_pages_per_query,
        delay_seconds=args.delay_seconds,
    )
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Checked %s URLs", payload["audit"]["urls_checked"])
    logging.info("Discovered records by company: %s", payload["audit"]["candidate_counts_from_discovery"])
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
