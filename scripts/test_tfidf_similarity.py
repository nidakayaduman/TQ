"""Test an explainable TF-IDF similar tender retrieval module.

This script is intentionally standalone. It does not modify the Streamlit app
and does not use FAISS, sentence transformers, external APIs, or quantity.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = Path(
    "/Users/kayaduman/Downloads/polifarma_synthetic_tender_dataset_2021_2025/"
    "polifarma_synthetic_tenders_2021_2025.csv"
)
RESULTS_PATH = REPO_ROOT / "data" / "tfidf_similarity_test_results.json"
REPORT_PATH = REPO_ROOT / "TFIDF_SIMILARITY_REPORT.md"

HISTORICAL_TEXT_FIELDS = [
    "tender_title",
    "product_name",
    "product_group",
    "buyer_institution",
    "region",
    "procedure_type",
]
QUERY_TEXT_FIELDS = ["product_name", "product_group", "region", "procedure_type"]
OUTPUT_FIELDS = [
    "tender_id",
    "year",
    "tender_title",
    "product_name",
    "product_group",
    "region",
    "procedure_type",
    "winning_unit_price_try",
    "inflation_adjusted_unit_price_2025_try",
]

WEIGHTS = {
    "tfidf": 0.60,
    "product_group": 0.15,
    "product_name": 0.10,
    "region": 0.10,
    "procedure_type": 0.05,
}

TEST_CASES = [
    {
        "product_name": "%0.9 NaCl 500 ml",
        "product_group": "IV Solution",
        "region": "Marmara",
        "procedure_type": "Açık İhale",
    },
    {
        "product_name": "%5 Dekstroz 500 ml",
        "product_group": "IV Solution",
        "region": "Ege",
        "procedure_type": "Açık İhale",
    },
    {
        "product_name": "Levofloksasin IV 500 mg/100 ml",
        "product_group": "Injectable",
        "region": "Akdeniz",
        "procedure_type": "Pazarlık (MD 21 B)",
    },
    {
        "product_name": "Sodyum Bikarbonat Ampul 10 ml",
        "product_group": "Injectable",
        "region": "İç Anadolu",
        "procedure_type": "Açık İhale",
    },
    {
        "product_name": "Periton Diyaliz Solüsyonu 2000 ml",
        "product_group": "Special Solution",
        "region": "Ege",
        "procedure_type": "Açık İhale",
    },
]


def normalize_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> set[str]:
    text = normalize_text(value)
    return set(re.findall(r"[\w.%]+", text, flags=re.UNICODE))


def combine_text(record: pd.Series | dict[str, Any], fields: list[str]) -> str:
    return " | ".join(normalize_text(record.get(field, "")) for field in fields)


def exact_match_score(left: Any, right: Any) -> float:
    return 1.0 if normalize_text(left) == normalize_text(right) else 0.0


def product_name_score(query_name: str, candidate_name: str) -> float:
    query_normalized = normalize_text(query_name)
    candidate_normalized = normalize_text(candidate_name)
    if not query_normalized or not candidate_normalized:
        return 0.0
    if query_normalized == candidate_normalized:
        return 1.0
    if query_normalized in candidate_normalized or candidate_normalized in query_normalized:
        return 1.0

    query_tokens = tokenize(query_name)
    candidate_tokens = tokenize(candidate_name)
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = len(query_tokens & candidate_tokens)
    return round(overlap / max(len(query_tokens), len(candidate_tokens)), 4)


def final_score(
    tfidf_score: float,
    product_group: float,
    product_name: float,
    region: float,
    procedure_type: float,
) -> float:
    return (
        WEIGHTS["tfidf"] * tfidf_score
        + WEIGHTS["product_group"] * product_group
        + WEIGHTS["product_name"] * product_name
        + WEIGHTS["region"] * region
        + WEIGHTS["procedure_type"] * procedure_type
    )


def retrieve_similar_tenders(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    query: dict[str, str],
    candidate_count: int = 30,
    result_count: int = 10,
) -> list[dict[str, Any]]:
    query_text = combine_text(query, QUERY_TEXT_FIELDS)
    query_vector = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vector, matrix)[0]
    top_indices = scores.argsort()[::-1][:candidate_count]

    candidates: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[int(idx)]
        tfidf = round(float(scores[idx]), 6)
        group_score = exact_match_score(query["product_group"], row["product_group"])
        name_score = product_name_score(query["product_name"], row["product_name"])
        region_score = exact_match_score(query["region"], row["region"])
        procedure_score = exact_match_score(query["procedure_type"], row["procedure_type"])
        hybrid = round(
            final_score(tfidf, group_score, name_score, region_score, procedure_score),
            6,
        )

        item = {field: row[field].item() if hasattr(row[field], "item") else row[field] for field in OUTPUT_FIELDS}
        item.update(
            {
                "initial_tfidf_rank": rank,
                "tfidf_score": tfidf,
                "product_group_score": group_score,
                "product_name_score": name_score,
                "region_score": region_score,
                "procedure_type_score": procedure_score,
                "final_similarity_score": hybrid,
            }
        )
        candidates.append(item)

    return sorted(
        candidates,
        key=lambda item: (
            item["final_similarity_score"],
            item["tfidf_score"],
            item["product_name_score"],
            item["region_score"],
            item["procedure_type_score"],
        ),
        reverse=True,
    )[:result_count]


def summarize_quality(query: dict[str, str], results: list[dict[str, Any]]) -> dict[str, Any]:
    top_product_group_matches = sum(
        1 for item in results if normalize_text(item["product_group"]) == normalize_text(query["product_group"])
    )
    top_product_name_exact = sum(
        1 for item in results if normalize_text(item["product_name"]) == normalize_text(query["product_name"])
    )
    top_product_name_partial = sum(1 for item in results if item["product_name_score"] > 0)
    top_region_matches = sum(1 for item in results if normalize_text(item["region"]) == normalize_text(query["region"]))
    top_procedure_matches = sum(
        1 for item in results if normalize_text(item["procedure_type"]) == normalize_text(query["procedure_type"])
    )
    return {
        "top_10_product_group_matches": top_product_group_matches,
        "top_10_product_name_exact_matches": top_product_name_exact,
        "top_10_product_name_exact_or_partial_matches": top_product_name_partial,
        "top_10_region_matches": top_region_matches,
        "top_10_procedure_type_matches": top_procedure_matches,
        "passes_mvp_quality_bar": top_product_group_matches >= 8 and top_product_name_partial >= 8,
    }


def build_results(source_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(source_csv)
    missing_fields = [field for field in HISTORICAL_TEXT_FIELDS + OUTPUT_FIELDS if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing required columns: {missing_fields}")

    searchable_text = df.apply(lambda row: combine_text(row, HISTORICAL_TEXT_FIELDS), axis=1)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(searchable_text)

    test_results = []
    quality_checks = []
    for index, query in enumerate(TEST_CASES, start=1):
        retrieved = retrieve_similar_tenders(df, vectorizer, matrix, query)
        quality = summarize_quality(query, retrieved)
        test_results.append(
            {
                "test_case_id": index,
                "query": query,
                "top_10_retrieved_tenders": retrieved,
                "quality_check": quality,
            }
        )
        quality_checks.append(quality)

    all_cases_pass = all(item["passes_mvp_quality_bar"] for item in quality_checks)
    return {
        "module": "TF-IDF + cosine similarity similar tender retrieval",
        "source_file": str(source_csv),
        "dataset_row_count": int(len(df)),
        "historical_searchable_text_fields": HISTORICAL_TEXT_FIELDS,
        "query_text_fields": QUERY_TEXT_FIELDS,
        "candidate_retrieval": {
            "method": "sklearn TfidfVectorizer + cosine_similarity",
            "candidate_count_by_text_similarity": 30,
            "final_result_count_after_business_rerank": 10,
            "quantity_used": False,
        },
        "weights": WEIGHTS,
        "component_score_notes": {
            "tfidf_score": "Raw cosine similarity from TF-IDF vectors, 0 to 1.",
            "product_group_score": "Raw exact-match score, 0 or 1.",
            "product_name_score": "Raw score: 1 for exact or substring match, otherwise token-overlap ratio from 0 to 1.",
            "region_score": "Raw exact-match score, 0 or 1.",
            "procedure_type_score": "Raw exact-match score, 0 or 1.",
            "final_similarity_score": "Weighted score using the configured weights, 0 to 1.",
        },
        "test_cases": test_results,
        "overall_quality": {
            "all_cases_pass_mvp_quality_bar": all_cases_pass,
            "average_top_10_product_group_matches": round(
                sum(item["top_10_product_group_matches"] for item in quality_checks) / len(quality_checks),
                2,
            ),
            "average_top_10_product_name_exact_or_partial_matches": round(
                sum(item["top_10_product_name_exact_or_partial_matches"] for item in quality_checks)
                / len(quality_checks),
                2,
            ),
            "average_top_10_region_matches": round(
                sum(item["top_10_region_matches"] for item in quality_checks) / len(quality_checks),
                2,
            ),
            "average_top_10_procedure_type_matches": round(
                sum(item["top_10_procedure_type_matches"] for item in quality_checks) / len(quality_checks),
                2,
            ),
            "streamlit_mvp_assessment": (
                "Good enough for the Streamlit MVP: retrieval consistently prioritizes matching product group "
                "and exact or closely related product names, while region and procedure type influence ranking "
                "without dominating it."
                if all_cases_pass
                else "Not yet good enough for the Streamlit MVP: at least one test case fails the product-context quality bar."
            ),
        },
        "future_enhancement_options": [
            "multilingual sentence embeddings",
            "FAISS/vector database",
            "hybrid semantic + structured retrieval",
            "learned ranking using real client feedback",
        ],
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def build_report(results: dict[str, Any]) -> str:
    overall = results["overall_quality"]
    lines = [
        "# TF-IDF Similar Tender Retrieval Report",
        "",
        f"**Source file:** `{results['source_file']}`",
        f"**Dataset rows:** {results['dataset_row_count']}",
        "",
        "## Method",
        "",
        "- Historical searchable text uses `tender_title`, `product_name`, `product_group`, `buyer_institution`, `region`, and `procedure_type`.",
        "- Query text uses `product_name`, `product_group`, `region`, and `procedure_type`.",
        "- The first pass retrieves the top 30 candidates with `sklearn` `TfidfVectorizer` and `cosine_similarity`.",
        "- The second pass re-ranks candidates with the requested business rules.",
        "- Quantity is not used in this version.",
        "",
        "## Hybrid Score",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Component", "Weight", "Score behavior"],
            [
                ["TF-IDF cosine similarity", "60%", "Raw cosine similarity from 0 to 1"],
                ["Product group exact match", "15%", "1 for exact match, otherwise 0"],
                ["Product name exact or partial match", "10%", "1 for exact/substring match, otherwise token-overlap ratio"],
                ["Region exact match", "10%", "1 for exact match, otherwise 0"],
                ["Procedure type exact match", "5%", "1 for exact match, otherwise 0"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Quality Summary",
            "",
            f"- All test cases pass MVP quality bar: `{overall['all_cases_pass_mvp_quality_bar']}`",
            f"- Average top-10 product group matches: {overall['average_top_10_product_group_matches']}",
            f"- Average top-10 exact or partial product-name matches: {overall['average_top_10_product_name_exact_or_partial_matches']}",
            f"- Average top-10 region matches: {overall['average_top_10_region_matches']}",
            f"- Average top-10 procedure-type matches: {overall['average_top_10_procedure_type_matches']}",
            f"- Streamlit MVP assessment: {overall['streamlit_mvp_assessment']}",
            "",
            "Region and procedure type are visible in the ranking, but the product context dominates as intended.",
            "",
            "## Test Case Results",
            "",
        ]
    )

    for case in results["test_cases"]:
        query = case["query"]
        quality = case["quality_check"]
        lines.extend(
            [
                f"### Test Case {case['test_case_id']}",
                "",
                f"**Query:** `{query['product_name']}` / `{query['product_group']}` / `{query['region']}` / `{query['procedure_type']}`",
                "",
                f"- Product group matches in top 10: {quality['top_10_product_group_matches']}",
                f"- Exact product-name matches in top 10: {quality['top_10_product_name_exact_matches']}",
                f"- Exact or partial product-name matches in top 10: {quality['top_10_product_name_exact_or_partial_matches']}",
                f"- Region matches in top 10: {quality['top_10_region_matches']}",
                f"- Procedure-type matches in top 10: {quality['top_10_procedure_type_matches']}",
                "",
            ]
        )
        rows = []
        for rank, item in enumerate(case["top_10_retrieved_tenders"], start=1):
            rows.append(
                [
                    rank,
                    item["tender_id"],
                    item["year"],
                    item["tender_title"],
                    item["product_name"],
                    item["product_group"],
                    item["region"],
                    item["procedure_type"],
                    item["winning_unit_price_try"],
                    item["inflation_adjusted_unit_price_2025_try"],
                    item["tfidf_score"],
                    item["product_group_score"],
                    item["product_name_score"],
                    item["region_score"],
                    item["procedure_type_score"],
                    item["final_similarity_score"],
                ]
            )
        lines.extend(
            markdown_table(
                [
                    "Rank",
                    "Tender ID",
                    "Year",
                    "Tender title",
                    "Product",
                    "Group",
                    "Region",
                    "Procedure",
                    "Winning unit price TRY",
                    "Inflation adjusted unit price 2025 TRY",
                    "TF-IDF",
                    "Group score",
                    "Product score",
                    "Region score",
                    "Procedure score",
                    "Final score",
                ],
                rows,
            )
        )
        lines.append("")

    lines.extend(
        [
            "## MVP Positioning",
            "",
            "This is an MVP-grade, explainable similarity engine. It is lightweight and suitable for Streamlit Community Cloud because it uses only a small in-memory TF-IDF matrix and deterministic business-rule re-ranking.",
            "",
            "Future enhancement options:",
            "",
            "- multilingual sentence embeddings",
            "- FAISS/vector database",
            "- hybrid semantic + structured retrieval",
            "- learned ranking using real client feedback",
            "",
            "No Streamlit app changes, price corridor, margin simulation, FAISS, sentence-transformers, or external APIs were used.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    args = parser.parse_args()

    results = build_results(args.source_csv)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")

    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(results["overall_quality"]["streamlit_mvp_assessment"])


if __name__ == "__main__":
    main()
