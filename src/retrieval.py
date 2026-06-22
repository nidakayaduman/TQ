"""Similar won tender retrieval with local text embeddings and structural KNN signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .feature_masking import mask_actual_result_fields
from .schema import normalize_schema

TEXT_FIELDS = ["product_name", "product_group", "buyer_institution", "region", "procedure_type"]
PRICE_LIKE_FIELDS = {
    "estimated_unit_cost",
    "estimated_unit_cost_try",
    "internal_unit_cost_try",
    "estimated_total_cost_try",
    "actual_won_unit_price",
    "winning_unit_price_try",
    "actual_won_total_amount",
    "contract_value_try",
    "actual_margin_pct",
    "gross_margin_pct",
    "gross_profit_try",
    "discount_to_estimated_cost_pct",
    "inflation_adjusted_unit_price_2025_try",
    "inflation_adjusted_unit_price_2026_try",
    "inflation_adjusted_contract_value_2025_try",
    "inflation_adjusted_contract_value_2026_try",
}


def _norm(value: Any) -> str:
    return "" if pd.isna(value) else str(value).casefold().strip()


def _combine(row: pd.Series | dict[str, Any], fields: list[str]) -> str:
    return " | ".join(_norm(row.get(field, "")) for field in fields)


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(parsed):
        return 0.0
    return parsed


def numeric_similarity(left: float, right: float) -> float:
    denominator = max(abs(float(left)), abs(float(right)), 1.0)
    return max(0.0, 1.0 - abs(float(left) - float(right)) / denominator)


def _drop_price_like_fields(record: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(record).items() if key not in PRICE_LIKE_FIELDS}


@dataclass
class RetrievalEngine:
    training_data: pd.DataFrame
    embedder: HashingVectorizer
    embedding_matrix: Any

    @classmethod
    def fit(cls, training_data: pd.DataFrame) -> "RetrievalEngine":
        data = normalize_schema(training_data).reset_index(drop=True)
        texts = data.apply(lambda row: _combine(row, TEXT_FIELDS), axis=1)
        embedder = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=256,
            alternate_sign=False,
            norm=None,
        )
        embedding_matrix = normalize(embedder.transform(texts))
        return cls(training_data=data, embedder=embedder, embedding_matrix=embedding_matrix)

    def retrieve(self, query: pd.Series | dict[str, Any], top_k: int = 50) -> pd.DataFrame:
        safe_query = _drop_price_like_fields(mask_actual_result_fields(query))
        query_text = _combine(safe_query, TEXT_FIELDS)
        text_vector = normalize(self.embedder.transform([query_text]))
        embedding_scores = cosine_similarity(text_vector, self.embedding_matrix)[0]
        scores = []
        for idx, row in self.training_data.iterrows():
            product_group = 1.0 if _norm(row.get("product_group")) == _norm(safe_query.get("product_group")) else 0.0
            region = 1.0 if _norm(row.get("region")) == _norm(safe_query.get("region")) else 0.0
            procedure = 1.0 if _norm(row.get("procedure_type")) == _norm(safe_query.get("procedure_type")) else 0.0
            institution_type = (
                1.0 if _norm(row.get("buyer_institution_type")) == _norm(safe_query.get("buyer_institution_type")) else 0.0
            )
            quantity = numeric_similarity(_safe_float(row.get("quantity")), _safe_float(safe_query.get("quantity")))
            delivery = numeric_similarity(_safe_float(row.get("delivery_months")), _safe_float(safe_query.get("delivery_months")))
            competitors = numeric_similarity(
                _safe_float(row.get("competitor_count_estimate")),
                _safe_float(safe_query.get("competitor_count_estimate")),
            )
            score = (
                0.42 * float(embedding_scores[idx])
                + 0.15 * product_group
                + 0.10 * region
                + 0.08 * procedure
                + 0.05 * institution_type
                + 0.12 * quantity
                + 0.05 * delivery
                + 0.03 * competitors
            )
            scores.append(score)
        result = self.training_data.copy()
        result["overall_similarity_score"] = np.clip(scores, 0, 1)
        return result.sort_values("overall_similarity_score", ascending=False).head(top_k).reset_index(drop=True)


def retrieval_quality(similar: pd.DataFrame, query: pd.Series | dict[str, Any], top_k: int = 50) -> dict[str, float]:
    frame = similar.head(top_k)
    if frame.empty:
        return {
            "topk_avg_similarity": 0.0,
            "product_group_match_rate": 0.0,
            "region_match_rate": 0.0,
            "procedure_type_match_rate": 0.0,
            "buyer_institution_type_match_rate": 0.0,
            "quantity_band_match_rate": 0.0,
            "quantity_similarity_avg": 0.0,
            "delivery_similarity_avg": 0.0,
            "low_evidence_flag": 1.0,
        }
    quantity_scores = [
        numeric_similarity(_safe_float(value), _safe_float(query.get("quantity")))
        for value in frame.get("quantity", pd.Series(dtype=float))
    ]
    delivery_scores = [
        numeric_similarity(_safe_float(value), _safe_float(query.get("delivery_months")))
        for value in frame.get("delivery_months", pd.Series(dtype=float))
    ]
    return {
        "topk_avg_similarity": float(frame["overall_similarity_score"].mean()),
        "product_group_match_rate": float((frame["product_group"].map(_norm) == _norm(query.get("product_group"))).mean()),
        "region_match_rate": float((frame["region"].map(_norm) == _norm(query.get("region"))).mean()),
        "procedure_type_match_rate": float(
            (frame["procedure_type"].map(_norm) == _norm(query.get("procedure_type"))).mean()
        ),
        "buyer_institution_type_match_rate": float(
            (frame.get("buyer_institution_type", pd.Series(dtype=str)).map(_norm) == _norm(query.get("buyer_institution_type"))).mean()
        )
        if "buyer_institution_type" in frame
        else 0.0,
        "quantity_band_match_rate": float((frame.get("quantity_bucket", "") == query.get("quantity_bucket", "")).mean())
        if "quantity_bucket" in frame
        else 0.0,
        "quantity_similarity_avg": float(np.mean(quantity_scores)) if quantity_scores else 0.0,
        "delivery_similarity_avg": float(np.mean(delivery_scores)) if delivery_scores else 0.0,
        "low_evidence_flag": float(len(frame) < min(top_k, 10)),
    }
