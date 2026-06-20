"""Similar won tender retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .feature_masking import mask_actual_result_fields
from .schema import normalize_schema

TEXT_FIELDS = ["product_name", "product_group", "buyer_institution", "region", "procedure_type"]


def _norm(value: Any) -> str:
    return "" if pd.isna(value) else str(value).casefold().strip()


def _combine(row: pd.Series | dict[str, Any], fields: list[str]) -> str:
    return " | ".join(_norm(row.get(field, "")) for field in fields)


def numeric_similarity(left: float, right: float) -> float:
    denominator = max(abs(float(left)), abs(float(right)), 1.0)
    return max(0.0, 1.0 - abs(float(left) - float(right)) / denominator)


@dataclass
class RetrievalEngine:
    training_data: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: Any

    @classmethod
    def fit(cls, training_data: pd.DataFrame) -> "RetrievalEngine":
        data = normalize_schema(training_data).reset_index(drop=True)
        texts = data.apply(lambda row: _combine(row, TEXT_FIELDS), axis=1)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = normalize(vectorizer.fit_transform(texts))
        return cls(training_data=data, vectorizer=vectorizer, matrix=matrix)

    def retrieve(self, query: pd.Series | dict[str, Any], top_k: int = 50) -> pd.DataFrame:
        safe_query = mask_actual_result_fields(query)
        query_text = _combine(safe_query, TEXT_FIELDS)
        text_vector = normalize(self.vectorizer.transform([query_text]))
        text_scores = cosine_similarity(text_vector, self.matrix)[0]
        scores = []
        for idx, row in self.training_data.iterrows():
            product_group = 1.0 if _norm(row.get("product_group")) == _norm(safe_query.get("product_group")) else 0.0
            region = 1.0 if _norm(row.get("region")) == _norm(safe_query.get("region")) else 0.0
            procedure = 1.0 if _norm(row.get("procedure_type")) == _norm(safe_query.get("procedure_type")) else 0.0
            quantity = numeric_similarity(float(row.get("quantity", 0)), float(safe_query.get("quantity", 0)))
            delivery = numeric_similarity(float(row.get("delivery_months", 0)), float(safe_query.get("delivery_months", 0)))
            competitors = numeric_similarity(
                float(row.get("competitor_count_estimate", 0)),
                float(safe_query.get("competitor_count_estimate", 0)),
            )
            score = (
                0.45 * float(text_scores[idx])
                + 0.15 * product_group
                + 0.10 * region
                + 0.08 * procedure
                + 0.12 * quantity
                + 0.05 * delivery
                + 0.05 * competitors
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
            "quantity_band_match_rate": 0.0,
        }
    return {
        "topk_avg_similarity": float(frame["overall_similarity_score"].mean()),
        "product_group_match_rate": float((frame["product_group"].map(_norm) == _norm(query.get("product_group"))).mean()),
        "region_match_rate": float((frame["region"].map(_norm) == _norm(query.get("region"))).mean()),
        "quantity_band_match_rate": float((frame.get("quantity_bucket", "") == query.get("quantity_bucket", "")).mean())
        if "quantity_bucket" in frame
        else 0.0,
    }

