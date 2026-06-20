"""Confidence scoring helpers."""

from __future__ import annotations

import numpy as np


def confidence_from_similarity_and_count(avg_similarity: float, similar_count: int, model_disagreement: float = 0.0) -> float:
    count_score = min(100.0, similar_count / 30.0 * 100.0)
    similarity_score = float(np.clip(avg_similarity * 100.0, 0, 100))
    disagreement_penalty = float(np.clip(model_disagreement * 100.0, 0, 60))
    return float(np.clip(0.55 * similarity_score + 0.35 * count_score - 0.10 * disagreement_penalty, 0, 100))


def confidence_label(score: float) -> str:
    if score >= 75:
        return "Yüksek"
    if score >= 50:
        return "Orta"
    return "Düşük"

