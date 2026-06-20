"""End-to-end scenario recommendation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..clustering import ProfileFitModel
from ..confidence import confidence_from_similarity_and_count
from ..retrieval import RetrievalEngine, retrieval_quality
from ..price_corridor import corridor_for_similar_tenders
from .scenario_generator import generate_candidate_scenarios
from .scenario_scorer import score_scenario
from .scenario_validator import validate_scenario


def rank_scenarios(
    train_df: pd.DataFrame,
    tender: dict[str, Any],
    include_actual: dict[str, Any] | None = None,
    top_k: int = 50,
) -> dict[str, Any]:
    retriever = RetrievalEngine.fit(train_df)
    similar = retriever.retrieve(tender, top_k=top_k)
    retrieval = retrieval_quality(similar, tender, top_k=top_k)
    corridor = corridor_for_similar_tenders(similar)
    avg_similarity = float(similar["overall_similarity_score"].mean()) if not similar.empty else 0.0
    confidence_score = confidence_from_similarity_and_count(avg_similarity, len(similar))
    profile_model = ProfileFitModel.fit(train_df)
    scenarios = generate_candidate_scenarios(tender, corridor, include_actual=include_actual)
    scored = []
    for scenario in scenarios:
        validation = validate_scenario(scenario, tender, corridor)
        margin = validation["computed_margin_pct"]
        profile = profile_model.score(tender, proposed_price=scenario["proposed_unit_price"], margin_pct=margin)
        scored.append(score_scenario(scenario, tender, corridor, profile, confidence_score, validation))
    scored_df = pd.DataFrame(scored).sort_values("scenario_score", ascending=False).reset_index(drop=True)
    return {
        "similar": similar,
        "retrieval_quality": retrieval,
        "top10_avg_similarity": float(similar.head(10)["overall_similarity_score"].mean()) if not similar.empty else 0.0,
        "top50_avg_similarity": avg_similarity,
        "corridor": corridor,
        "model_confidence_score": confidence_score,
        "scenarios": scored_df,
    }
