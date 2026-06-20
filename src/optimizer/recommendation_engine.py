"""End-to-end scenario recommendation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..clustering import ProfileFitModel
from ..config_loader import load_app_config, load_scenario_weights, load_soft_penalties
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
    top_k: int | None = None,
) -> dict[str, Any]:
    if top_k is None:
        top_k = int(load_app_config().get("app", {}).get("default_top_k", 50))
    retriever = RetrievalEngine.fit(train_df)
    similar = retriever.retrieve(tender, top_k=top_k)
    retrieval = retrieval_quality(similar, tender, top_k=top_k)
    corridor = corridor_for_similar_tenders(similar)
    avg_similarity = float(similar["overall_similarity_score"].mean()) if not similar.empty else 0.0
    confidence_score = confidence_from_similarity_and_count(avg_similarity, len(similar))
    profile_model = ProfileFitModel.fit(train_df)
    profile = profile_model.score(tender)
    weights = load_scenario_weights()
    soft_penalties = load_soft_penalties()
    scenarios = generate_candidate_scenarios(tender, corridor, include_actual=include_actual)
    scored = []
    for scenario in scenarios:
        validation = validate_scenario(scenario, tender, corridor)
        scored.append(
            score_scenario(
                scenario,
                tender,
                corridor,
                profile,
                confidence_score,
                validation,
                weights=weights,
                soft_penalties=soft_penalties,
            )
        )
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
