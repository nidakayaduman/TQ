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


def mark_pareto_frontier(scored_df: pd.DataFrame) -> pd.DataFrame:
    output = scored_df.copy()
    output["risk_suitability_score"] = 100 - output["risk_penalty_score"].astype(float)
    output["is_pareto_efficient"] = False
    valid = output[output["hard_constraints_valid"].astype(bool)].copy()
    if valid.empty:
        return output
    objectives = valid[["won_profile_fit_score", "margin_score", "risk_suitability_score", "model_confidence_score"]].astype(float)
    pareto_indices: list[int] = []
    for idx, row in objectives.iterrows():
        dominated = (
            (objectives >= row).all(axis=1)
            & (objectives > row).any(axis=1)
        ).any()
        if not dominated:
            pareto_indices.append(idx)
    output.loc[pareto_indices, "is_pareto_efficient"] = True
    return output


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
    profile_for_score = {
        **profile,
        "topk_avg_similarity": avg_similarity,
        "retrieval_quality": retrieval,
    }
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
                profile_for_score,
                confidence_score,
                validation,
                weights=weights,
                soft_penalties=soft_penalties,
            )
        )
    scored_df = pd.DataFrame(scored)
    scored_df = mark_pareto_frontier(scored_df).sort_values(
        ["hard_constraints_valid", "is_pareto_efficient", "scenario_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    valid_count = int(scored_df["hard_constraints_valid"].astype(bool).sum()) if "hard_constraints_valid" in scored_df else 0
    failure_reason = ""
    if valid_count == 0:
        failure_reason = (
            "Geçerli öneri üretilemedi. Tanımlı kesin kurallar içinde fiyat, marj veya teslim planı "
            "geçmiş kazanılmış profil bandına yaklaştırılamıyor. Manuel teklif komitesi incelemesi önerilir."
        )
    evidence = (
        f"Top-{len(similar)} emsal içinde ortalama benzerlik {avg_similarity:.2f}; "
        f"ürün grubu eşleşmesi %{retrieval['product_group_match_rate'] * 100:.0f}, "
        f"bölge eşleşmesi %{retrieval['region_match_rate'] * 100:.0f}."
    )
    if not scored_df.empty:
        scored_df["evidence_from_similar_tenders"] = evidence
    return {
        "similar": similar,
        "retrieval_quality": retrieval,
        "top10_avg_similarity": float(similar.head(10)["overall_similarity_score"].mean()) if not similar.empty else 0.0,
        "top50_avg_similarity": avg_similarity,
        "corridor": corridor,
        "model_confidence_score": confidence_score,
        "scenarios": scored_df,
        "valid_scenario_count": valid_count,
        "failure_reason": failure_reason,
    }
