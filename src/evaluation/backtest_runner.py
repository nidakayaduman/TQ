"""Pseudo-live temporal backtest runner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..advisor.fallback_advisor import build_fallback_advisor
from ..advisor.output_validator import validate_advisor_output
from ..clustering import ProfileFitModel
from ..config_loader import load_app_config, load_scenario_weights, load_soft_penalties
from ..confidence import confidence_from_similarity_and_count
from ..constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from ..feature_masking import mask_actual_result_fields
from ..leakage_audit import audit_pre_reveal_input
from ..model_version import MODEL_VERSION
from ..optimizer.scenario_generator import generate_candidate_scenarios
from ..optimizer.scenario_scorer import score_scenario
from ..optimizer.scenario_validator import validate_scenario
from ..price_corridor import actual_inside_band, corridor_for_similar_tenders
from ..retrieval import RetrievalEngine, retrieval_quality
from ..schema import normalize_schema


def actual_rank_percentile(scored_scenarios: pd.DataFrame) -> float:
    actual = scored_scenarios[scored_scenarios["is_actual_configuration_candidate"]]
    if actual.empty:
        return 0.0
    actual_score = float(actual.iloc[0]["scenario_score"])
    below = int((scored_scenarios["scenario_score"] < actual_score).sum())
    return float(100 * below / max(len(scored_scenarios), 1))


def run_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    if top_k is None:
        top_k = int(load_app_config().get("app", {}).get("default_top_k", 50))
    train = normalize_schema(train_df)
    test = normalize_schema(test_df)
    retriever = RetrievalEngine.fit(train)
    profile_model = ProfileFitModel.fit(train)
    weights = load_scenario_weights()
    soft_penalties = load_soft_penalties()
    rows: list[dict[str, Any]] = []
    for _, tender_row in test.iterrows():
        tender = tender_row.to_dict()
        tender_id = str(tender["tender_id"])
        masked = mask_actual_result_fields(tender)
        audit = audit_pre_reveal_input(tender_id, masked)
        if audit["leakage_detected"]:
            raise ValueError(f"Leakage detected in backtest input for {tender_id}")

        similar = retriever.retrieve(masked, top_k=top_k)
        corridor = corridor_for_similar_tenders(similar)
        actual_price = float(tender[CANONICAL_PRICE_COLUMN])
        actual_margin = float(tender.get(CANONICAL_MARGIN_COLUMN, 0))
        avg_similarity = float(similar["overall_similarity_score"].mean()) if not similar.empty else 0.0
        confidence_score = confidence_from_similarity_and_count(avg_similarity, len(similar))
        profile = profile_model.score(masked)
        profile_for_score = {
            **profile,
            "topk_avg_similarity": avg_similarity,
        }
        recommendation_scenarios = generate_candidate_scenarios(masked, corridor)
        scored = []
        for scenario in recommendation_scenarios:
            validation = validate_scenario(scenario, masked, corridor)
            scored.append(
                score_scenario(
                    scenario,
                    masked,
                    corridor,
                    profile_for_score,
                    confidence_score,
                    validation,
                    weights=weights,
                    soft_penalties=soft_penalties,
                )
            )
        scored_df = pd.DataFrame(scored).sort_values(
            ["hard_constraints_valid", "scenario_score"],
            ascending=[False, False],
        ).reset_index(drop=True)
        best = scored_df.iloc[0].to_dict()

        rank_scenarios = [
            *recommendation_scenarios,
            *generate_candidate_scenarios(masked, corridor, include_actual={"actual_won_unit_price": actual_price})[-1:],
        ]
        rank_scored = []
        for scenario in rank_scenarios:
            validation = validate_scenario(scenario, masked, corridor)
            rank_scored.append(
                score_scenario(
                    scenario,
                    masked,
                    corridor,
                    profile_for_score,
                    confidence_score,
                    validation,
                    weights=weights,
                    soft_penalties=soft_penalties,
                )
            )
        rank_scored_df = pd.DataFrame(rank_scored)
        retrieval = retrieval_quality(similar, masked, top_k=top_k)
        similar_summary = "; ".join(
            f"{row['tender_id']} ({float(row['overall_similarity_score']):.2f})"
            for _, row in similar.head(5).iterrows()
        )
        context = {
            **best,
            **profile,
            "similar_tender_count": len(similar),
            "cluster_name": profile["cluster_name"],
        }
        advisor = build_fallback_advisor(context)
        advisor_validation = validate_advisor_output(advisor)
        rows.append(
            {
                "tender_id": tender_id,
                "tender_date": tender["tender_date"],
                "year": int(tender["year"]),
                "product_name": tender.get("product_name"),
                "product_group": tender.get("product_group"),
                "buyer_institution": tender.get("buyer_institution"),
                "buyer_institution_type": tender.get("buyer_institution_type"),
                "region": tender.get("region"),
                "procedure_type": tender.get("procedure_type"),
                "quantity_bucket": tender.get("quantity_bucket"),
                "delivery_bucket": tender.get("delivery_bucket"),
                "actual_won_unit_price": actual_price,
                "actual_margin_pct": actual_margin,
                "predicted_low_price": corridor["predicted_low_price"],
                "predicted_mid_price": corridor["predicted_mid_price"],
                "predicted_high_price": corridor["predicted_high_price"],
                "actual_inside_band": actual_inside_band(actual_price, corridor),
                "absolute_error_mid": abs(actual_price - corridor["predicted_mid_price"]),
                "percentage_error_mid": abs(actual_price - corridor["predicted_mid_price"]) / max(abs(actual_price), 1.0) * 100,
                "band_width": corridor["band_width"],
                "coverage_adjusted_band_score": 0.0,
                "actual_won_scenario_rank_percentile": actual_rank_percentile(rank_scored_df),
                "selected_scenario_id": best.get("scenario_id", ""),
                "selected_is_actual_configuration_candidate": bool(best.get("is_actual_configuration_candidate", False)),
                "top10_avg_similarity": float(similar.head(10)["overall_similarity_score"].mean()),
                "top50_avg_similarity": avg_similarity,
                "won_profile_fit_score": profile["won_profile_fit_score"],
                "is_inlier": profile["is_inlier"],
                "inlier_score": profile["inlier_score"],
                "anomaly_score": profile.get("anomaly_score"),
                "isolation_threshold": profile.get("isolation_threshold"),
                "manual_review_flag": profile.get("manual_review_flag"),
                "manual_review_reasons": "; ".join(profile.get("manual_review_reasons", [])),
                "topk_profile_score": profile.get("topk_profile_score"),
                "mixed_cluster_score": profile.get("mixed_cluster_score"),
                "cluster_purity_score": profile.get("cluster_purity_score"),
                "profile_score_components": profile.get("profile_score_components"),
                "isolation_contamination": profile["isolation_contamination"],
                "training_anomaly_rate": profile["training_anomaly_rate"],
                "segment_anomaly_rate": profile["segment_anomaly_rate"],
                "price_band_fit_score": best["price_band_fit_score"],
                "margin_score": best["margin_score"],
                "risk_score": 100 - best["risk_penalty_score"],
                "model_confidence_score": confidence_score,
                "scenario_score": best["scenario_score"],
                "cluster_id": profile["cluster_id"],
                "cluster_name": profile["cluster_name"],
                "cluster_assignment_confidence": profile.get("cluster_assignment_confidence"),
                "cluster_distance": profile.get("cluster_distance"),
                "cluster_second_distance": profile.get("cluster_second_distance"),
                "cluster_silhouette_score": profile.get("cluster_silhouette_score"),
                "cluster_inertia": profile.get("cluster_inertia"),
                "cluster_min_size": profile.get("cluster_min_size"),
                "cluster_max_size": profile.get("cluster_max_size"),
                "cluster_dominant_procedure_type": profile.get("cluster_dominant_procedure_type"),
                "cluster_dominant_procedure_type_ratio": profile.get("cluster_dominant_procedure_type_ratio"),
                "leakage_audit_status": audit["audit_status"],
                "leakage_blocked_fields_present": "; ".join(audit["blocked_fields_present"]),
                "leakage_masked_fields_count": audit["masked_fields_count"],
                "advisor_validation_status": advisor_validation["advisor_validation_status"],
                "llm_validation_status": advisor_validation.get("llm_validation_status", advisor_validation["advisor_validation_status"]),
                "advisor_schema_valid": advisor_validation.get("schema_valid", False),
                "advisor_forbidden_claims_detected": advisor_validation.get("forbidden_claims_detected", False),
                "advisor_grounding_score": advisor_validation.get("grounding_score", 0.0),
                "advisor_prompt_injection_detected": advisor_validation.get("prompt_injection_detected", False),
                "advisor_fallback_used": True,
                "risk_flags": "; ".join(best["risk_flags"]),
                "hard_constraints_valid": bool(best["hard_constraints_valid"]),
                "invalid_reason": best.get("invalid_reason", ""),
                "soft_penalty_explanations": best.get("soft_penalty_explanations", ""),
                "hard_constraint_status": best.get("hard_constraint_status", ""),
                "caveat": best.get("caveat", ""),
                "failure_reason": "",
                "retrieval_product_group_match_rate": retrieval["product_group_match_rate"],
                "retrieval_region_match_rate": retrieval["region_match_rate"],
                "retrieval_quantity_band_match_rate": retrieval["quantity_band_match_rate"],
                "top_similar_tenders_summary": similar_summary,
                "reveal_status": "revealed_for_backtest",
                "soft_penalty_score": best["soft_penalty_score"],
                **MODEL_VERSION,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        avg_width = (output["band_width"] / output["predicted_mid_price"].replace(0, pd.NA).abs()).fillna(0).mean()
        output["coverage_adjusted_band_score"] = output["actual_inside_band"].astype(float) * (1 - min(float(avg_width), 1.0))
    return output
