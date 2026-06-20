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
        scenarios = generate_candidate_scenarios(
            masked,
            corridor,
            include_actual={"actual_won_unit_price": actual_price},
        )
        scored = []
        for scenario in scenarios:
            validation = validate_scenario(scenario, masked, corridor)
            scored.append(
                score_scenario(
                    scenario,
                    masked,
                    corridor,
                    profile,
                    confidence_score,
                    validation,
                    weights=weights,
                    soft_penalties=soft_penalties,
                )
            )
        scored_df = pd.DataFrame(scored).sort_values("scenario_score", ascending=False).reset_index(drop=True)
        best = scored_df.iloc[0].to_dict()
        retrieval = retrieval_quality(similar, masked, top_k=top_k)
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
                "actual_won_scenario_rank_percentile": actual_rank_percentile(scored_df),
                "top10_avg_similarity": float(similar.head(10)["overall_similarity_score"].mean()),
                "top50_avg_similarity": avg_similarity,
                "won_profile_fit_score": profile["won_profile_fit_score"],
                "is_inlier": profile["is_inlier"],
                "inlier_score": profile["inlier_score"],
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
                "leakage_audit_status": audit["audit_status"],
                "advisor_validation_status": advisor_validation["advisor_validation_status"],
                "risk_flags": "; ".join(best["risk_flags"]),
                "hard_constraints_valid": bool(best["hard_constraints_valid"]),
                "retrieval_product_group_match_rate": retrieval["product_group_match_rate"],
                "retrieval_region_match_rate": retrieval["region_match_rate"],
                "retrieval_quantity_band_match_rate": retrieval["quantity_band_match_rate"],
                "soft_penalty_score": best["soft_penalty_score"],
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        avg_width = (output["band_width"] / output["predicted_mid_price"].replace(0, pd.NA).abs()).fillna(0).mean()
        output["coverage_adjusted_band_score"] = output["actual_inside_band"].astype(float) * (1 - min(float(avg_width), 1.0))
    return output
