"""Expert review template export."""

from __future__ import annotations

import pandas as pd


def expert_review_template(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "tender_id",
        "product_name",
        "product_group",
        "region",
        "buyer_institution",
        "buyer_institution_type",
        "quantity_bucket",
        "top10_avg_similarity",
        "top50_avg_similarity",
        "won_profile_fit_score",
        "cluster_id",
        "cluster_name",
        "is_inlier",
        "inlier_score",
        "predicted_low_price",
        "predicted_mid_price",
        "predicted_high_price",
        "actual_inside_band",
        "actual_won_scenario_rank_percentile",
        "scenario_score",
        "hard_constraints_valid",
        "invalid_reason",
        "risk_flags",
        "soft_penalty_score",
        "leakage_audit_status",
        "advisor_validation_status",
        "revealed_actual_result",
        "actual_won_unit_price",
        "actual_margin_pct",
        "expert_profile_fit_score",
        "expert_price_comment",
        "expert_risk_comment",
        "expert_decision",
        "manual_review_notes",
        "reviewer",
        "review_date",
    ]
    template = results.copy()
    if "revealed_actual_result" not in template:
        template["revealed_actual_result"] = True
    for column in columns:
        if column not in template:
            template[column] = ""
    return template[columns]
