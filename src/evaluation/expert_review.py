"""Expert review template export."""

from __future__ import annotations

import pandas as pd


def expert_review_template(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "tender_id",
        "scenario_score",
        "won_profile_fit_score",
        "predicted_mid_price",
        "actual_won_unit_price",
        "expert_profile_fit_score",
        "expert_price_comment",
        "expert_risk_comment",
        "reviewer",
        "review_date",
    ]
    template = results.copy()
    for column in columns:
        if column not in template:
            template[column] = ""
    return template[columns]

