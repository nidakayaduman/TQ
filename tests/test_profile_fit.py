from src.clustering import ProfileFitModel
from src.feature_masking import mask_actual_result_fields


def test_profile_fit_scores_between_zero_and_100(tiny_df):
    model = ProfileFitModel.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df.iloc[-1].to_dict())
    score = model.score(query, proposed_price=18, margin_pct=20)
    assert 0 <= score["won_profile_fit_score"] <= 100
    assert "cluster_name" in score


def test_profile_fit_does_not_depend_on_scenario_price(tiny_df):
    model = ProfileFitModel.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df.iloc[-1].to_dict())
    low_price = model.score(query, proposed_price=1, margin_pct=-80)
    high_price = model.score(query, proposed_price=10_000, margin_pct=95)
    assert low_price["won_profile_fit_score"] == high_price["won_profile_fit_score"]
    assert low_price["cluster_id"] == high_price["cluster_id"]
