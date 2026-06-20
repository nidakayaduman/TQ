from src.clustering import ProfileFitModel
from src.feature_masking import mask_actual_result_fields


def test_profile_fit_scores_between_zero_and_100(tiny_df):
    model = ProfileFitModel.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df.iloc[-1].to_dict())
    score = model.score(query, proposed_price=18, margin_pct=20)
    assert 0 <= score["won_profile_fit_score"] <= 100
    assert "cluster_name" in score

