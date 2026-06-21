from src.clustering import LIVE_ASSIGNMENT_FEATURES, ProfileFitModel
from src.evaluation.stress_tests import evaluate_synthetic_outliers
from src.feature_masking import mask_actual_result_fields


def test_profile_fit_scores_between_zero_and_100(tiny_df):
    model = ProfileFitModel.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df.iloc[-1].to_dict())
    score = model.score(query, proposed_price=18, margin_pct=20)
    assert 0 <= score["won_profile_fit_score"] <= 100
    assert "cluster_name" in score
    assert isinstance(score["is_inlier"], bool)
    assert "anomaly_score" in score
    assert "manual_review_flag" in score


def test_profile_fit_does_not_depend_on_scenario_price(tiny_df):
    model = ProfileFitModel.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df.iloc[-1].to_dict())
    low_price = model.score(query, proposed_price=1, margin_pct=-80)
    high_price = model.score(query, proposed_price=10_000, margin_pct=95)
    assert low_price["won_profile_fit_score"] == high_price["won_profile_fit_score"]
    assert low_price["cluster_id"] == high_price["cluster_id"]


def test_profile_fit_live_assignment_uses_only_bid_time_profile_fields():
    forbidden = {
        "actual_won_unit_price",
        "won_unit_price",
        "actual_margin_pct",
        "actual_unit_margin",
        "final_contract_amount",
        "actual_award_result",
        "actual_delivery_result",
        "scenario_price",
        "recommended_bid_price",
    }
    assert forbidden.isdisjoint(set(LIVE_ASSIGNMENT_FEATURES))


def test_profile_fit_does_not_depend_on_actual_or_recommended_price_fields(tiny_df):
    model = ProfileFitModel.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df.iloc[-1].to_dict())
    polluted = {
        **query,
        "actual_won_unit_price": 1_000_000,
        "won_unit_price": 1_000_000,
        "actual_margin_pct": -95,
        "actual_unit_margin": -1_000,
        "final_contract_amount": 999_999_999,
        "actual_award_result": "lost",
        "actual_delivery_result": "failed",
        "scenario_price": 999_999,
        "recommended_bid_price": 888_888,
    }
    clean_score = model.score(query)
    polluted_score = model.score(polluted)
    for key in [
        "won_profile_fit_score",
        "cluster_id",
        "cluster_assignment_confidence",
        "cluster_distance",
        "anomaly_score",
        "is_inlier",
    ]:
        assert clean_score[key] == polluted_score[key]


def test_profile_fit_returns_cluster_diagnostics(tiny_df):
    model = ProfileFitModel.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df.iloc[-1].to_dict())
    score = model.score(query)
    assert score["cluster_id"] is not None
    assert 0 <= score["cluster_assignment_confidence"] <= 100
    assert score["cluster_distance"] >= 0
    assert score["cluster_second_distance"] >= score["cluster_distance"]
    assert score["cluster_count"] >= 1
    assert "cluster_silhouette_score" in score
    assert "cluster_inertia" in score
    assert "nearest_cluster_examples" in score
    assert isinstance(score["nearest_cluster_examples"], list)


def test_synthetic_outlier_test_produces_manual_review_signal(tiny_df):
    train = tiny_df[tiny_df["year"] <= 2023]
    base = mask_actual_result_fields(tiny_df[tiny_df["year"] == 2025].iloc[0].to_dict())
    results = evaluate_synthetic_outliers(train, base, top_k=8)
    assert not results.empty
    assert (results["Manuel inceleme önerisi"] == "Evet").any()
