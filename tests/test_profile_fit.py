from src.clustering import LIVE_ASSIGNMENT_FEATURES, ProfileFitModel, gower_distance_matrix
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


def test_gower_distance_rewards_mixed_type_similarity():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "product_group": "A",
                "product_name": "A1",
                "buyer_institution": "Kurum 1",
                "region": "Marmara",
                "procedure_type": "Açık",
                "buyer_institution_type": "Kamu",
                "quantity": 100,
                "delivery_months": 6,
                "competitor_count_estimate": 2,
            },
            {
                "product_group": "A",
                "product_name": "A1",
                "buyer_institution": "Kurum 1",
                "region": "Marmara",
                "procedure_type": "Açık",
                "buyer_institution_type": "Kamu",
                "quantity": 110,
                "delivery_months": 6,
                "competitor_count_estimate": 2,
            },
            {
                "product_group": "B",
                "product_name": "B1",
                "buyer_institution": "Kurum 9",
                "region": "Ege",
                "procedure_type": "Pazarlık",
                "buyer_institution_type": "Özel",
                "quantity": 900,
                "delivery_months": 18,
                "competitor_count_estimate": 8,
            },
        ]
    )
    distances = gower_distance_matrix(frame)
    assert distances[0, 1] < distances[0, 2]


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
        "estimated_unit_cost",
        "estimated_unit_cost_try",
        "internal_unit_cost_try",
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
        "estimated_unit_cost": 999_999,
        "estimated_unit_cost_try": 999_999,
        "internal_unit_cost_try": 999_999,
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
        "topk_profile_score",
        "mixed_cluster_score",
        "cluster_purity_score",
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
    assert "topk_profile_score" in score
    assert "mixed_cluster_score" in score
    assert "nearest_profile_density_score" in score
    assert "global_inlier_score" in score
    assert "segment_inlier_score" in score
    assert "isolation_calibration_scope" in score
    assert "cluster_purity_score" in score
    assert "profile_score_components" in score
    assert "manual_review_reasons" in score
    assert isinstance(score["nearest_cluster_examples"], list)
    assert 0 <= score["topk_profile_score"] <= 100
    assert 0 <= score["mixed_cluster_score"] <= 100
    assert 0 <= score["nearest_profile_density_score"] <= 100
    assert 0 <= score["global_inlier_score"] <= 100
    assert 0 <= score["segment_inlier_score"] <= 100
    assert score["inlier_score"] >= score["global_inlier_score"]
    assert 0 <= score["cluster_purity_score"] <= 100


def test_synthetic_outlier_test_produces_manual_review_signal(tiny_df):
    train = tiny_df[tiny_df["year"] <= 2023]
    base = mask_actual_result_fields(tiny_df[tiny_df["year"] == 2025].iloc[0].to_dict())
    results = evaluate_synthetic_outliers(train, base, top_k=8)
    assert not results.empty
    assert (results["Manuel inceleme önerisi"] == "Evet").any()
