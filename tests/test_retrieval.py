from src.feature_masking import mask_actual_result_fields
from src.retrieval import RetrievalEngine, retrieval_quality


def test_retrieval_returns_ranked_similar_tenders(tiny_df):
    engine = RetrievalEngine.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df[tiny_df["year"] == 2025].iloc[0].to_dict())
    similar = engine.retrieve(query, top_k=5)
    assert len(similar) == 5
    assert similar["overall_similarity_score"].is_monotonic_decreasing


def test_retrieval_does_not_depend_on_price_or_cost_fields(tiny_df):
    engine = RetrievalEngine.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df[tiny_df["year"] == 2025].iloc[0].to_dict())
    polluted = {
        **query,
        "actual_won_unit_price": 999_999,
        "winning_unit_price_try": 999_999,
        "actual_margin_pct": -90,
        "gross_margin_pct": -90,
        "estimated_unit_cost": 999_999,
        "estimated_unit_cost_try": 999_999,
        "internal_unit_cost_try": 999_999,
    }
    clean = engine.retrieve(query, top_k=5)
    dirty = engine.retrieve(polluted, top_k=5)
    assert clean["tender_id"].tolist() == dirty["tender_id"].tolist()
    assert clean["overall_similarity_score"].tolist() == dirty["overall_similarity_score"].tolist()


def test_retrieval_quality_exposes_structural_match_rates(tiny_df):
    engine = RetrievalEngine.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df[tiny_df["year"] == 2025].iloc[0].to_dict())
    similar = engine.retrieve(query, top_k=5)
    quality = retrieval_quality(similar, query, top_k=5)
    for key in [
        "topk_avg_similarity",
        "procedure_type_match_rate",
        "buyer_institution_type_match_rate",
        "quantity_similarity_avg",
        "delivery_similarity_avg",
        "low_evidence_flag",
    ]:
        assert key in quality
