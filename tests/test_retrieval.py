from src.feature_masking import mask_actual_result_fields
from src.retrieval import RetrievalEngine


def test_retrieval_returns_ranked_similar_tenders(tiny_df):
    engine = RetrievalEngine.fit(tiny_df[tiny_df["year"] <= 2023])
    query = mask_actual_result_fields(tiny_df[tiny_df["year"] == 2025].iloc[0].to_dict())
    similar = engine.retrieve(query, top_k=5)
    assert len(similar) == 5
    assert similar["overall_similarity_score"].is_monotonic_decreasing

