from src.feature_masking import mask_actual_result_fields


def test_feature_masking_removes_actual_result_fields(tiny_df):
    row = tiny_df.iloc[0].to_dict()
    masked = mask_actual_result_fields(row)
    assert "actual_won_unit_price" not in masked
    assert "actual_margin_pct" not in masked
    assert "contract_date" not in masked

