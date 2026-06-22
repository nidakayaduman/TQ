from src.feature_masking import mask_actual_result_fields
from src.leakage_audit import audit_pre_reveal_input


def test_feature_masking_removes_actual_result_fields(tiny_df):
    row = tiny_df.iloc[0].to_dict()
    masked = mask_actual_result_fields(row)
    assert "actual_won_unit_price" not in masked
    assert "actual_margin_pct" not in masked
    assert "contract_date" not in masked


def test_feature_masking_removes_configured_blocked_fields():
    row = {
        "tender_id": "T-1",
        "product_group": "Serum",
        "won_unit_price": 10.0,
        "won_total_amount": 100.0,
        "final_contract_amount": 100.0,
        "actual_margin_pct": 12.0,
    }
    masked = mask_actual_result_fields(row)
    audit = audit_pre_reveal_input(row["tender_id"], masked)

    assert "won_unit_price" not in masked
    assert "won_total_amount" not in masked
    assert "final_contract_amount" not in masked
    assert "actual_margin_pct" not in masked
    assert audit["audit_status"] == "pass"
