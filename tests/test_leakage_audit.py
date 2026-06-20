from src.feature_masking import mask_actual_result_fields
from src.leakage_audit import audit_pre_reveal_input


def test_leakage_audit_fails_when_actual_fields_present(tiny_df):
    row = tiny_df.iloc[0].to_dict()
    audit = audit_pre_reveal_input(row["tender_id"], row)
    assert audit["leakage_detected"]
    assert audit["audit_status"] == "fail"


def test_leakage_audit_passes_after_masking(tiny_df):
    row = tiny_df.iloc[0].to_dict()
    audit = audit_pre_reveal_input(row["tender_id"], mask_actual_result_fields(row))
    assert not audit["leakage_detected"]
    assert audit["audit_status"] == "pass"

