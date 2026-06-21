from src.evaluation.backtest_runner import run_backtest


def test_backtest_runner_outputs_required_columns(tiny_df):
    train = tiny_df[tiny_df["year"] <= 2024]
    test = tiny_df[tiny_df["year"] == 2025].head(2)
    results = run_backtest(train, test, top_k=6)
    required = {
        "tender_id",
        "actual_won_unit_price",
        "predicted_low_price",
        "predicted_mid_price",
        "predicted_high_price",
        "actual_won_scenario_rank_percentile",
        "won_profile_fit_score",
        "scenario_score",
        "leakage_audit_status",
        "advisor_validation_status",
        "selected_scenario_id",
        "selected_is_actual_configuration_candidate",
    }
    assert required.issubset(results.columns)
    assert set(results["leakage_audit_status"]) == {"pass"}
    assert not results["selected_is_actual_configuration_candidate"].any()
    assert "S_ACTUAL_HISTORICAL_CONFIG" not in set(results["selected_scenario_id"])
