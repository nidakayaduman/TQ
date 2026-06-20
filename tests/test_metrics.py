import pandas as pd

from src.evaluation.metrics import optimizer_metrics, price_corridor_metrics


def test_metrics_compute_band_width_penalty():
    results = pd.DataFrame(
        {
            "actual_won_unit_price": [10, 20],
            "predicted_mid_price": [11, 19],
            "predicted_low_price": [8, 10],
            "predicted_high_price": [14, 30],
            "actual_inside_band": [True, True],
            "actual_won_scenario_rank_percentile": [80, 90],
            "hard_constraints_valid": [True, False],
        }
    )
    metrics = price_corridor_metrics(results)
    opt = optimizer_metrics(results)
    assert metrics["coverage_adjusted_band_score"] < 1
    assert opt["top30_hit_rate"] == 1
    assert opt["hard_constraint_violation_rate"] == 0.5

