from src.config_loader import active_config_summary, load_hard_constraints, load_scenario_weights, load_soft_penalties


def test_yaml_configs_are_loaded():
    hard = load_hard_constraints()
    soft = load_soft_penalties()
    weights = load_scenario_weights()
    summary = active_config_summary()

    assert hard["minimum_margin_pct"] == 8.0
    assert hard["margin"]["min_margin_pct"] == 0.08
    assert hard["price"]["max_price_over_topk_p90_multiplier"] == 1.15
    assert hard["leakage"]["blocked_fields_before_reveal"]
    assert soft["price_outside_band_penalty"] == 12.0
    assert soft["low_similarity"]["threshold"] == 0.55
    assert soft["model_disagreement"]["penalty_points"] == 6
    assert weights["won_profile_fit_score"] == 0.30
    assert summary["default_top_k"] == 50
