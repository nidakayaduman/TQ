from src.config_loader import active_config_summary, load_hard_constraints, load_scenario_weights, load_soft_penalties


def test_yaml_configs_are_loaded():
    hard = load_hard_constraints()
    soft = load_soft_penalties()
    weights = load_scenario_weights()
    summary = active_config_summary()

    assert hard["minimum_margin_pct"] == 8.0
    assert soft["price_outside_band_penalty"] == 12.0
    assert weights["won_profile_fit_score"] == 0.30
    assert summary["default_top_k"] == 50
