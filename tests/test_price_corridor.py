from src.price_corridor import actual_inside_band, percentile_corridor, price_band_fit_score


def test_price_corridor_contains_expected_keys(tiny_df):
    corridor = percentile_corridor(tiny_df["actual_won_unit_price"])
    assert corridor["predicted_low_price"] <= corridor["predicted_mid_price"] <= corridor["predicted_high_price"]
    assert 0 <= price_band_fit_score(corridor["predicted_mid_price"], corridor) <= 100
    assert actual_inside_band(corridor["predicted_mid_price"], corridor)
