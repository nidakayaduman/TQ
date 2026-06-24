import pandas as pd

from src.price_corridor import actual_inside_band, corridor_for_similar_tenders, percentile_corridor, price_band_fit_score


def test_price_corridor_contains_expected_keys(tiny_df):
    corridor = percentile_corridor(tiny_df["actual_won_unit_price"])
    assert corridor["predicted_low_price"] <= corridor["predicted_mid_price"] <= corridor["predicted_high_price"]
    assert 0 <= price_band_fit_score(corridor["predicted_mid_price"], corridor) <= 100
    assert actual_inside_band(corridor["predicted_mid_price"], corridor)


def test_corridor_keeps_p25_p75_after_year_trend_adjustment():
    similar = pd.DataFrame(
        {
            "year": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "product_group": ["A"] * 8,
            "actual_won_unit_price": [110, 120, 95, 105, 70, 80, 50, 60],
        }
    )
    raw = percentile_corridor(similar["actual_won_unit_price"])
    adjusted = corridor_for_similar_tenders(similar)
    assert adjusted["predicted_low_price"] <= adjusted["predicted_mid_price"] <= adjusted["predicted_high_price"]
    assert adjusted["predicted_mid_price"] < raw["predicted_mid_price"]
    assert adjusted["band_width"] <= raw["band_width"]
