from src.split_strategy import rolling_backtest_splits, temporal_split


def test_temporal_split_uses_year_order(tiny_df):
    split = temporal_split(tiny_df)
    assert split["train"]["year"].max() <= 2023
    assert set(split["validation"]["year"]) == {2024}
    assert set(split["test"]["year"]) == {2025}


def test_temporal_split_breaks_same_date_ties_by_tender_id(tiny_df):
    shuffled = tiny_df.sort_values("tender_id", ascending=False).copy()
    shuffled["tender_date"] = shuffled["year"].astype(str) + "-03-01"

    split = temporal_split(shuffled)

    assert split["test"]["tender_id"].tolist() == sorted(split["test"]["tender_id"].tolist())


def test_rolling_backtest_splits_are_temporal(tiny_df):
    splits = rolling_backtest_splits(tiny_df)
    assert splits
    assert all(max(item["train_years"]) < item["test_year"] for item in splits)
