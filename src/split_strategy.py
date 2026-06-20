"""Temporal split strategies for pseudo-live evaluation."""

from __future__ import annotations

import pandas as pd

from .config_loader import load_app_config
from .schema import normalize_schema


def temporal_split(
    df: pd.DataFrame,
    train_end_year: int | None = None,
    validation_year: int | None = None,
    test_year: int | None = None,
) -> dict[str, pd.DataFrame]:
    config = load_app_config().get("backtest", {})
    train_end_year = int(train_end_year if train_end_year is not None else config.get("train_end_year", 2023))
    validation_year = int(validation_year if validation_year is not None else config.get("validation_year", 2024))
    test_year = int(test_year if test_year is not None else config.get("test_year", 2025))
    data = normalize_schema(df).sort_values("tender_date")
    train = data[data["year"] <= train_end_year].copy()
    validation = data[data["year"] == validation_year].copy()
    test = data[data["year"] == test_year].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty train, validation, or test set.")
    return {"train": train, "validation": validation, "test": test}


def rolling_backtest_splits(df: pd.DataFrame, min_train_year: int | None = None) -> list[dict[str, object]]:
    data = normalize_schema(df).sort_values("tender_date")
    years = sorted(int(year) for year in data["year"].dropna().unique())
    if min_train_year is not None:
        years = [year for year in years if year >= min_train_year]
    splits: list[dict[str, object]] = []
    for index in range(1, len(years)):
        train_years = years[:index]
        test_year = years[index]
        train = data[data["year"].isin(train_years)].copy()
        test = data[data["year"] == test_year].copy()
        if not train.empty and not test.empty:
            splits.append(
                {
                    "train_years": train_years,
                    "test_year": test_year,
                    "train": train,
                    "test": test,
                }
            )
    return splits
