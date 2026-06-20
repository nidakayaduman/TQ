"""Simple baseline comparisons."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..constants import CANONICAL_PRICE_COLUMN
from ..schema import normalize_schema

FEATURES = ["product_group", "region", "procedure_type", "quantity", "delivery_months", "competitor_count_estimate"]


def _pipeline(model: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["product_group", "region", "procedure_type"]),
            ("num", StandardScaler(), ["quantity", "delivery_months", "competitor_count_estimate"]),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_data = normalize_schema(train)
    test_data = normalize_schema(test)
    rows: list[dict[str, object]] = []
    product_medians = train_data.groupby("product_group")[CANONICAL_PRICE_COLUMN].median()
    global_median = float(train_data[CANONICAL_PRICE_COLUMN].median())
    cost_plus = pd.to_numeric(test_data["estimated_unit_cost"], errors="coerce").fillna(global_median * 0.8) / 0.80

    models = {
        "Linear Regression": _pipeline(LinearRegression()),
        "Tree-based model": _pipeline(RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3)),
    }
    predictions = {
        "Product group median": test_data["product_group"].map(product_medians).fillna(global_median).astype(float),
        "Cost plus fixed margin": cost_plus.astype(float),
    }
    for name, model in models.items():
        model.fit(train_data[FEATURES], train_data[CANONICAL_PRICE_COLUMN].astype(float))
        predictions[name] = pd.Series(model.predict(test_data[FEATURES]), index=test_data.index)
    predictions["Ensemble"] = pd.concat(predictions.values(), axis=1).mean(axis=1)

    actual = test_data[CANONICAL_PRICE_COLUMN].astype(float)
    for name, predicted in predictions.items():
        error = (actual - predicted).abs()
        rows.append(
            {
                "Model": name,
                "MAE": float(error.mean()),
                "MAPE": float((error / actual.abs() * 100).mean()),
                "Coverage": 0.0,
                "Avg Band Width": 0.0,
            }
        )
    return pd.DataFrame(rows)

