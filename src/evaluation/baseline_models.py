"""Simple baseline comparisons."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..constants import CANONICAL_PRICE_COLUMN
from ..retrieval import RetrievalEngine
from ..schema import normalize_schema

FEATURES = ["product_group", "region", "procedure_type", "quantity", "delivery_months", "competitor_count_estimate"]
MIN_REASONABLE_PRICE = 0.01
EPSILON = 1e-9


def _pipeline(model: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["product_group", "region", "procedure_type"]),
            ("num", StandardScaler(), ["quantity", "delivery_months", "competitor_count_estimate"]),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def clean_numeric_outputs(values: pd.Series, fallback: pd.Series | float, max_multiplier: float = 8.0) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    fallback_series = (
        pd.Series([float(fallback)] * len(series), index=series.index)
        if isinstance(fallback, (int, float))
        else pd.to_numeric(fallback, errors="coerce").reindex(series.index).astype(float)
    )
    fallback_series = fallback_series.replace([np.inf, -np.inf], np.nan).fillna(MIN_REASONABLE_PRICE)
    upper = (fallback_series.abs() * max_multiplier).clip(lower=MIN_REASONABLE_PRICE)
    invalid = series.isna() | (series <= MIN_REASONABLE_PRICE) | (series.abs() > upper)
    return series.mask(invalid, fallback_series).clip(lower=MIN_REASONABLE_PRICE)


def safe_mape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan)
    predicted_values = pd.to_numeric(predicted, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = actual_values.notna() & predicted_values.notna() & (actual_values.abs() > EPSILON)
    if not valid.any():
        return 0.0
    return float(((actual_values[valid] - predicted_values[valid]).abs() / actual_values[valid].abs() * 100).mean())


def _predict_model_series(model: Pipeline, features: pd.DataFrame, fallback: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = pd.Series(model.predict(features), index=features.index)
    return clean_numeric_outputs(raw, fallback)


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_data = normalize_schema(train)
    test_data = normalize_schema(test)
    rows: list[dict[str, object]] = []
    product_medians = train_data.groupby("product_group")[CANONICAL_PRICE_COLUMN].median()
    global_median = float(train_data[CANONICAL_PRICE_COLUMN].median())
    cost_plus = pd.to_numeric(test_data["estimated_unit_cost"], errors="coerce").fillna(global_median * 0.8) / 0.80
    retriever = RetrievalEngine.fit(train_data)
    topk_values = []
    for _, row in test_data.iterrows():
        similar = retriever.retrieve(row.to_dict(), top_k=50)
        topk_values.append(float(similar[CANONICAL_PRICE_COLUMN].median()) if not similar.empty else global_median)
    topk_median = pd.Series(topk_values, index=test_data.index)

    models = {
        "Linear Regression Baseline": _pipeline(LinearRegression()),
        "Random Forest / Ağaç Tabanlı Baseline": _pipeline(
            RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3)
        ),
    }
    predictions = {
        "Product Group Median": test_data["product_group"].map(product_medians).fillna(global_median).astype(float),
        "Top-K Median": topk_median.astype(float),
        "Cost Plus Margin": cost_plus.astype(float),
    }
    for name, model in models.items():
        model.fit(train_data[FEATURES], train_data[CANONICAL_PRICE_COLUMN].astype(float))
        predictions[name] = _predict_model_series(model, test_data[FEATURES], predictions["Product Group Median"])

    actual = test_data[CANONICAL_PRICE_COLUMN].astype(float)
    descriptions = {
        "Product Group Median": "Aynı ürün grubundaki geçmiş kazanılmış fiyatların medyanını baz alır.",
        "Top-K Median": "Seçili ihaleye en çok benzeyen geçmiş ihalelerin medyan fiyatını kullanır.",
        "Cost Plus Margin": "Tahmini maliyetin üzerine hedef marj ekleyerek fiyat üretir.",
        "Linear Regression Baseline": "Sayısal ve kategorik alanlardan doğrusal fiyat tahmini üretir.",
        "Random Forest / Ağaç Tabanlı Baseline": "Ağaç tabanlı baseline modeldir; doğrusal olmayan örüntüleri yakalamaya çalışır.",
    }
    for name, predicted in predictions.items():
        predicted = clean_numeric_outputs(predicted, predictions["Product Group Median"])
        error = (actual - predicted).abs()
        low = predicted * 0.85
        high = predicted * 1.15
        rows.append(
            {
                "Model": name,
                "MAE": float(error.mean()),
                "MAPE": safe_mape(actual, predicted),
                "Coverage": float(((actual >= low) & (actual <= high)).mean()),
                "Avg Band Width": float((high - low).mean()),
                "Description": descriptions[name],
            }
        )
    return pd.DataFrame(rows)


def predict_baseline_prices(train: pd.DataFrame, tender: dict[str, object]) -> pd.DataFrame:
    train_data = normalize_schema(train)
    tender_df = normalize_schema(pd.DataFrame([dict(tender)]))
    product_medians = train_data.groupby("product_group")[CANONICAL_PRICE_COLUMN].median()
    global_median = float(train_data[CANONICAL_PRICE_COLUMN].median())
    product_group = tender_df["product_group"].iloc[0]
    median_prediction = float(product_medians.get(product_group, global_median))
    estimated_cost = float(pd.to_numeric(tender_df["estimated_unit_cost"], errors="coerce").fillna(global_median * 0.8).iloc[0])
    similar = RetrievalEngine.fit(train_data).retrieve(tender_df.iloc[0].to_dict(), top_k=50)
    topk_prediction = float(similar[CANONICAL_PRICE_COLUMN].median()) if not similar.empty else median_prediction
    def usable_prediction(raw_value: float, fallback_value: float) -> tuple[float, bool, str]:
        upper = max(abs(float(fallback_value)) * 8.0, MIN_REASONABLE_PRICE)
        if pd.isna(raw_value) or raw_value <= MIN_REASONABLE_PRICE:
            return max(float(fallback_value), MIN_REASONABLE_PRICE), True, "geçersiz"
        if abs(float(raw_value)) > upper:
            return max(float(fallback_value), MIN_REASONABLE_PRICE), True, "aşırı uç"
        return max(float(raw_value), MIN_REASONABLE_PRICE), False, ""

    predictions: list[dict[str, object]] = [
        {
            "method": "Product Group Median",
            "prediction": median_prediction,
            "description": "Ürün grubu medyanı; yeterli ürün grubu yoksa genel medyan kullanılır.",
            "confidence": "Orta",
        },
        {
            "method": "Top-K Median",
            "prediction": topk_prediction,
            "description": "Seçili ihaleye en çok benzeyen geçmiş kazanılmış ihalelerin medyan fiyatı.",
            "confidence": "Orta" if len(similar) >= 10 else "Düşük",
        },
        {
            "method": "Cost Plus Margin",
            "prediction": max(estimated_cost / 0.80, MIN_REASONABLE_PRICE),
            "description": "Tahmini maliyet üzerine yaklaşık %20 hedef karlılık oranı eklenmiş referans fiyat.",
            "confidence": "Orta",
        },
    ]
    models = {
        "Linear Regression Baseline": _pipeline(LinearRegression()),
        "Random Forest / Ağaç Tabanlı Baseline": _pipeline(
            RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3)
        ),
    }
    for name, model in models.items():
        model.fit(train_data[FEATURES], train_data[CANONICAL_PRICE_COLUMN].astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_prediction = float(model.predict(tender_df[FEATURES])[0])
        prediction, adjusted, adjustment_reason = usable_prediction(raw_prediction, median_prediction)
        base_description = (
            "Sayısal ve kategorik alanlardan doğrusal fiyat referansı."
            if name == "Linear Regression Baseline"
            else "Random Forest / ağaç tabanlı baseline; miktar, bölge ve ürün grubu ilişkilerini daha esnek okur."
        )
        description = (
            f"{base_description} Model {adjustment_reason} fiyat ürettiği için ürün grubu medyanı güvenli referans olarak kullanıldı."
            if adjusted
            else base_description
        )
        predictions.append(
            {
                "method": name,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "prediction_adjusted": adjusted,
                "description": description,
                "confidence": "Düşük" if adjusted else "Orta",
            }
        )
    return pd.DataFrame(predictions)
