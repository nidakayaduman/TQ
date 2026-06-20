"""Won profile fit models using Isolation Forest and K-Means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from .schema import normalize_schema

CATEGORICAL = ["product_group", "region", "procedure_type", "buyer_institution_type"]
NUMERIC = ["quantity", "delivery_months", "competitor_count_estimate", "estimated_unit_cost"]


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_schema(df)
    for column in NUMERIC:
        if column not in data:
            data[column] = 0.0
    for column in CATEGORICAL:
        if column not in data:
            data[column] = "Bilinmiyor"
    return data[[*CATEGORICAL, *NUMERIC]].copy()


def _profile_name(group: pd.DataFrame) -> str:
    product_group = str(group["product_group"].mode().iloc[0]) if not group["product_group"].mode().empty else "Karma"
    region = str(group["region"].mode().iloc[0]) if not group["region"].mode().empty else "Karma"
    quantity_label = "yüksek hacimli" if group["quantity"].median() >= group["quantity"].quantile(0.67) else "standart hacimli"
    return f"{product_group} / {region} / {quantity_label} profil"


@dataclass
class ProfileFitModel:
    preprocessor: ColumnTransformer
    isolation_model: IsolationForest
    historical_scores: np.ndarray
    cluster_preprocessor: ColumnTransformer
    cluster_model: KMeans
    cluster_profiles: dict[int, dict[str, Any]]
    cluster_distances: dict[int, np.ndarray]

    @classmethod
    def fit(cls, df: pd.DataFrame, n_clusters: int = 4) -> "ProfileFitModel":
        data = normalize_schema(df).reset_index(drop=True)
        features = _feature_frame(data)
        preprocessor = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("num", StandardScaler(), NUMERIC)]
        )
        encoded = preprocessor.fit_transform(features)
        isolation_model = IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
        isolation_model.fit(encoded)
        historical_scores = np.sort(isolation_model.decision_function(encoded))

        cluster_columns = ["product_group", "quantity", CANONICAL_PRICE_COLUMN, CANONICAL_MARGIN_COLUMN]
        cluster_frame = data[cluster_columns].copy()
        cluster_preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["product_group"]),
                ("num", StandardScaler(), ["quantity", CANONICAL_PRICE_COLUMN, CANONICAL_MARGIN_COLUMN]),
            ]
        )
        cluster_encoded = cluster_preprocessor.fit_transform(cluster_frame)
        cluster_count = min(n_clusters, max(1, len(data)))
        cluster_model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
        labels = cluster_model.fit_predict(cluster_encoded)
        distances = cluster_model.transform(cluster_encoded)
        assigned_distances = distances[np.arange(len(labels)), labels]
        profiles: dict[int, dict[str, Any]] = {}
        distance_map: dict[int, np.ndarray] = {}
        working = data.copy()
        working["cluster_id"] = labels
        for cluster_id, group in working.groupby("cluster_id"):
            cid = int(cluster_id)
            profiles[cid] = {
                "cluster_id": cid,
                "cluster_name": _profile_name(group),
                "count": int(len(group)),
                "dominant_product_group": str(group["product_group"].mode().iloc[0]) if not group["product_group"].mode().empty else "Karma",
                "median_price": float(group[CANONICAL_PRICE_COLUMN].median()),
                "median_margin": float(group[CANONICAL_MARGIN_COLUMN].median()),
            }
            distance_map[cid] = np.sort(assigned_distances[labels == cid])
        return cls(
            preprocessor=preprocessor,
            isolation_model=isolation_model,
            historical_scores=historical_scores,
            cluster_preprocessor=cluster_preprocessor,
            cluster_model=cluster_model,
            cluster_profiles=profiles,
            cluster_distances=distance_map,
        )

    def score(self, query: pd.Series | dict[str, Any], proposed_price: float | None = None, margin_pct: float | None = None) -> dict[str, Any]:
        query_frame = pd.DataFrame([dict(query)])
        features = _feature_frame(query_frame)
        encoded = self.preprocessor.transform(features)
        raw_score = float(self.isolation_model.decision_function(encoded)[0])
        percentile = float(np.searchsorted(self.historical_scores, raw_score, side="right") / len(self.historical_scores) * 100)
        inlier = bool(self.isolation_model.predict(encoded)[0] == 1)

        cluster_input = normalize_schema(query_frame)
        if proposed_price is None:
            proposed_price = float(cluster_input.get(CANONICAL_PRICE_COLUMN, pd.Series([0])).iloc[0] or 0)
        if margin_pct is None:
            margin_pct = float(cluster_input.get(CANONICAL_MARGIN_COLUMN, pd.Series([0])).iloc[0] or 0)
        cluster_frame = pd.DataFrame(
            [
                {
                    "product_group": cluster_input["product_group"].iloc[0],
                    "quantity": cluster_input["quantity"].iloc[0],
                    CANONICAL_PRICE_COLUMN: proposed_price,
                    CANONICAL_MARGIN_COLUMN: margin_pct,
                }
            ]
        )
        encoded_cluster = self.cluster_preprocessor.transform(cluster_frame)
        distances = self.cluster_model.transform(encoded_cluster)[0]
        cluster_id = int(np.argmin(distances))
        reference = self.cluster_distances.get(cluster_id, np.array([distances[cluster_id]]))
        distance_percentile = float(np.searchsorted(reference, distances[cluster_id], side="right") / max(len(reference), 1))
        cluster_score = float(np.clip((1 - distance_percentile) * 100, 0, 100))
        won_profile_fit_score = float(np.clip(0.65 * percentile + 0.35 * cluster_score, 0, 100))
        profile = self.cluster_profiles[cluster_id]
        return {
            "won_profile_fit_score": won_profile_fit_score,
            "is_inlier": inlier,
            "inlier_score": float(np.clip(percentile, 0, 100)),
            "cluster_id": cluster_id,
            "cluster_name": profile["cluster_name"],
            "cluster_score": cluster_score,
            "cluster_count": profile.get("count"),
            "cluster_dominant_product_group": profile.get("dominant_product_group"),
            "cluster_median_price": profile.get("median_price"),
            "cluster_median_margin": profile.get("median_margin"),
        }
