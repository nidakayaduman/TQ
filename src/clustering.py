"""Won profile fit models using Isolation Forest and K-Means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config_loader import load_app_config
from .constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from .schema import normalize_schema

CATEGORICAL = ["product_group", "region", "procedure_type", "buyer_institution_type"]
NUMERIC = ["quantity", "delivery_months", "competitor_count_estimate", "estimated_unit_cost"]
CLUSTER_TRAINING_FEATURES = [*CATEGORICAL, *NUMERIC]
LIVE_ASSIGNMENT_FEATURES = [*CATEGORICAL, *NUMERIC]
DEFAULT_CONTAMINATION = 0.05


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _configured_contamination() -> float:
    try:
        config = load_app_config()
        value = float(config.get("profile_fit", {}).get("isolation_contamination", DEFAULT_CONTAMINATION))
    except Exception:
        value = DEFAULT_CONTAMINATION
    return float(np.clip(value, 0.01, 0.20))


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_schema(df)
    for column in NUMERIC:
        if column not in data:
            data[column] = 0.0
    for column in CATEGORICAL:
        if column not in data:
            data[column] = "Bilinmiyor"
    return data[[*CATEGORICAL, *NUMERIC]].copy()


def _dense_array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return np.asarray(value)


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
    cluster_encoded: np.ndarray
    cluster_labels: np.ndarray
    cluster_examples: pd.DataFrame
    cluster_quality: dict[str, Any]
    contamination: float
    training_inlier_rate: float
    training_anomaly_rate: float
    segment_anomaly_rates: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, n_clusters: int = 4, contamination: float | None = None) -> "ProfileFitModel":
        data = normalize_schema(df).reset_index(drop=True)
        features = _feature_frame(data)
        contamination_value = _configured_contamination() if contamination is None else float(np.clip(contamination, 0.01, 0.20))
        preprocessor = ColumnTransformer(
            [("cat", _one_hot_encoder(), CATEGORICAL), ("num", StandardScaler(), NUMERIC)]
        )
        encoded = _dense_array(preprocessor.fit_transform(features))
        isolation_model = IsolationForest(n_estimators=200, contamination=contamination_value, random_state=42)
        isolation_model.fit(encoded)
        historical_scores = np.sort(isolation_model.decision_function(encoded))
        training_predictions = isolation_model.predict(encoded)
        training_inlier_rate = float((training_predictions == 1).mean())
        training_anomaly_rate = float((training_predictions == -1).mean())
        segment_anomaly_rates = {
            str(segment): float((training_predictions[group.index.to_numpy()] == -1).mean())
            for segment, group in data.groupby("product_group")
        }

        cluster_frame = features[CLUSTER_TRAINING_FEATURES].copy()
        cluster_preprocessor = ColumnTransformer(
            [
                ("cat", _one_hot_encoder(), CATEGORICAL),
                ("num", StandardScaler(), NUMERIC),
            ]
        )
        cluster_encoded = _dense_array(cluster_preprocessor.fit_transform(cluster_frame))
        cluster_count = min(n_clusters, max(1, len(data)))
        cluster_model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            labels = cluster_model.fit_predict(cluster_encoded)
            distances = cluster_model.transform(cluster_encoded)
        distances = np.nan_to_num(distances, nan=np.finfo(float).max, posinf=np.finfo(float).max, neginf=np.finfo(float).max)
        assigned_distances = distances[np.arange(len(labels)), labels]
        silhouette = 0.0
        if cluster_count > 1 and len(data) > cluster_count:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                silhouette = float(silhouette_score(cluster_encoded, labels))
        size_counts = pd.Series(labels).value_counts().sort_index()
        small_cluster_count = int((size_counts < max(2, int(len(data) * 0.03))).sum())
        cluster_quality = {
            "cluster_silhouette_score": silhouette,
            "cluster_inertia": float(getattr(cluster_model, "inertia_", 0.0)),
            "cluster_count": int(cluster_count),
            "cluster_min_size": int(size_counts.min()) if not size_counts.empty else 0,
            "cluster_max_size": int(size_counts.max()) if not size_counts.empty else 0,
            "small_cluster_count": small_cluster_count,
            "empty_cluster_count": int(max(cluster_count - size_counts.size, 0)),
        }
        profiles: dict[int, dict[str, Any]] = {}
        distance_map: dict[int, np.ndarray] = {}
        working = data.copy()
        working["cluster_id"] = labels
        working["distance_to_cluster_center"] = assigned_distances
        for cluster_id, group in working.groupby("cluster_id"):
            cid = int(cluster_id)
            dominant_product = str(group["product_group"].mode().iloc[0]) if not group["product_group"].mode().empty else "Karma"
            dominant_institution = str(group["buyer_institution_type"].mode().iloc[0]) if not group["buyer_institution_type"].mode().empty else "Karma"
            dominant_region = str(group["region"].mode().iloc[0]) if not group["region"].mode().empty else "Karma"
            profiles[cid] = {
                "cluster_id": cid,
                "cluster_name": _profile_name(group),
                "count": int(len(group)),
                "dominant_product_group": dominant_product,
                "dominant_product_group_ratio": float((group["product_group"] == dominant_product).mean()),
                "dominant_institution_type": dominant_institution,
                "dominant_institution_type_ratio": float((group["buyer_institution_type"] == dominant_institution).mean()),
                "dominant_region": dominant_region,
                "dominant_region_ratio": float((group["region"] == dominant_region).mean()),
                "average_quantity": float(group["quantity"].mean()),
                "median_delivery_months": float(group["delivery_months"].median()),
                "average_price": float(group[CANONICAL_PRICE_COLUMN].mean()),
                "average_margin": float(group[CANONICAL_MARGIN_COLUMN].mean()),
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
            cluster_encoded=np.asarray(cluster_encoded),
            cluster_labels=np.asarray(labels),
            cluster_examples=working,
            cluster_quality=cluster_quality,
            contamination=contamination_value,
            training_inlier_rate=training_inlier_rate,
            training_anomaly_rate=training_anomaly_rate,
            segment_anomaly_rates=segment_anomaly_rates,
        )

    def score(self, query: pd.Series | dict[str, Any], proposed_price: float | None = None, margin_pct: float | None = None) -> dict[str, Any]:
        query_frame = pd.DataFrame([dict(query)])
        features = _feature_frame(query_frame)
        encoded = _dense_array(self.preprocessor.transform(features))
        raw_score = float(self.isolation_model.decision_function(encoded)[0])
        percentile = float(np.searchsorted(self.historical_scores, raw_score, side="right") / len(self.historical_scores) * 100)
        inlier = bool(self.isolation_model.predict(encoded)[0] == 1)

        cluster_frame = features[LIVE_ASSIGNMENT_FEATURES].copy()
        encoded_cluster = _dense_array(self.cluster_preprocessor.transform(cluster_frame))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            distances = self.cluster_model.transform(encoded_cluster)[0]
        distances = np.nan_to_num(distances, nan=np.finfo(float).max, posinf=np.finfo(float).max, neginf=np.finfo(float).max)
        cluster_id = int(np.argmin(distances))
        ordered_distances = np.sort(distances)
        nearest_distance = float(ordered_distances[0]) if len(ordered_distances) else 0.0
        second_distance = float(ordered_distances[1]) if len(ordered_distances) > 1 else nearest_distance
        assignment_confidence = 100.0 if second_distance <= 0 else float(np.clip((1 - nearest_distance / second_distance) * 100, 0, 100))
        reference = self.cluster_distances.get(cluster_id, np.array([distances[cluster_id]]))
        distance_percentile = float(np.searchsorted(reference, distances[cluster_id], side="right") / max(len(reference), 1))
        cluster_score = float(np.clip((1 - distance_percentile) * 100, 0, 100))
        won_profile_fit_score = float(np.clip(0.65 * percentile + 0.35 * cluster_score, 0, 100))
        profile = self.cluster_profiles[cluster_id]
        member_mask = self.cluster_labels == cluster_id
        member_indices = np.where(member_mask)[0]
        nearest_examples: list[dict[str, Any]] = []
        if len(member_indices):
            member_vectors = self.cluster_encoded[member_indices]
            member_distances = np.linalg.norm(member_vectors - np.asarray(encoded_cluster)[0], axis=1)
            safe_columns = [
                "tender_id",
                "product_group",
                "product_name",
                "buyer_institution_type",
                "region",
                "procedure_type",
                "quantity",
                "delivery_months",
                "distance_to_cluster_center",
            ]
            for order_idx in np.argsort(member_distances)[:5]:
                row = self.cluster_examples.iloc[int(member_indices[order_idx])]
                nearest_examples.append(
                    {
                        key: row.get(key)
                        for key in safe_columns
                        if key in self.cluster_examples.columns
                    }
                    | {"query_distance": float(member_distances[order_idx])}
                )
        return {
            "won_profile_fit_score": won_profile_fit_score,
            "is_inlier": inlier,
            "inlier_score": float(np.clip(percentile, 0, 100)),
            "anomaly_score": raw_score,
            "isolation_threshold": 0.0,
            "manual_review_flag": not inlier,
            "cluster_id": cluster_id,
            "cluster_name": profile["cluster_name"],
            "cluster_score": cluster_score,
            "cluster_assignment_confidence": assignment_confidence,
            "cluster_distance": nearest_distance,
            "cluster_second_distance": second_distance,
            "cluster_distance_percentile": distance_percentile,
            "cluster_count": profile.get("count"),
            "cluster_dominant_product_group": profile.get("dominant_product_group"),
            "cluster_dominant_product_group_ratio": profile.get("dominant_product_group_ratio"),
            "cluster_dominant_institution_type": profile.get("dominant_institution_type"),
            "cluster_dominant_institution_type_ratio": profile.get("dominant_institution_type_ratio"),
            "cluster_dominant_region": profile.get("dominant_region"),
            "cluster_dominant_region_ratio": profile.get("dominant_region_ratio"),
            "cluster_average_quantity": profile.get("average_quantity"),
            "cluster_median_delivery_months": profile.get("median_delivery_months"),
            "cluster_average_price": profile.get("average_price"),
            "cluster_average_margin": profile.get("average_margin"),
            "cluster_median_price": profile.get("median_price"),
            "cluster_median_margin": profile.get("median_margin"),
            "nearest_cluster_examples": nearest_examples,
            **self.cluster_quality,
            "isolation_contamination": self.contamination,
            "training_inlier_rate": self.training_inlier_rate,
            "training_anomaly_rate": self.training_anomaly_rate,
            "segment_anomaly_rate": self.segment_anomaly_rates.get(str(features["product_group"].iloc[0])),
        }
