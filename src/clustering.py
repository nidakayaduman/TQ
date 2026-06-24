"""Won profile fit models using Isolation Forest, Top-K retrieval evidence, and mixed-type clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config_loader import load_app_config
from .constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from .retrieval import RetrievalEngine, retrieval_quality
from .schema import normalize_schema

CATEGORICAL = ["product_group", "product_name", "buyer_institution", "region", "procedure_type", "buyer_institution_type"]
NUMERIC = ["quantity", "delivery_months", "competitor_count_estimate"]
CLUSTER_TRAINING_FEATURES = [*CATEGORICAL, *NUMERIC]
LIVE_ASSIGNMENT_FEATURES = [*CATEGORICAL, *NUMERIC]
DEFAULT_CONTAMINATION = 0.05
DEFAULT_PROFILE_WEIGHTS = {"topk": 0.50, "isolation": 0.35, "cluster": 0.15}
LOW_CLUSTER_PURITY_THRESHOLD = 0.50
SEGMENT_CALIBRATION_MIN_SIZE = 8
SEGMENT_MODEL_MIN_SIZE = 12
PROFILE_NEIGHBOR_MIN_SIZE = 5
PROFILE_NEIGHBOR_K = 10


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


def _profile_weights() -> dict[str, float]:
    try:
        configured = load_app_config().get("profile_fit", {}).get("score_weights", {})
    except Exception:
        configured = {}
    weights = {
        key: float(configured.get(key, default))
        for key, default in DEFAULT_PROFILE_WEIGHTS.items()
    }
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        return DEFAULT_PROFILE_WEIGHTS.copy()
    return {key: max(value, 0.0) / total for key, value in weights.items()}


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


def _key(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _percentile_from_scores(scores: np.ndarray, raw_score: float) -> float:
    if len(scores) == 0:
        return 0.0
    return float(np.searchsorted(scores, raw_score, side="right") / len(scores) * 100)


def _nearest_mean_distance(distances: np.ndarray, k: int = PROFILE_NEIGHBOR_K) -> float:
    values = np.asarray(distances, dtype=float)
    if len(values) == 0:
        return 1.0
    ordered = np.sort(values)
    return float(ordered[: min(k, len(ordered))].mean())


def _safe_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in NUMERIC:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0).astype(float)
    for column in CATEGORICAL:
        output[column] = output[column].fillna("Bilinmiyor").astype(str)
    return output


def _numeric_ranges(features: pd.DataFrame) -> dict[str, float]:
    ranges: dict[str, float] = {}
    for column in NUMERIC:
        values = pd.to_numeric(features[column], errors="coerce").fillna(0.0).astype(float)
        ranges[column] = max(float(values.max() - values.min()), 1.0)
    return ranges


def gower_distance_matrix(features: pd.DataFrame) -> np.ndarray:
    """Compute a simple Gower distance matrix for categorical + numeric tender profile fields."""
    data = _safe_numeric_frame(features)
    n = len(data)
    distances = np.zeros((n, n), dtype=float)
    ranges = _numeric_ranges(data)
    feature_count = max(len(CATEGORICAL) + len(NUMERIC), 1)
    for column in CATEGORICAL:
        values = data[column].astype(str).to_numpy()
        distances += (values[:, None] != values[None, :]).astype(float)
    for column in NUMERIC:
        values = data[column].astype(float).to_numpy()
        distances += np.abs(values[:, None] - values[None, :]) / ranges[column]
    return np.clip(distances / feature_count, 0, 1)


def gower_distance_to_frame(query: pd.Series | dict[str, Any], features: pd.DataFrame, ranges: dict[str, float]) -> np.ndarray:
    data = _safe_numeric_frame(features)
    query_data = _safe_numeric_frame(pd.DataFrame([dict(query)]))
    distances = np.zeros(len(data), dtype=float)
    feature_count = max(len(CATEGORICAL) + len(NUMERIC), 1)
    for column in CATEGORICAL:
        distances += (data[column].astype(str).to_numpy() != str(query_data[column].iloc[0])).astype(float)
    for column in NUMERIC:
        query_value = float(query_data[column].iloc[0])
        distances += np.abs(data[column].astype(float).to_numpy() - query_value) / max(ranges.get(column, 1.0), 1.0)
    return np.clip(distances / feature_count, 0, 1)


def _profile_name(group: pd.DataFrame, product_ratio: float | None = None, region_ratio: float | None = None) -> str:
    if (
        product_ratio is not None
        and region_ratio is not None
        and min(product_ratio, region_ratio) < LOW_CLUSTER_PURITY_THRESHOLD
    ):
        return "Karma profil"
    product_group = str(group["product_group"].mode().iloc[0]) if not group["product_group"].mode().empty else "Karma"
    region = str(group["region"].mode().iloc[0]) if not group["region"].mode().empty else "Karma"
    quantity_label = "yüksek hacimli" if group["quantity"].median() >= group["quantity"].quantile(0.67) else "standart hacimli"
    return f"{product_group} / {region} / {quantity_label} profil"


@dataclass
class ProfileFitModel:
    preprocessor: ColumnTransformer
    isolation_model: IsolationForest
    historical_scores: np.ndarray
    product_isolation_models: dict[str, IsolationForest]
    product_isolation_scores: dict[str, np.ndarray]
    group_isolation_models: dict[str, IsolationForest]
    group_isolation_scores: dict[str, np.ndarray]
    cluster_model: AgglomerativeClustering
    cluster_profiles: dict[int, dict[str, Any]]
    cluster_distances: dict[int, np.ndarray]
    cluster_labels: np.ndarray
    cluster_examples: pd.DataFrame
    cluster_features: pd.DataFrame
    gower_ranges: dict[str, float]
    cluster_quality: dict[str, Any]
    retriever: RetrievalEngine
    score_weights: dict[str, float]
    contamination: float
    training_inlier_rate: float
    training_anomaly_rate: float
    segment_anomaly_rates: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, n_clusters: int = 4, contamination: float | None = None) -> "ProfileFitModel":
        data = normalize_schema(df).reset_index(drop=True)
        features = _feature_frame(data)
        contamination_value = _configured_contamination() if contamination is None else float(np.clip(contamination, 0.01, 0.20))
        score_weights = _profile_weights()
        preprocessor = ColumnTransformer(
            [("cat", _one_hot_encoder(), CATEGORICAL), ("num", StandardScaler(), NUMERIC)]
        )
        encoded = _dense_array(preprocessor.fit_transform(features))
        isolation_model = IsolationForest(n_estimators=200, contamination=contamination_value, random_state=42)
        isolation_model.fit(encoded)
        training_scores = isolation_model.decision_function(encoded)
        historical_scores = np.sort(training_scores)
        product_isolation_models: dict[str, IsolationForest] = {}
        product_isolation_scores: dict[str, np.ndarray] = {}
        for product, group in data.groupby("product_name"):
            if len(group) < SEGMENT_MODEL_MIN_SIZE:
                continue
            indices = group.index.to_numpy()
            segment_model = IsolationForest(n_estimators=200, contamination=contamination_value, random_state=42)
            segment_model.fit(encoded[indices])
            product_isolation_models[_key(product)] = segment_model
            product_isolation_scores[_key(product)] = np.sort(segment_model.decision_function(encoded[indices]))
        group_isolation_models: dict[str, IsolationForest] = {}
        group_isolation_scores: dict[str, np.ndarray] = {}
        for segment, group in data.groupby("product_group"):
            if len(group) < SEGMENT_MODEL_MIN_SIZE:
                continue
            indices = group.index.to_numpy()
            segment_model = IsolationForest(n_estimators=200, contamination=contamination_value, random_state=42)
            segment_model.fit(encoded[indices])
            group_isolation_models[_key(segment)] = segment_model
            group_isolation_scores[_key(segment)] = np.sort(segment_model.decision_function(encoded[indices]))
        training_predictions = isolation_model.predict(encoded)
        training_inlier_rate = float((training_predictions == 1).mean())
        training_anomaly_rate = float((training_predictions == -1).mean())
        segment_anomaly_rates = {
            str(segment): float((training_predictions[group.index.to_numpy()] == -1).mean())
            for segment, group in data.groupby("product_group")
        }

        cluster_frame = _safe_numeric_frame(features[CLUSTER_TRAINING_FEATURES].copy())
        gower_distances = gower_distance_matrix(cluster_frame)
        gower_ranges = _numeric_ranges(cluster_frame)
        cluster_count = min(n_clusters, max(1, len(data)))
        cluster_model = AgglomerativeClustering(n_clusters=cluster_count, metric="precomputed", linkage="average")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            labels = cluster_model.fit_predict(gower_distances) if len(data) else np.array([], dtype=int)
        labels = np.asarray(labels, dtype=int)
        assigned_distances = np.zeros(len(labels), dtype=float)
        for idx, label in enumerate(labels):
            member_indices = np.where(labels == label)[0]
            assigned_distances[idx] = float(gower_distances[idx, member_indices].mean()) if len(member_indices) else 0.0
        silhouette = 0.0
        if cluster_count > 1 and len(data) > cluster_count:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                silhouette = float(silhouette_score(gower_distances, labels, metric="precomputed"))
        size_counts = pd.Series(labels).value_counts().sort_index()
        small_cluster_count = int((size_counts < max(2, int(len(data) * 0.03))).sum())
        cluster_quality = {
            "cluster_silhouette_score": silhouette,
            "cluster_inertia": float(assigned_distances.sum()),
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
            dominant_procedure = str(group["procedure_type"].mode().iloc[0]) if not group["procedure_type"].mode().empty else "Karma"
            product_ratio = float((group["product_group"] == dominant_product).mean())
            institution_ratio = float((group["buyer_institution_type"] == dominant_institution).mean())
            region_ratio = float((group["region"] == dominant_region).mean())
            procedure_ratio = float((group["procedure_type"] == dominant_procedure).mean())
            purity_score = float(np.mean([product_ratio, institution_ratio, region_ratio, procedure_ratio]) * 100)
            profiles[cid] = {
                "cluster_id": cid,
                "cluster_name": _profile_name(group, product_ratio, region_ratio),
                "count": int(len(group)),
                "dominant_product_group": dominant_product,
                "dominant_product_group_ratio": product_ratio,
                "dominant_institution_type": dominant_institution,
                "dominant_institution_type_ratio": institution_ratio,
                "dominant_region": dominant_region,
                "dominant_region_ratio": region_ratio,
                "dominant_procedure_type": dominant_procedure,
                "dominant_procedure_type_ratio": procedure_ratio,
                "cluster_purity_score": purity_score,
                "average_quantity": float(group["quantity"].mean()),
                "median_delivery_months": float(group["delivery_months"].median()),
                "average_price": float(group[CANONICAL_PRICE_COLUMN].mean()),
                "average_margin": float(group[CANONICAL_MARGIN_COLUMN].mean()),
                "median_price": float(group[CANONICAL_PRICE_COLUMN].median()),
                "median_margin": float(group[CANONICAL_MARGIN_COLUMN].median()),
            }
            distance_map[cid] = np.sort(assigned_distances[labels == cid])
        retriever = RetrievalEngine.fit(data)
        return cls(
            preprocessor=preprocessor,
            isolation_model=isolation_model,
            historical_scores=historical_scores,
            product_isolation_models=product_isolation_models,
            product_isolation_scores=product_isolation_scores,
            group_isolation_models=group_isolation_models,
            group_isolation_scores=group_isolation_scores,
            cluster_model=cluster_model,
            cluster_profiles=profiles,
            cluster_distances=distance_map,
            cluster_labels=np.asarray(labels),
            cluster_examples=working,
            cluster_features=cluster_frame,
            gower_ranges=gower_ranges,
            cluster_quality=cluster_quality,
            retriever=retriever,
            score_weights=score_weights,
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
        global_percentile = _percentile_from_scores(self.historical_scores, raw_score)
        product_key = _key(features["product_name"].iloc[0])
        group_key = _key(features["product_group"].iloc[0])
        product_model = self.product_isolation_models.get(product_key)
        group_model = self.group_isolation_models.get(group_key)
        product_scores = self.product_isolation_scores.get(product_key, np.array([]))
        group_scores = self.group_isolation_scores.get(group_key, np.array([]))
        segment_percentile = global_percentile
        segment_inlier = bool(self.isolation_model.predict(encoded)[0] == 1)
        isolation_calibration_scope = "global"
        if product_model is not None and len(product_scores) >= SEGMENT_CALIBRATION_MIN_SIZE:
            product_raw_score = float(product_model.decision_function(encoded)[0])
            segment_percentile = _percentile_from_scores(product_scores, product_raw_score)
            segment_inlier = bool(product_model.predict(encoded)[0] == 1)
            isolation_calibration_scope = "product_name"
        elif group_model is not None and len(group_scores) >= SEGMENT_CALIBRATION_MIN_SIZE:
            group_raw_score = float(group_model.decision_function(encoded)[0])
            segment_percentile = _percentile_from_scores(group_scores, group_raw_score)
            segment_inlier = bool(group_model.predict(encoded)[0] == 1)
            isolation_calibration_scope = "product_group"
        percentile = float(max(global_percentile, segment_percentile))
        inlier = bool(self.isolation_model.predict(encoded)[0] == 1 or segment_inlier)

        similar = self.retriever.retrieve(query, top_k=min(50, max(len(self.cluster_examples), 1)))
        retrieval = retrieval_quality(similar, query, top_k=min(50, max(len(similar), 1)))
        topk_profile_score = float(np.clip(retrieval.get("topk_avg_similarity", 0.0) * 100, 0, 100))

        query_distances = gower_distance_to_frame(features.iloc[0].to_dict(), self.cluster_features, self.gower_ranges)
        cluster_average_distances = []
        for cluster_key in sorted(self.cluster_profiles):
            member_indices = np.where(self.cluster_labels == cluster_key)[0]
            cluster_average_distances.append(float(query_distances[member_indices].mean()) if len(member_indices) else 1.0)
        distances = np.asarray(cluster_average_distances, dtype=float)
        cluster_keys = sorted(self.cluster_profiles)
        cluster_pos = int(np.argmin(distances)) if len(distances) else 0
        cluster_id = int(cluster_keys[cluster_pos]) if cluster_keys else 0
        ordered_distances = np.sort(distances)
        nearest_distance = float(ordered_distances[0]) if len(ordered_distances) else 0.0
        second_distance = float(ordered_distances[1]) if len(ordered_distances) > 1 else nearest_distance
        assignment_confidence = 100.0 if second_distance <= 0 else float(np.clip((1 - nearest_distance / second_distance) * 100, 0, 100))
        reference = self.cluster_distances.get(cluster_id, np.array([nearest_distance]))
        distance_percentile = float(np.searchsorted(reference, nearest_distance, side="right") / max(len(reference), 1))
        cluster_score = float(np.clip((1 - distance_percentile) * 100, 0, 100))
        product_mask = self.cluster_features["product_name"].astype(str).to_numpy() == str(features["product_name"].iloc[0])
        group_mask = self.cluster_features["product_group"].astype(str).to_numpy() == str(features["product_group"].iloc[0])
        profile_scope = "all"
        profile_pool = np.ones(len(query_distances), dtype=bool)
        if int(product_mask.sum()) >= PROFILE_NEIGHBOR_MIN_SIZE:
            profile_pool = product_mask
            profile_scope = "product_name"
        elif int(group_mask.sum()) >= PROFILE_NEIGHBOR_MIN_SIZE:
            profile_pool = group_mask
            profile_scope = "product_group"
        nearest_profile_distance = _nearest_mean_distance(query_distances[profile_pool])
        nearest_profile_density_score = float(np.clip((1 - nearest_profile_distance) * 100, 0, 100))
        mixed_cluster_score = float(max(cluster_score, nearest_profile_density_score))
        profile = self.cluster_profiles[cluster_id]
        purity_score = float(profile.get("cluster_purity_score", 0.0) or 0.0)
        cluster_component = float(np.clip(0.70 * mixed_cluster_score + 0.30 * purity_score, 0, 100))
        won_profile_fit_score = float(
            np.clip(
                self.score_weights["topk"] * topk_profile_score
                + self.score_weights["isolation"] * float(np.clip(percentile, 0, 100))
                + self.score_weights["cluster"] * cluster_component,
                0,
                100,
            )
        )
        member_mask = self.cluster_labels == cluster_id
        member_indices = np.where(member_mask)[0]
        nearest_examples: list[dict[str, Any]] = []
        if len(member_indices):
            member_distances = query_distances[member_indices]
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
        manual_review_reasons: list[str] = []
        if not inlier:
            manual_review_reasons.append("Isolation Forest sıra dışı sinyali")
        if percentile < 20:
            manual_review_reasons.append("Düşük Isolation Forest profil yüzdesi")
        if topk_profile_score < 45:
            manual_review_reasons.append("Düşük Top-K emsal benzerliği")
        if won_profile_fit_score < 45:
            manual_review_reasons.append("Düşük yapısal profil uyumu")
        if purity_score < LOW_CLUSTER_PURITY_THRESHOLD * 100:
            manual_review_reasons.append("Karma veya düşük saflıkta profil kümesi")
        return {
            "won_profile_fit_score": won_profile_fit_score,
            "is_inlier": inlier,
            "inlier_score": float(np.clip(percentile, 0, 100)),
            "anomaly_score": raw_score,
            "isolation_threshold": 0.0,
            "global_inlier_score": float(np.clip(global_percentile, 0, 100)),
            "segment_inlier_score": float(np.clip(segment_percentile, 0, 100)),
            "isolation_calibration_scope": isolation_calibration_scope,
            "manual_review_flag": bool(manual_review_reasons),
            "manual_review_reasons": manual_review_reasons,
            "topk_profile_score": topk_profile_score,
            "mixed_cluster_score": mixed_cluster_score,
            "nearest_profile_density_score": nearest_profile_density_score,
            "nearest_profile_distance": nearest_profile_distance,
            "nearest_profile_scope": profile_scope,
            "cluster_purity_score": purity_score,
            "profile_score_components": {
                "topk_profile_score": topk_profile_score,
                "inlier_score": float(np.clip(percentile, 0, 100)),
                "global_inlier_score": float(np.clip(global_percentile, 0, 100)),
                "segment_inlier_score": float(np.clip(segment_percentile, 0, 100)),
                "mixed_cluster_score": mixed_cluster_score,
                "nearest_profile_density_score": nearest_profile_density_score,
                "cluster_purity_score": purity_score,
                "cluster_component_score": cluster_component,
                "weights": self.score_weights,
            },
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
            "cluster_dominant_procedure_type": profile.get("dominant_procedure_type"),
            "cluster_dominant_procedure_type_ratio": profile.get("dominant_procedure_type_ratio"),
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
