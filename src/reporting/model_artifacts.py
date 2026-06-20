"""Model artifact and metadata versioning for backtest runs."""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..clustering import ProfileFitModel
from ..config_loader import (
    load_app_config,
    load_hard_constraints,
    load_observability_config,
    load_scenario_weights,
    load_soft_penalties,
)
from ..evaluation.metrics import optimizer_metrics, price_corridor_metrics
from ..model_version import MODEL_VERSION
from ..retrieval import RetrievalEngine
from ..schema import normalize_schema
from .structured_logging import log_event, log_exception, model_version_string


def dataset_hash(df: pd.DataFrame) -> str:
    normalized = normalize_schema(df).sort_index(axis=1)
    csv = normalized.to_csv(index=False)
    return sha256(csv.encode("utf-8")).hexdigest()


def _date_range(df: pd.DataFrame) -> dict[str, str]:
    if df.empty or "tender_date" not in df:
        return {"start": "", "end": ""}
    dates = pd.to_datetime(df["tender_date"], errors="coerce").dropna()
    if dates.empty:
        return {"start": "", "end": ""}
    return {"start": dates.min().date().isoformat(), "end": dates.max().date().isoformat()}


def _feature_schema(df: pd.DataFrame) -> dict[str, list[str]]:
    normalized = normalize_schema(df)
    return {
        "columns": list(normalized.columns),
        "numeric_columns": [
            column
            for column in normalized.columns
            if pd.api.types.is_numeric_dtype(normalized[column])
        ],
        "categorical_columns": [
            column
            for column in normalized.columns
            if not pd.api.types.is_numeric_dtype(normalized[column])
        ],
    }


def write_backtest_artifacts(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: pd.DataFrame,
    top_k: int,
    run_id: str | None = None,
    artifact_dir: str | Path | None = None,
) -> Path:
    config = load_observability_config().get("artifacts", {})
    root = Path(artifact_dir or str(config.get("directory", "model_artifacts")))
    created_at = datetime.now(timezone.utc).isoformat()
    combined_hash = dataset_hash(pd.concat([train_df, test_df], ignore_index=True))
    resolved_run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{combined_hash[:8]}_{uuid.uuid4().hex[:6]}"
    run_dir = root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train = normalize_schema(train_df)
    test = normalize_schema(test_df)
    snapshot = {
        "app_config": load_app_config(),
        "hard_constraints": load_hard_constraints(),
        "soft_penalties": load_soft_penalties(),
        "observability": load_observability_config(),
    }
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=True), encoding="utf-8")

    split_manifest = {
        "run_id": resolved_run_id,
        "created_at": created_at,
        "training_date_range": _date_range(train),
        "test_date_range": _date_range(test),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_tender_ids": [str(value) for value in train.get("tender_id", pd.Series(dtype=str)).tolist()],
        "test_tender_ids": [str(value) for value in test.get("tender_id", pd.Series(dtype=str)).tolist()],
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "run_id": resolved_run_id,
        "model_version": model_version_string(),
        "config_version": MODEL_VERSION.get("config_version"),
        "training_date_range": split_manifest["training_date_range"],
        "test_date_range": split_manifest["test_date_range"],
        "created_at": created_at,
        "dataset_hash": combined_hash,
        "feature_schema": _feature_schema(pd.concat([train, test], ignore_index=True)),
        "top_k": int(top_k),
        "contamination": float(load_app_config().get("profile_fit", {}).get("isolation_contamination", 0.05)),
        "scenario_scoring_weights": load_scenario_weights(),
    }
    metrics = {
        "metadata": metadata,
        "price_corridor_metrics": price_corridor_metrics(results),
        "optimizer_metrics": optimizer_metrics(results),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    try:
        retriever = RetrievalEngine.fit(train)
        profile = ProfileFitModel.fit(train)
        with (run_dir / "retrieval_model.pkl").open("wb") as handle:
            pickle.dump(retriever, handle)
        with (run_dir / "profile_fit_model.pkl").open("wb") as handle:
            pickle.dump(profile, handle)
        with (run_dir / "cluster_model.pkl").open("wb") as handle:
            pickle.dump(profile.cluster_model, handle)
    except Exception:
        log_exception(
            "model_artifact_pickle_failed",
            module="artifacts",
            status="warning",
            message="Model pkl dosyaları yazılamadı; metadata ve metrics korundu.",
            run_id=resolved_run_id,
        )

    log_event(
        "model_artifact_written",
        module="artifacts",
        status="pass",
        message="Backtest model artifact bilgileri yazıldı.",
        run_id=resolved_run_id,
        artifact_dir=str(run_dir),
    )
    return run_dir
