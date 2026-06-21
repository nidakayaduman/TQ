from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.config_loader import load_observability_config
from src.evaluation.backtest_runner import run_backtest
from src.reporting.audit_log import AUDIT_LOG_SCHEMA, normalize_audit_event, validate_audit_log_event, write_audit_event
from src.reporting.model_artifacts import write_backtest_artifacts


def test_observability_config_file_exists_and_loads():
    path = Path("config/observability.yaml")
    assert path.exists()
    config = load_observability_config()
    assert config["logging"]["application_log"] == "app.jsonl"
    assert config["audit"]["default_user_id"] == "anonymous"
    assert config["artifacts"]["directory"] == "model_artifacts"


def test_audit_event_contains_required_fields(tmp_path):
    path = write_audit_event(
        {
            "event_type": "scenario_generated",
            "session_id": "test-session",
            "user_id": "anonymous",
            "tender_id": "T-1",
            "scenario_id": "S-1",
            "input_summary": "input",
            "output_summary": "output",
        },
        audit_dir=tmp_path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in AUDIT_LOG_SCHEMA:
        assert field in payload
    assert payload["input_hash"]
    assert payload["output_hash"]
    assert payload["reveal_status"] == "hidden"
    assert "user_id" not in payload
    assert payload["user_id_hash"]


def test_audit_event_schema_rejects_missing_required_field():
    event = normalize_audit_event({"event_type": "scenario_generated", "session_id": "s-1"})
    event.pop("event_type")
    result = validate_audit_log_event(event)
    assert not result["valid"]
    assert any("event_type" in error for error in result["errors"])


def test_audit_event_reveal_status_enum_is_enforced():
    event = normalize_audit_event({"event_type": "backtest_run", "reveal_status": "revealed_for_backtest"})
    assert event["reveal_status"] == "not_applicable"
    assert validate_audit_log_event(event)["valid"]

    event["reveal_status"] = "revealed_for_backtest"
    result = validate_audit_log_event(event)
    assert not result["valid"]
    assert any("enum" in error for error in result["errors"])


def test_audit_event_masks_hidden_actual_values():
    event = normalize_audit_event(
        {
            "event_type": "leakage_audit_result",
            "details": {"actual_margin_pct": 23.5, "safe": "ok"},
            "reveal_status": "hidden",
        }
    )
    assert "actual_margin_pct" not in event["details"]
    assert event["details"]["actual_margin_pct_masked"] is True
    assert event["details"]["actual_margin_pct_hash"]
    assert event["details"]["safe"] == "ok"


def test_backtest_artifacts_are_written(tmp_path, tiny_df):
    train = tiny_df[tiny_df["year"] <= 2024]
    test = tiny_df[tiny_df["year"] == 2025]
    results = run_backtest(train, test, top_k=6)

    run_dir = write_backtest_artifacts(
        train_df=train,
        test_df=test,
        results=results,
        top_k=6,
        run_id="test-run",
        artifact_dir=tmp_path,
    )

    assert (run_dir / "config_snapshot.yaml").exists()
    assert (run_dir / "split_manifest.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "retrieval_model.pkl").exists()
    assert (run_dir / "profile_fit_model.pkl").exists()
    assert (run_dir / "cluster_model.pkl").exists()

    snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "app_config" in snapshot
    assert metrics["metadata"]["run_id"] == "test-run"
    assert metrics["metadata"]["dataset_hash"]
    assert isinstance(metrics["metadata"]["scenario_scoring_weights"], dict)
