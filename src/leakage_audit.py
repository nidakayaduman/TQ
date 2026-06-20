"""Leakage checks for pre-reveal model inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .feature_masking import blocked_fields_present


def audit_pre_reveal_input(tender_id: str, model_input: pd.DataFrame | pd.Series | dict[str, Any]) -> dict[str, Any]:
    columns = list(model_input.columns) if isinstance(model_input, pd.DataFrame) else list(model_input.keys())
    blocked = blocked_fields_present(columns)
    return {
        "tender_id": str(tender_id),
        "leakage_detected": bool(blocked),
        "blocked_fields_present": blocked,
        "masked_fields_count": len(blocked),
        "audit_status": "fail" if blocked else "pass",
    }


def assert_no_leakage(tender_id: str, model_input: pd.DataFrame | pd.Series | dict[str, Any]) -> None:
    audit = audit_pre_reveal_input(tender_id, model_input)
    if audit["leakage_detected"]:
        raise ValueError(f"Leakage detected for {tender_id}: {audit['blocked_fields_present']}")

