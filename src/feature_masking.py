"""Feature masking for pseudo-live simulations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import ACTUAL_RESULT_FIELDS, BID_TIME_FIELDS
from .config_loader import load_hard_constraints


def configured_actual_result_fields() -> list[str]:
    try:
        leakage = load_hard_constraints().get("leakage", {})
        configured = leakage.get("blocked_fields_before_reveal", []) if isinstance(leakage, dict) else []
    except Exception:
        configured = []
    fields = [str(field) for field in configured if field]
    return list(dict.fromkeys([*ACTUAL_RESULT_FIELDS, *fields]))


def blocked_fields_present(columns: list[str] | pd.Index) -> list[str]:
    column_set = set(columns)
    return [field for field in configured_actual_result_fields() if field in column_set]


def mask_actual_result_fields(record: pd.Series | dict[str, Any] | pd.DataFrame) -> Any:
    if isinstance(record, pd.DataFrame):
        return record.drop(columns=blocked_fields_present(record.columns), errors="ignore")
    if isinstance(record, pd.Series):
        return record.drop(labels=blocked_fields_present(record.index), errors="ignore")
    return {key: value for key, value in record.items() if key not in ACTUAL_RESULT_FIELDS}


def select_bid_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    safe_columns = [column for column in BID_TIME_FIELDS if column in df.columns]
    return mask_actual_result_fields(df[safe_columns].copy())
