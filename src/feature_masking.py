"""Feature masking for pseudo-live simulations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import ACTUAL_RESULT_FIELDS, BID_TIME_FIELDS


def blocked_fields_present(columns: list[str] | pd.Index) -> list[str]:
    column_set = set(columns)
    return [field for field in ACTUAL_RESULT_FIELDS if field in column_set]


def mask_actual_result_fields(record: pd.Series | dict[str, Any] | pd.DataFrame) -> Any:
    if isinstance(record, pd.DataFrame):
        return record.drop(columns=blocked_fields_present(record.columns), errors="ignore")
    if isinstance(record, pd.Series):
        return record.drop(labels=blocked_fields_present(record.index), errors="ignore")
    return {key: value for key, value in record.items() if key not in ACTUAL_RESULT_FIELDS}


def select_bid_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    safe_columns = [column for column in BID_TIME_FIELDS if column in df.columns]
    return mask_actual_result_fields(df[safe_columns].copy())

