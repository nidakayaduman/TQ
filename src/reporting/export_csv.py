"""CSV export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import DISCLAIMER


def export_dataframe_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_disclaimer = df.copy()
    if "disclaimer" not in with_disclaimer.columns:
        with_disclaimer.insert(0, "disclaimer", DISCLAIMER)
    with_disclaimer.to_csv(output_path, index=False)
    return output_path


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    with_disclaimer = df.copy()
    if "disclaimer" not in with_disclaimer.columns:
        with_disclaimer.insert(0, "disclaimer", DISCLAIMER)
    return with_disclaimer.to_csv(index=False).encode("utf-8")
