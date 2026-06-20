from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.schema import normalize_schema


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    path = Path("data/x_ilac_synthetic_tenders_2021_2025.csv")
    return normalize_schema(pd.read_csv(path))


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    return normalize_schema(
        pd.DataFrame(
            [
                {
                    "tender_id": f"T-{year}-{idx}",
                    "year": year,
                    "tender_date": f"{year}-03-01",
                    "product_name": "Serum A",
                    "product_group": "IV Solution" if idx % 2 == 0 else "Injectable",
                    "buyer_institution": "Kamu Hastanesi",
                    "buyer_institution_type": "Kamu",
                    "region": "Marmara" if idx % 2 == 0 else "Ege",
                    "procedure_type": "Açık İhale",
                    "quantity": 1000 + idx * 200,
                    "delivery_months": 6,
                    "competitor_count_estimate": 3,
                    "estimated_unit_cost": 8 + idx,
                    "actual_won_unit_price": 12 + idx + (year - 2021),
                    "actual_margin_pct": 20,
                    "actual_won_total_amount": (12 + idx) * (1000 + idx * 200),
                }
                for year in [2021, 2022, 2023, 2024, 2025]
                for idx in range(4)
            ]
        )
    )

