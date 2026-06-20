"""Shared constants for Tender IQ."""

from __future__ import annotations

DISCLAIMER = (
    "Bu skor gerçek kazanma olasılığı değildir. Geçmişte kazanılmış ihalelere "
    "benzerlik, fiyat bandı uyumu, karlılık/risk dengesi ve model güvenini "
    "gösterir. Gerçek kazanma olasılığı için güvenilir kazanılmış ve "
    "kaybedilmiş ihale verisi gerekir."
)

BID_TIME_FIELDS = [
    "tender_id",
    "tender_date",
    "year",
    "product_name",
    "product_group",
    "buyer_institution",
    "buyer_institution_type",
    "region",
    "procedure_type",
    "quantity",
    "quantity_bucket",
    "delivery_months",
    "delivery_bucket",
    "competitor_count_estimate",
    "estimated_unit_cost",
    "estimated_unit_cost_try",
]

ACTUAL_RESULT_FIELDS = [
    "actual_won_unit_price",
    "actual_won_total_amount",
    "won_unit_price",
    "won_total_amount",
    "actual_margin_pct",
    "actual_unit_margin",
    "final_contract_result",
    "final_contract_amount",
    "actual_delivery_result",
    "actual_award_result",
    "revealed_actual_result",
    "contract_award_date",
    "winning_unit_price_try",
    "contract_value_try",
    "gross_margin_pct",
    "gross_profit_try",
    "result",
    "contract_date",
]

REQUIRED_BASE_COLUMNS = [
    "tender_id",
    "tender_date",
    "product_name",
    "product_group",
    "buyer_institution",
    "region",
    "procedure_type",
    "quantity",
    "delivery_months",
    "competitor_count_estimate",
]

CANONICAL_PRICE_COLUMN = "actual_won_unit_price"
CANONICAL_MARGIN_COLUMN = "actual_margin_pct"
CANONICAL_TOTAL_COLUMN = "actual_won_total_amount"

FORBIDDEN_CLAIM_TERMS = [
    "p" + "_win",
    "win" + "_" + "probability",
    "probability" + "_of_winning",
    "true" + "_" + "win" + "_" + "probability",
    "guaranteed" + "_win",
    "p(" + "win)",
    "kazanma olasılığı",
    "gerçek kazanma olasılığı",
    "kesin kazanır",
    "kesin alınır",
    "garanti",
    "kazanım garantisi",
    "award is guaranteed",
    "probability of winning",
    "competitor will bid",
    "rakip şu fiyatı verecek",
    "bu teklif kazanır",
    "ihale kesin kazanılır",
]

SAFE_SCORE_COLUMNS = [
    "won_profile_fit_score",
    "price_band_fit_score",
    "margin_score",
    "risk_score",
    "model_confidence_score",
    "scenario_score",
]
