from __future__ import annotations

from src.optimizer.scenario_scorer import score_scenario
from src.optimizer.scenario_validator import validate_scenario


def _tender() -> dict[str, object]:
    return {
        "tender_id": "T1",
        "tender_date": "2025-01-01",
        "product_name": "A",
        "product_group": "G",
        "buyer_institution": "B",
        "buyer_institution_type": "Kamu",
        "region": "R",
        "procedure_type": "P",
        "quantity": 10,
        "quantity_bucket": "Orta",
        "delivery_months": 6,
        "delivery_bucket": "Orta",
        "competitor_count_estimate": 3,
        "estimated_unit_cost": 80,
    }


def _profile(**overrides: object) -> dict[str, object]:
    profile = {
        "won_profile_fit_score": 75,
        "is_inlier": True,
        "cluster_score": 80,
        "topk_avg_similarity": 0.80,
    }
    profile.update(overrides)
    return profile


def _score(
    scenario: dict[str, object],
    tender: dict[str, object] | None = None,
    corridor: dict[str, float] | None = None,
    profile: dict[str, object] | None = None,
):
    tender = tender or _tender()
    corridor = corridor or {
        "predicted_low_price": 100,
        "predicted_mid_price": 120,
        "predicted_high_price": 140,
        "p10": 95,
        "p90": 135,
        "band_width": 40,
    }
    validation = validate_scenario(scenario, tender, corridor)
    return score_scenario(scenario, tender, corridor, profile or _profile(), 75, validation)


def test_low_similarity_soft_penalty_does_not_block_scenario():
    result = _score(
        {"scenario_id": "S1", "proposed_unit_price": 120, "estimated_unit_cost": 80, "delivery_months": 6},
        profile=_profile(topk_avg_similarity=0.30),
    )

    assert result["hard_constraints_valid"]
    assert any(item["rule"] == "low_similarity" for item in result["soft_penalties"])
    assert result["scenario_score"] > 0


def test_wide_band_soft_penalty_is_created():
    result = _score(
        {"scenario_id": "S1", "proposed_unit_price": 150, "estimated_unit_cost": 80, "delivery_months": 6},
        corridor={
            "predicted_low_price": 80,
            "predicted_mid_price": 150,
            "predicted_high_price": 220,
            "p10": 80,
            "p90": 200,
            "band_width": 140,
        },
    )

    assert result["hard_constraints_valid"]
    assert any(item["rule"] == "wide_band" for item in result["soft_penalties"])


def test_model_disagreement_soft_penalty_is_created():
    scenario = {
        "scenario_id": "S1",
        "proposed_unit_price": 120,
        "estimated_unit_cost": 80,
        "delivery_months": 6,
        "baseline_predictions": [{"prediction": 90}, {"prediction": 150}],
    }

    result = _score(scenario)

    assert result["hard_constraints_valid"]
    assert any(item["rule"] == "model_disagreement" for item in result["soft_penalties"])


def test_soft_penalty_lowers_score_without_invalidating():
    clean = _score({"scenario_id": "S1", "proposed_unit_price": 120, "estimated_unit_cost": 80, "delivery_months": 6})
    penalized = _score(
        {"scenario_id": "S1", "proposed_unit_price": 120, "estimated_unit_cost": 80, "delivery_months": 6},
        profile=_profile(topk_avg_similarity=0.30),
    )

    assert clean["hard_constraints_valid"]
    assert penalized["hard_constraints_valid"]
    assert penalized["risk_penalty_score"] > clean["risk_penalty_score"]
    assert penalized["scenario_score"] < clean["scenario_score"]
