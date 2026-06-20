from src.optimizer.scenario_validator import validate_scenario


def test_scenario_validator_blocks_low_margin():
    tender = {
        "tender_id": "T1",
        "tender_date": "2025-01-01",
        "product_name": "A",
        "product_group": "G",
        "buyer_institution": "B",
        "region": "R",
        "procedure_type": "P",
        "quantity": 10,
        "delivery_months": 6,
        "estimated_unit_cost": 100,
    }
    scenario = {"proposed_unit_price": 101, "estimated_unit_cost": 100, "delivery_months": 6}
    corridor = {"predicted_high_price": 130, "p90": 130}
    result = validate_scenario(scenario, tender, corridor)
    assert not result["valid"]
    assert result["violations"]
    assert "minimum marj eşiğinin altında" in result["violations"][0]


def test_scenario_validator_returns_explainable_output():
    tender = {
        "tender_id": "T1",
        "tender_date": "2025-01-01",
        "product_name": "A",
        "product_group": "G",
        "buyer_institution": "B",
        "region": "R",
        "procedure_type": "P",
        "quantity": 10,
        "delivery_months": 6,
        "estimated_unit_cost": 100,
    }
    scenario = {"scenario_id": "S001", "proposed_unit_price": 130, "estimated_unit_cost": 100, "delivery_months": 6}
    corridor = {"predicted_low_price": 110, "predicted_high_price": 140, "p10": 105, "p90": 135}
    result = validate_scenario(scenario, tender, corridor)
    assert result["scenario_id"] == "S001"
    assert result["is_valid"]
    assert result["hard_constraint_violations"] == []
    assert "explainability" in result


def test_scenario_validator_blocks_price_above_p90_without_override():
    tender = {
        "tender_id": "T1",
        "tender_date": "2025-01-01",
        "product_name": "A",
        "product_group": "G",
        "buyer_institution": "B",
        "region": "R",
        "procedure_type": "P",
        "quantity": 10,
        "delivery_months": 6,
        "estimated_unit_cost": 80,
    }
    scenario = {"scenario_id": "S_HIGH", "proposed_unit_price": 116, "estimated_unit_cost": 80, "delivery_months": 6}
    corridor = {"predicted_low_price": 90, "predicted_high_price": 110, "p10": 90, "p90": 100}

    result = validate_scenario(scenario, tender, corridor)

    assert not result["is_valid"]
    assert any("üst fiyat seviyesinin belirgin şekilde üzerine" in item for item in result["hard_constraint_violations"])


def test_scenario_validator_blocks_price_below_p10_without_override():
    tender = {
        "tender_id": "T1",
        "tender_date": "2025-01-01",
        "product_name": "A",
        "product_group": "G",
        "buyer_institution": "B",
        "region": "R",
        "procedure_type": "P",
        "quantity": 10,
        "delivery_months": 6,
        "estimated_unit_cost": 10,
    }
    scenario = {"scenario_id": "S_LOW", "proposed_unit_price": 84, "estimated_unit_cost": 10, "delivery_months": 6}
    corridor = {"predicted_low_price": 95, "predicted_high_price": 120, "p10": 100, "p90": 115}

    result = validate_scenario(scenario, tender, corridor)

    assert not result["is_valid"]
    assert any("alt fiyat seviyesinin çok altında" in item for item in result["hard_constraint_violations"])


def test_scenario_validator_blocks_missing_required_field():
    tender = {
        "tender_id": "T1",
        "tender_date": "2025-01-01",
        "product_name": "A",
        "product_group": "G",
        "buyer_institution": "B",
        "region": "R",
        "procedure_type": "P",
        "quantity": 10,
        "delivery_months": 6,
    }
    scenario = {"scenario_id": "S_MISSING", "proposed_unit_price": 130, "delivery_months": 6}
    corridor = {"predicted_low_price": 100, "predicted_high_price": 140, "p10": 100, "p90": 130}

    result = validate_scenario(scenario, tender, corridor)

    assert not result["is_valid"]
    assert any("Zorunlu veri alanları eksik" in item for item in result["hard_constraint_violations"])


def test_scenario_validator_blocks_pre_reveal_leakage_field():
    tender = {
        "tender_id": "T1",
        "tender_date": "2025-01-01",
        "product_name": "A",
        "product_group": "G",
        "buyer_institution": "B",
        "region": "R",
        "procedure_type": "P",
        "quantity": 10,
        "delivery_months": 6,
        "estimated_unit_cost": 80,
        "actual_margin_pct": 20,
    }
    scenario = {"scenario_id": "S_LEAK", "proposed_unit_price": 130, "estimated_unit_cost": 80, "delivery_months": 6}
    corridor = {"predicted_low_price": 100, "predicted_high_price": 140, "p10": 100, "p90": 130}

    result = validate_scenario(scenario, tender, corridor)

    assert not result["is_valid"]
    assert any("Sızıntı tespit edildi" in item for item in result["hard_constraint_violations"])
