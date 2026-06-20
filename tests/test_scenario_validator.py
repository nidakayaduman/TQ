from src.optimizer.scenario_validator import validate_scenario


def test_scenario_validator_blocks_low_margin():
    tender = {
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


def test_scenario_validator_returns_explainable_output():
    tender = {
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
