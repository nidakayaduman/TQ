from src.advisor.fallback_advisor import build_fallback_advisor
from src.advisor.output_validator import validate_advisor_output


def test_fallback_advisor_validates():
    output = build_fallback_advisor(
        {
            "won_profile_fit_score": 70,
            "price_band_fit_score": 80,
            "margin_score": 60,
            "model_confidence_score": 75,
            "risk_flags": [],
            "similar_tender_count": 12,
            "cluster_name": "IV Solution profil",
        }
    )
    result = validate_advisor_output(output)
    assert result["valid"]
    assert result["advisor_validation_status"] == "pass"

