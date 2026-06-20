from pathlib import Path

from src.advisor.forbidden_claim_detector import detect_forbidden_claims


def test_forbidden_claim_detector_catches_unsafe_claims():
    unsafe = "Bu senaryo " + ("guaranteed" + "_win") + " gibi sunulamaz."
    result = detect_forbidden_claims(unsafe)
    assert result["forbidden_claims_detected"]


def test_forbidden_claim_detector_catches_probability_claims():
    result = detect_forbidden_claims("Bu ihalenin kazanma olasılığı %70.")
    assert result["forbidden_claims_detected"]


def test_app_does_not_use_deprecated_score_label():
    app_text = Path("app.py").read_text(encoding="utf-8")
    assert "p(" + "win)" not in app_text
    assert "Emsal " + "p(" + "win)" not in app_text
