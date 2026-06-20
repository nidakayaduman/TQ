"""Compatibility wrapper for positive/unsupported claim detection."""

from __future__ import annotations

from .forbidden_claim_detector import detect_forbidden_claims


def detect_positive_claims(text: str) -> dict[str, object]:
    return detect_forbidden_claims(text)
