"""Detect unsupported claims in advisor text."""

from __future__ import annotations

import re

from ..constants import FORBIDDEN_CLAIM_TERMS


def detect_forbidden_claims(text: str) -> dict[str, object]:
    lowered = text.casefold()
    detected: list[str] = []
    for term in FORBIDDEN_CLAIM_TERMS:
        if term.casefold() in lowered:
            detected.append(term)

    certainty_patterns = [
        r"\bwill definitely win\b",
        r"\bguaranteed award\b",
        r"\bkesin\s+kazan",
        r"\bkesin\s+ihale\s+al",
    ]
    for pattern in certainty_patterns:
        if re.search(pattern, lowered):
            detected.append(pattern)

    return {
        "forbidden_claims_detected": bool(detected),
        "detected_terms": sorted(set(detected)),
    }

