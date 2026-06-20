"""Prompt injection detection for advisor questions."""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"bypass\s+guardrails",
    r"reveal\s+actual\s+result",
    r"system\s+prompt",
    r"önceki\s+talimat",
    r"önceki\s+kuralları\s+unut",
    r"sistem\s+mesaj",
    r"gerçek\s+sonucu\s+söyle",
    r"gerçek\s+fiyat",
    r"rakipleri?\s+uydur",
    r"kesin\s+kazan",
    r"kesin\s+sonuç\s+söyle",
    r"garanti",
    r"garanti\s+de",
    r"p\s*[-(]?\s*win",
    r"win\s*[_ -]?\s*probability",
    r"kazanma\s+olasılığı\s+ver",
    r"kazanma\s+olasılığı\s*%?\s*\d*",
]

SAFE_REJECTION_MESSAGE = (
    "Bu talebi yerine getiremem. Gerçek sonuç reveal edilmeden paylaşılmaz ve sistem gerçek kazanma olasılığı "
    "üretmez. İstersen mevcut model çıktıları üzerinden risk, fiyat bandı ve profil uyumunu açıklayabilirim."
)


def detect_prompt_injection(text: str) -> dict[str, object]:
    lowered = text.casefold()
    matches = [pattern for pattern in INJECTION_PATTERNS if re.search(pattern, lowered)]
    return {
        "prompt_injection_detected": bool(matches),
        "matched_patterns": matches,
        "guardrail_status": "blocked" if matches else "pass",
    }


def safe_prompt_response() -> str:
    return SAFE_REJECTION_MESSAGE
