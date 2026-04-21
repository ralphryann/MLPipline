"""Ticket text normalization (train and inference must share this)."""

from __future__ import annotations

import re
from typing import Iterable

EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
EXTRA_SPACE_RE = re.compile(r"\s+")
ALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z\s.,!?]")


def clean_ticket_text(text: str) -> str:
    """Lowercase, mask PII-like tokens, strip noise; keep basic punctuation."""
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = HTML_RE.sub(" ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = URL_RE.sub(" <url> ", text)
    text = ALLOWED_CHARS_RE.sub(" ", text)
    text = EXTRA_SPACE_RE.sub(" ", text).strip()
    return text


def clean_ticket_text_batch(texts: Iterable[str]) -> list[str]:
    return [clean_ticket_text(t) for t in texts]
