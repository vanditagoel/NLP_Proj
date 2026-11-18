"""Unicode-safe text cleaning and Indic script / language detection."""
from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_SYMBOL_RE = re.compile(r"#(\S+)")
MULTISPACE_RE = re.compile(r"\s+")

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def clean_text(text: str) -> str:
    """Clean social/media text while preserving Devanagari, Bengali, and English punctuation."""
    if not isinstance(text, str):
        return ""
    norm = unicodedata.normalize("NFC", text)
    norm = URL_RE.sub(" ", norm)
    norm = MENTION_RE.sub(" ", norm)
    norm = HASHTAG_SYMBOL_RE.sub(r"\1", norm)
    norm = MULTISPACE_RE.sub(" ", norm).strip()
    return norm


def detect_script_language(text: str) -> tuple[str, str]:
    """Detect Hindi (Devanagari), Bengali (Bengali script), or English reliably without external APIs."""
    if not isinstance(text, str) or not text.strip():
        return "English", "en"

    hi_count = len(DEVANAGARI_RE.findall(text))
    bn_count = len(BENGALI_RE.findall(text))
    en_count = len(LATIN_RE.findall(text))

    if hi_count >= 2 and hi_count >= bn_count:
        return "Hindi", "hi"
    if bn_count >= 2 and bn_count > hi_count:
        return "Bengali", "bn"
    if hi_count > 0 and hi_count >= bn_count and en_count == 0:
        return "Hindi", "hi"
    if bn_count > 0 and bn_count > hi_count and en_count == 0:
        return "Bengali", "bn"

    try:
        from langdetect import detect

        code = detect(text)
        if code == "hi":
            return "Hindi", "hi"
        if code == "bn":
            return "Bengali", "bn"
    except Exception:
        pass

    return "English", "en"
