import re

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?।])\s+", normalized) if part.strip()]


def detect_language(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Please enter an article before analyzing it.")
    if len(normalized) < 20:
        raise ValueError("Please enter at least 20 characters of article text.")
    try:
        code = detect(normalized)
    except LangDetectException as exc:
        raise ValueError("The language could not be detected reliably.") from exc
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError("Only English, Hindi, and Marathi are supported in this version.")
    return SUPPORTED_LANGUAGES[code], code
