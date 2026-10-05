import re

from .model_adapters import ModelUnavailableError, get_ner
from .models import Entity

_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    r"|\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_DEVANAGARI_DATE_PATTERN = re.compile(r"(?<!\w)\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मे|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+\d{4}(?!\w)")


def extract_entities_regex(text: str) -> list[Entity]:
    found: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, label: str) -> None:
        value = value.strip(" ,.;:()[]")
        key = (value.casefold(), label)
        if value and key not in seen:
            seen.add(key)
            found.append(Entity(value, label))

    for match in list(_DATE_PATTERN.finditer(text)) + list(_DEVANAGARI_DATE_PATTERN.finditer(text)):
        add(match.group(), "DATE")

    for match in re.finditer(r"\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?", text):
        add(match.group(), "PERSON")
    for match in re.finditer(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text):
        add(match.group(), "PERSON")

    org_terms = r"(?:Government|Ministry|University|Company|Corporation|Organization|सरकार|मंत्रालय|विद्यापीठ|कंपनी|संस्था)"
    for match in re.finditer(rf"(?:[A-Z][\w.-]*\s+)?{org_terms}(?:\s+[A-Z][\w.-]*)?", text, re.IGNORECASE):
        add(match.group(), "ORGANIZATION")

    location_terms = r"(?:India|Maharashtra|Delhi|Mumbai|Pune|भारत|महाराष्ट्र|दिल्ली|मुंबई|पुणे)"
    for match in re.finditer(rf"(?<!\w){location_terms}(?!\w)", text, re.IGNORECASE):
        add(match.group(), "LOCATION")

    return found


def extract_entities_transformer(text: str) -> list[Entity]:
    ner = get_ner()
    found: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    label_map = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "DATE": "DATE",
    }
    for item in ner(text):
        label = label_map.get(item.get("entity_group", item.get("entity", "")))
        value = item.get("word", "").strip()
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            start = end = -1
        if 0 <= start < end <= len(text):
            value = text[start:end].strip()
        if not label or not value:
            continue
        key = (value.casefold(), label)
        if key not in seen:
            seen.add(key)
            found.append(Entity(value, label))
    return found


def extract_entities(text: str, method: str = "regex") -> list[Entity]:
    return extract_entities_with_status(text, method)[0]


def extract_entities_with_status(text: str, method: str = "regex") -> tuple[list[Entity], str]:
    if method == "transformer":
        try:
            return extract_entities_transformer(text), "Transformer NER"
        except ModelUnavailableError:
            return extract_entities_regex(text), "Regex baseline (transformer unavailable)"
    return extract_entities_regex(text), "Regex entity baseline"
