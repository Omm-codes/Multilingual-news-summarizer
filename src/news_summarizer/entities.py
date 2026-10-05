"""Named-entity extraction for Sutra.

Provides two strategies:
- extract_entities_regex  — fast, offline, no model required.
- extract_entities_transformer — uses ai4bharat/IndicNER (BERT-based).

BERT token limit
----------------
IndicNER (like all BERT-based models) has a hard 512-token limit per
input.  Long articles are chunked by sentence so each chunk stays within
that limit, and entities are deduplicated across chunks.  This preserves
full-article coverage without truncating silently.
"""

from __future__ import annotations

import re

from .model_adapters import ModelUnavailableError, get_ner
from .models import Entity
from .text_processing import split_sentences

# ---- Regex patterns used by the baseline extractor -------------------------

_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    r"|\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_DEVANAGARI_DATE_PATTERN = re.compile(
    r"(?<!\w)\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मे|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+\d{4}(?!\w)"
)

_KNOWN_LOCATIONS = (
    "United States", "New York", "Stony Brook", "United Kingdom", "South Africa",
    "India", "Maharashtra", "Gujarat", "Rajasthan", "Goa", "Delhi", "Mumbai",
    "Pune", "Jaipur", "Surat", "Kochi", "Nagpur", "Nashik", "Aurangabad",
    "भारत", "महाराष्ट्र", "गुजरात", "राजस्थान", "दिल्ली", "मुंबई", "पुणे",
)
_LOCATION_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(value) for value in _KNOWN_LOCATIONS) + r")(?!\w)",
    re.IGNORECASE,
)
_KNOWN_ORGANIZATIONS = (
    "National Rifle Association", "National Rifle", "National Health Service",
    "National Health", "UK Statistics Authority", "Statistics Authority",
    "Royal Statistical Society", "Royal Statistical", "Stony Brook University",
    "Yale University", "University of Chicago", "The Guardian", "Financial Times",
    "FiveThirtyEight", "Government", "Ministry", "University", "Company",
    "Corporation", "Organization", "Authority", "Society", "Service", "Association",
    "Institute", "College", "सरकार", "मंत्रालय", "विद्यापीठ", "कंपनी", "संस्था",
)
_ORGANIZATION_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(value) for value in _KNOWN_ORGANIZATIONS) + r")(?!\w)",
    re.IGNORECASE,
)
_KNOWN_PEOPLE = {
    "Michael Pollan", "Dan Kahan", "Charles Taber", "Milton Lodge", "Benjamin Franklin",
    "Richard Feynman", "Rebecca Goldin", "Mona Chalabi", "David Spiegelhalter",
    "Groucho Marx", "Donald Trump", "Bill O'Reilly", "Tim Harford",
}

# BERT-based models accept at most 512 sub-word tokens per sequence.
# We use a conservative character budget so chunking works safely across
# languages with variable token/character ratios (Devanagari uses more chars
# per token than Latin script).  ~350 chars ≈ ~120–180 BERT tokens for
# mixed Indic/English text, leaving plenty of headroom for [CLS]/[SEP].
_NER_CHUNK_CHARS = 350


def extract_entities_regex(text: str) -> list[Entity]:
    """Return entities extracted by rule-based regex patterns."""
    found: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    typed_spans: list[tuple[int, int, str]] = []

    def add(value: str, label: str) -> None:
        value = value.strip(" ,.;:()[]")
        key = (value.casefold(), label)
        if value and key not in seen:
            seen.add(key)
            found.append(Entity(value, label))

    for match in list(_DATE_PATTERN.finditer(text)) + list(_DEVANAGARI_DATE_PATTERN.finditer(text)):
        add(match.group(), "DATE")
        typed_spans.append((match.start(), match.end(), "DATE"))

    for match in _LOCATION_PATTERN.finditer(text):
        add(match.group(), "LOCATION")
        typed_spans.append((match.start(), match.end(), "LOCATION"))

    for match in _ORGANIZATION_PATTERN.finditer(text):
        add(match.group(), "ORGANIZATION")
        typed_spans.append((match.start(), match.end(), "ORGANIZATION"))

    person_patterns = [
        re.compile(
            r"\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Professor|Minister)\s+"
            r"([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2})"
        ),
        re.compile(r"\b(?:" + "|".join(re.escape(value) for value in _KNOWN_PEOPLE) + r")\b"),
    ]
    for pattern in person_patterns:
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group()
            start = match.start(1) if match.lastindex else match.start()
            end = match.end(1) if match.lastindex else match.end()
            if any(start < span_end and end > span_start for span_start, span_end, _ in typed_spans):
                continue
            add(value, "PERSON")

    return found


def _chunk_text_for_ner(text: str) -> list[str]:
    """Split text into sentence-aligned chunks safe for BERT's 512-token limit.

    Each chunk stays under _NER_CHUNK_CHARS characters.  Sentences are never
    split mid-sentence; overflow sentences start a new chunk.
    """
    sentences = split_sentences(text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current_parts and current_len + len(sentence) > _NER_CHUNK_CHARS:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(sentence)
        current_len += len(sentence)
    if current_parts:
        chunks.append(" ".join(current_parts))
    return chunks or [text]


def _extract_from_chunk(ner_pipeline: object, chunk: str) -> list[Entity]:
    """Run the NER pipeline on a single chunk and normalise labels."""
    label_map = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "DATE": "DATE",
    }
    found: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    for item in ner_pipeline(chunk):  # type: ignore[operator]
        label = label_map.get(item.get("entity_group", item.get("entity", "")))
        value = item.get("word", "").strip()
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            start = end = -1
        if 0 <= start < end <= len(chunk):
            value = chunk[start:end].strip()
        if not label or not value:
            continue
        key = (value.casefold(), label)
        if key not in seen:
            seen.add(key)
            found.append(Entity(value, label))
    return found


def extract_entities_transformer(text: str) -> list[Entity]:
    """Return entities extracted by IndicNER with safe chunking for long text."""
    ner = get_ner()
    chunks = _chunk_text_for_ner(text)

    all_found: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        for entity in _extract_from_chunk(ner, chunk):
            key = (entity.text.casefold(), entity.label)
            if key not in seen:
                seen.add(key)
                all_found.append(entity)

    return all_found


def extract_entities(text: str, method: str = "regex") -> list[Entity]:
    return extract_entities_with_status(text, method)[0]


def extract_entities_with_status(text: str, method: str = "regex") -> tuple[list[Entity], str]:
    if method == "transformer":
        try:
            return extract_entities_transformer(text), "Transformer NER"
        except ModelUnavailableError:
            return extract_entities_regex(text), "Regex baseline (transformer unavailable)"
    return extract_entities_regex(text), "Regex entity baseline"
