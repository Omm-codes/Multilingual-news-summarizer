from dataclasses import dataclass, field
from typing import Literal

Language = Literal["English", "Hindi", "Marathi"]
EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "DATE"]


@dataclass(frozen=True)
class Entity:
    text: str
    label: EntityType


@dataclass
class AnalysisResult:
    language: str
    language_code: str
    summary: str
    entities: list[Entity] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = "Pasted text"
    mode: str = "Lightweight extractive demo"
    processing_time: float = 0.0          # seconds, end-to-end pipeline
    input_sentence_count: int = 0         # sentences in the cleaned input
    keywords: list[str] = field(default_factory=list)  # top content keywords
