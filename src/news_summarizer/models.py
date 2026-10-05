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
