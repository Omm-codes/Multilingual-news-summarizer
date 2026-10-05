from .entities import extract_entities_with_status
from .models import AnalysisResult
from .summarization import summarize_with_status
from .text_processing import detect_language, normalize_text

MAX_ARTICLE_CHARACTERS = 100_000


def analyze_text(
    text: str,
    sentence_count: int = 3,
    source: str = "Pasted text",
    summary_method: str = "baseline",
    entity_method: str = "baseline",
) -> AnalysisResult:
    cleaned = normalize_text(text)
    if len(cleaned) > MAX_ARTICLE_CHARACTERS:
        raise ValueError("Please limit the article to 100,000 characters.")
    language, code = detect_language(cleaned)
    summary, summary_mode = summarize_with_status(cleaned, sentence_count, summary_method)
    entities, entity_mode = extract_entities_with_status(cleaned, entity_method)
    warnings = []
    if "unavailable" in summary_mode or "unavailable" in entity_mode:
        warnings.append("Optional transformer models were unavailable, so the local baseline was used.")
    return AnalysisResult(
        language=language,
        language_code=code,
        summary=summary,
        entities=entities,
        warnings=warnings,
        source=source,
        mode=f"{summary_mode} + {entity_mode}",
    )
