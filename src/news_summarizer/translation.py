"""Translation wrapper for Sutra.

Uses the single cached NLLB-200 pipeline from model_adapters. Language
codes are forwarded per-call as forced_bos_token_id so the model does not
need to be reloaded for each language direction.
"""

from __future__ import annotations

from .model_adapters import LANGUAGE_CODES, ModelUnavailableError, get_translator
from .summarization import resolve_summary_length


def translate_summary(
    summary: str,
    target_language: str,
    source_language: str,
    method: str = "transformer",
    summary_length: str | int = "Medium",
) -> tuple[str, str]:
    """Return (translated_text, status_note).

    Falls back to the original summary with an explanation note when
    translation is not requested, not configured, or the model is
    unavailable.
    """
    if target_language == source_language:
        return summary, "Source language"
    if method != "transformer":
        return summary, "Translation is not configured; showing the source-language summary."

    src_code = LANGUAGE_CODES.get(source_language)
    tgt_code = LANGUAGE_CODES.get(target_language)

    if not src_code or not tgt_code:
        return summary, "Translation model unavailable; showing the source-language summary."

    try:
        # The NLLB-200 pipeline is loaded once and shared across all
        # language directions (see model_adapters.get_translator docstring).
        translator = get_translator()
        translated = translator(
            summary,
            src_lang=src_code,
            tgt_lang=tgt_code,
            max_length=resolve_summary_length(summary_length).translation_max_length,
            do_sample=False,
        )
        if translated and translated[0].get("translation_text"):
            return translated[0]["translation_text"].strip(), "Translated with NLLB-200"
    except (KeyError, ModelUnavailableError):
        pass
    return summary, "Translation model unavailable; showing the source-language summary."
