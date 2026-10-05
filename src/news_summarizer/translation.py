from .model_adapters import ModelUnavailableError, get_translator, LANGUAGE_CODES


def translate_summary(summary: str, target_language: str, source_language: str, method: str = "transformer") -> tuple[str, str]:
    if target_language == source_language:
        return summary, "Source language"
    if method != "transformer":
        return summary, "Translation is not configured; showing the source-language summary."

    try:
        translator = get_translator(LANGUAGE_CODES[source_language], LANGUAGE_CODES[target_language])
        translated = translator(summary, max_length=512, do_sample=False)
        if translated and translated[0].get("translation_text"):
            return translated[0]["translation_text"].strip(), "Translated with NLLB-200"
    except (KeyError, ModelUnavailableError):
        pass
    return summary, "Translation model unavailable; showing the source-language summary."
