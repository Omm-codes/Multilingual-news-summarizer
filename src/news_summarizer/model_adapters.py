from functools import lru_cache
from typing import Any

SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
NER_MODEL = "ai4bharat/IndicNER"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

LANGUAGE_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
}


class ModelUnavailableError(RuntimeError):
    """Raised when optional transformer dependencies or model weights are unavailable."""


def _load_pipeline(task: str, model_name: str, **kwargs: Any) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ModelUnavailableError(
            "Transformer support is not installed. Install the 'models' optional dependencies."
        ) from exc

    try:
        return pipeline(task, model=model_name, **kwargs)
    except Exception as exc:
        raise ModelUnavailableError(
            f"The model '{model_name}' could not be loaded. Check your network or local model cache."
        ) from exc


@lru_cache(maxsize=1)
def get_summarizer() -> Any:
    return _load_pipeline("text2text-generation", SUMMARY_MODEL)


@lru_cache(maxsize=1)
def get_ner() -> Any:
    return _load_pipeline("token-classification", NER_MODEL, aggregation_strategy="simple")


@lru_cache(maxsize=None)
def get_translator(source_code: str, target_code: str) -> Any:
    return _load_pipeline(
        "translation",
        TRANSLATION_MODEL,
        src_lang=source_code,
        tgt_lang=target_code,
    )
