"""Model adapter layer for Sutra transformer models.

Responsibilities
----------------
- Expose three model factories: get_summarizer, get_ner, get_translator.
- Load each model lazily (only when first requested) and cache at the
  process level using lru_cache so Streamlit reruns and evaluation loops
  reuse the same in-memory objects.
- Detect CUDA automatically; fall back to CPU silently when unavailable.
- Suppress only the specific third-party warnings that are harmless and
  unavoidable with the pinned model versions (documented inline).
- Raise ModelUnavailableError for any load failure so callers can fall
  back gracefully without a crash.

Caching notes
-------------
lru_cache is process-level and survives Streamlit reruns (Streamlit keeps
the server process alive between requests). Adding st.cache_resource on
top would create a redundant second caching layer and is intentionally
avoided here.

NLLB-200 caching note
---------------------
facebook/nllb-200-distilled-600M is a single multilingual model. Older
code created one pipeline() instance per (src, tgt) direction, which
re-loaded the same 2.4 GB weights for each language pair. The model is
now loaded once and language codes are forwarded per-call via the
pipeline's generate kwargs, saving memory and startup time.
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

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


def _get_device() -> int:
    """Return 0 for the first CUDA GPU or -1 for CPU.

    Never raises; CUDA absence is treated as a normal CPU-only environment.
    """
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            logger.info("CUDA available — transformer models will run on GPU 0.")
            return 0
    except ImportError:
        pass
    logger.debug("No CUDA detected — transformer models will run on CPU.")
    return -1


def _load_pipeline(task: str, model_name: str, **kwargs: Any) -> Any:
    """Load a HuggingFace pipeline, raising ModelUnavailableError on any failure."""
    try:
        from transformers import AutoTokenizer, pipeline  # noqa: PLC0415
    except ImportError as exc:
        raise ModelUnavailableError(
            "Transformer support is not installed. Install the 'models' optional dependencies."
        ) from exc

    device = _get_device()

    # Silence the T5Tokenizer "legacy" warning emitted by transformers>=4.38
    # when a SentencePiece-backed tokenizer is loaded with use_fast=True.
    # Passing use_fast=False through tokenizer_kwargs is not supported by
    # recent pipeline versions; it is otherwise forwarded as a model kwarg.
    # Construct the tokenizer explicitly so the option reaches the tokenizer.
    if "mT5" in model_name or "mt5" in model_name.lower():
        kwargs["tokenizer"] = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Narrow filter for the SentencePiece byte-fallback log line that
    # sentencepiece emits at the C-extension level for Devanagari tokens.
    # We filter only this message and only within this call; we do not
    # suppress all UserWarnings globally.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*byte fallback.*",
            category=UserWarning,
        )
        try:
            return pipeline(task, model=model_name, device=device, **kwargs)
        except Exception as exc:
            detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            raise ModelUnavailableError(
                f"The model '{model_name}' could not be loaded. "
                f"Check your network, Hugging Face access, local model cache, or available memory. Details: {detail}"
            ) from exc


# ---------------------------------------------------------------------------
# Public model factories — each loaded lazily and cached for the lifetime
# of the Python process.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_summarizer() -> Any:
    """Return the cached mT5 summarisation pipeline."""
    logger.info("Loading summarisation model '%s' (first use).", SUMMARY_MODEL)
    return _load_pipeline("text2text-generation", SUMMARY_MODEL)


@lru_cache(maxsize=1)
def get_ner() -> Any:
    """Return the cached IndicNER pipeline."""
    logger.info("Loading NER model '%s' (first use).", NER_MODEL)
    return _load_pipeline(
        "token-classification",
        NER_MODEL,
        aggregation_strategy="simple",
    )


@lru_cache(maxsize=1)
def get_translator() -> Any:
    """Return the cached NLLB-200 translation pipeline.

    The pipeline is loaded once for all language directions.  Language codes
    are supplied at inference time via the forced_bos_token_id override in
    translation.py, avoiding the need to create one pipeline per (src, tgt)
    pair and reloading the 2.4 GB weights multiple times.
    """
    logger.info("Loading translation model '%s' (first use).", TRANSLATION_MODEL)
    return _load_pipeline("translation_xx_to_yy", TRANSLATION_MODEL)


def warm_up_models(include_translation: bool = False) -> list[str]:
    """Load the selected transformer pipelines and return their model names."""
    loaded = [get_summarizer(), get_ner()]
    names = [SUMMARY_MODEL, NER_MODEL]
    if include_translation:
        loaded.append(get_translator())
        names.append(TRANSLATION_MODEL)
    return names
