"""Sutra — Multilingual Smart News & Text Summarizer with Entity Extraction."""

import json
import logging
import math

import streamlit as st

from news_summarizer.ingestion import extract_url_text
from news_summarizer.model_adapters import ModelUnavailableError, warm_up_models
from news_summarizer.pipeline import analyze_text
from news_summarizer.translation import translate_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sutra | News Intelligence",
    page_icon=":material/article:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Sample article
# ---------------------------------------------------------------------------
SAMPLE_TEXT = (
    "The Maharashtra Government announced a new farming program in Mumbai on 12/03/2026. "
    "The Ministry said the program will support farmers across India. "
    "Officials expect the plan to reach more districts this year. "
    "Minister Priya Sharma stated that a budget of ₹500 crore has been allocated for the initiative. "
    "The Ministry of Agriculture will oversee the rollout beginning April 2026."
)

ENTITY_ICONS = {
    "PERSON": ":material/person:",
    "ORGANIZATION": ":material/business:",
    "LOCATION": ":material/location_on:",
    "DATE": ":material/calendar_today:",
}

ENTITY_COLORS = {
    "PERSON": "blue",
    "ORGANIZATION": "orange",
    "LOCATION": "green",
    "DATE": "violet",
}


def load_sample() -> None:
    st.session_state["article_input"] = SAMPLE_TEXT


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "article_input" not in st.session_state:
    st.session_state["article_input"] = ""

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Sutra", icon=":material/article:")
st.subheader("Multilingual Smart News & Text Summarizer with Entity Extraction")
st.caption(
    "Detect language · Summarize multilingual text · Extract named entities · Translate on demand  \n"
    "Supports **English**, **Hindi**, and **Marathi**. Text is processed locally and never stored."
)

# ---------------------------------------------------------------------------
# How it works — collapsible, useful for viva demonstration
# ---------------------------------------------------------------------------
with st.expander("How it works", icon=":material/help_outline:"):
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
**NLP Pipeline**

1. **Language detection** — `langdetect` identifies English, Hindi, or Marathi using character n-gram statistics with a fixed seed for reproducibility.
2. **Text normalisation** — strips HTML markup and collapses whitespace; sentence boundaries are split on `.`, `!`, `?`, and `।` (Devanagari danda).
3. **Extractive summarisation** *(Baseline)* — word-frequency scoring ranks sentences; position bias keeps lead sentences; no model download required.
4. **Abstractive summarisation** *(Transformer)* — uses `csebuetnlp/mT5_multilingual_XLSum`, a multilingual T5 model fine-tuned on XL-Sum. Downloaded lazily from Hugging Face.
5. **Entity extraction** *(Baseline)* — regex patterns capture dates, title-cased person names, known organisation keywords, and a curated location list for EN/HI/MR.
6. **Entity extraction** *(Transformer)* — uses `ai4bharat/IndicNER`, a BERT-based model for Indic named-entity recognition, with span reconstruction for subword tokens.
7. **Translation** *(optional)* — uses `facebook/nllb-200-distilled-600M` (NLLB-200), a 600M-parameter multilingual translation model.
            """
        )
    with col_right:
        st.markdown(
            """
**Models & Evaluation**

| Component | Baseline | Transformer |
|---|---|---|
| Summarisation | TF-IDF extractive | mT5 / XL-Sum |
| NER | Regex patterns | IndicNER (BERT) |
| Translation | — (fallback) | NLLB-200 |

**Evaluation metrics**
- Summarisation: ROUGE-1, ROUGE-2, ROUGE-L (F1)
- NER: Precision, Recall, F1 (exact match)

Run `python evaluation/run_evaluation.py` to reproduce scores on the curated 15-article multilingual benchmark (5 articles per language).

**Dataset** — 15 manually written news-style articles (EN/HI/MR) with hand-annotated reference summaries and entity labels. No web-scraped content.
            """
        )

st.write("")

# ---------------------------------------------------------------------------
# Section 1 — Input
# ---------------------------------------------------------------------------
st.header("1. Input article", anchor=False)
st.caption("Choose a source, configure the NLP options, and process one article at a time.")

input_mode = st.segmented_control(
    "Input source",
    ["Paste article", "Article URL"],
    default="Paste article",
    key="input_mode",
)

with st.container(border=True):
    if input_mode == "Paste article":
        st.subheader("Article text", anchor=False)
        st.caption("Paste an English, Hindi, or Marathi news article. Analysis runs entirely on-device.")
    else:
        st.subheader("Article URL", anchor=False)
        st.caption("Enter a public article URL. Publisher paywalls and heavy JavaScript can affect extraction quality.")

    with st.form("analysis_form", border=False):
        if input_mode == "Paste article":
            article_text = st.text_area(
                "Article text",
                height=220,
                placeholder="Paste an English, Hindi, or Marathi news article here…",
                key="article_input",
                label_visibility="collapsed",
            )
        else:
            article_url = st.text_input(
                "Article URL",
                placeholder="https://example.com/news/article",
                key="article_url",
                label_visibility="collapsed",
            )
            article_text = ""

        control_col, language_col, model_col, action_col = st.columns(
            [1, 1, 1.25, 1], vertical_alignment="bottom"
        )
        with control_col:
            summary_length = st.selectbox(
                "Summary length",
                ["Short", "Medium", "Long"],
                index=1,
                help="Target source coverage: Short 8.5%, Medium 10.5%, Long 25.5%. Actual output varies by article structure.",
            )
        with language_col:
            output_language = st.selectbox(
                "Output language",
                ["Same as article", "English", "Hindi", "Marathi"],
            )
        with model_col:
            model_mode = st.selectbox(
                "NLP engine",
                ["Local baseline", "Transformer (downloads models)"],
                help=(
                    "**Baseline** — extractive summarisation + regex NER. Fast, offline, no GPU needed.\n\n"
                    "**Transformer** — mT5 (abstractive summary) + IndicNER + NLLB-200 (translation). "
                    "Models are downloaded from Hugging Face on first use (~2–4 GB total)."
                ),
            )
        with action_col:
            analyze = st.form_submit_button(
                "Analyse article",
                type="primary",
                icon=":material/auto_awesome:",
                use_container_width=True,
            )

    warm_models = False
    if model_mode.startswith("Transformer"):
        warm_models = st.button(
            "Download transformer models",
            icon=":material/download:",
            help="Load the selected transformer models before analysing an article. The first download can be several gigabytes.",
        )

    if input_mode == "Paste article":
        st.button(
            "Load sample article",
            icon=":material/experiment:",
            on_click=load_sample,
            use_container_width=False,
        )

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if warm_models:
    try:
        include_translation = output_language != "Same as article"
        with st.spinner("Downloading and loading the selected transformer models…"):
            loaded_models = warm_up_models(include_translation=include_translation)
        st.success(
            f"Transformer models ready ({len(loaded_models)} loaded). You can analyse the article now.",
            icon=":material/check_circle:",
        )
    except ModelUnavailableError as error:
        st.error(str(error), icon=":material/error:")

if analyze:
    try:
        model_method = "transformer" if model_mode.startswith("Transformer") else "baseline"

        # Build an informative spinner message based on cache state.
        if model_method == "transformer":
            from news_summarizer import model_adapters as _ma  # noqa: PLC0415
            _summ_cached = _ma.get_summarizer.cache_info().currsize > 0
            _ner_cached = _ma.get_ner.cache_info().currsize > 0
            if not _summ_cached or not _ner_cached:
                _analysis_msg = (
                    "Loading transformer models for the first time — "
                    "this may take a moment while weights are read from disk…"
                )
            else:
                _analysis_msg = "Running transformer inference (using cached models)…"
        else:
            _analysis_msg = "Analysing article with local baseline…"

        source = "Pasted text"
        with st.spinner(_analysis_msg):
            if input_mode == "Article URL":
                article_text, source = extract_url_text(article_url)
            result = analyze_text(
                article_text,
                summary_length,
                source,
                summary_method=model_method,
                entity_method=model_method,
            )

        source_summary = result.summary
        requested_language = (
            result.language if output_language == "Same as article" else output_language
        )

        # Translation is a separate step so it can show its own spinner.
        if requested_language != result.language and model_method == "transformer":
            from news_summarizer import model_adapters as _ma2  # noqa: PLC0415
            _trans_cached = _ma2.get_translator.cache_info().currsize > 0
            _trans_msg = (
                "Running translation (using cached NLLB-200 model)…"
                if _trans_cached
                else "Loading translation model for the first time — this may take a moment…"
            )
            with st.spinner(_trans_msg):
                translated, translation_note = translate_summary(
                    source_summary,
                    requested_language,
                    result.language,
                    summary_length=summary_length,
                )
        else:
            translated, translation_note = translate_summary(
                source_summary,
                requested_language,
                result.language,
                summary_length=summary_length,
            )

        result.summary = translated
        if "unavailable" in translation_note or "not configured" in translation_note:
            result.warnings.append(translation_note)
        st.session_state["result"] = result
        st.session_state["source_text"] = article_text
        st.session_state["source_summary"] = source_summary
        st.session_state["translation_language"] = requested_language
        st.session_state["translation_note"] = translation_note

    except ValueError as error:
        st.error(str(error), icon=":material/error:")
    except Exception:
        logger.exception("Article processing failed")
        st.error(
            "Something went wrong while processing this article. "
            "Try pasting the text directly if you used a URL.",
            icon=":material/error:",
        )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.get("result")

if not result:
    # Empty state — help the user understand what to do
    st.write("")
    with st.container(border=True):
        st.markdown(
            "#### No analysis yet\n\n"
            "Paste a news article (or paste a URL) above, choose your options, and click **Analyse article**.\n\n"
            "Not sure what to paste? Click **Load sample article** to see a demo."
        )
else:
    source_text = st.session_state.get("source_text", "")
    source_summary = st.session_state.get("source_summary", result.summary)
    translation_language = st.session_state.get("translation_language", result.language)
    translation_note = st.session_state.get("translation_note", "Source language")
    summary_method, entity_method = result.mode.split(" + ", maxsplit=1)

    st.write("")

    # -----------------------------------------------------------------------
    # Section 2 — Analysis overview
    # -----------------------------------------------------------------------
    st.header("2. Analysis overview", anchor=False)

    # --- Key metrics ---
    input_words = len(source_text.split())
    summary_words = len(source_summary.split())
    compression_pct: int | str
    if input_words > 0 and summary_words > 0 and summary_words < input_words:
        compression_pct = f"{math.floor((1 - summary_words / input_words) * 100)}%"
    else:
        compression_pct = "—"

    metric_cols = st.columns(5)
    metrics = [
        (result.language, "Detected language", None),
        (input_words, "Input words", None),
        (summary_words, "Summary words", None),
        (compression_pct, "Text reduction", None),
        (len(result.entities), "Entities found", None),
    ]
    for col, (value, label, _) in zip(metric_cols, metrics):
        with col:
            st.metric(label, value)

    st.write("")

    # --- Pipeline status panel ---
    st.subheader("Pipeline status", anchor=False)
    pipe_cols = st.columns(4)

    with pipe_cols[0]:
        st.markdown("**Language detection**")
        st.success(f"✓ {result.language} detected", icon=":material/language:")

    with pipe_cols[1]:
        st.markdown("**Summarisation**")
        if "unavailable" in summary_method:
            st.warning(f"Fallback: {summary_method}", icon=":material/warning:")
        elif "Transformer" in summary_method:
            st.success(f"✓ {summary_method}", icon=":material/smart_toy:")
        else:
            st.info(f"✓ {summary_method}", icon=":material/auto_awesome_mosaic:")

    with pipe_cols[2]:
        st.markdown("**Named-entity recognition**")
        if "unavailable" in entity_method:
            st.warning(f"Fallback: {entity_method}", icon=":material/warning:")
        elif "Transformer" in entity_method:
            st.success(f"✓ {entity_method}", icon=":material/smart_toy:")
        else:
            st.info(f"✓ {entity_method}", icon=":material/manage_search:")

    with pipe_cols[3]:
        st.markdown("**Translation**")
        if translation_language != result.language:
            if "unavailable" in translation_note or "not configured" in translation_note:
                st.warning(f"Fallback — {translation_note}", icon=":material/warning:")
            else:
                st.success(f"✓ → {translation_language} ({translation_note})", icon=":material/translate:")
        else:
            st.info("Not requested", icon=":material/translate:")

    # Reading stats footer row
    stats_cols = st.columns(3)
    with stats_cols[0]:
        st.caption(f"**Input sentences:** {result.input_sentence_count}")
    with stats_cols[1]:
        st.caption(f"**Processing time:** {result.processing_time} s")
    with stats_cols[2]:
        st.caption(f"**Source:** {result.source}")

    # --- Top keywords ---
    if result.keywords:
        st.write("")
        st.markdown("**Top keywords extracted from article**")
        kw_cols = st.columns(len(result.keywords))
        for col, kw in zip(kw_cols, result.keywords):
            with col:
                st.badge(kw, color="blue")

    st.write("")

    # -----------------------------------------------------------------------
    # Section 3 — Generated summary
    # -----------------------------------------------------------------------
    st.header("3. Generated summary", anchor=False)

    tab_summary, tab_compare = st.tabs(["Summary", "Original vs. summary"])

    with tab_summary:
        with st.container(border=True):
            st.subheader(
                f"{'Abstractive' if 'Transformer' in summary_method else 'Extractive'} brief · {result.language}",
                anchor=False,
            )
            st.write(source_summary)
            st.caption(f"Method: {summary_method} · Source: {result.source}")

        if translation_language != result.language:
            with st.container(border=True):
                st.subheader(f"Translated brief · {translation_language}", anchor=False)
                st.write(result.summary)
                st.caption(translation_note)

        # Download actions
        with st.container():
            dl_col1, dl_col2, _ = st.columns([1, 1, 3])
            download_summary = source_summary
            if translation_language != result.language:
                download_summary += f"\n\nTranslated ({translation_language}):\n{result.summary}"
            with dl_col1:
                st.download_button(
                    "Download brief (.txt)",
                    (
                        f"Sutra Analysis\n"
                        f"Language: {result.language}\n"
                        f"Method: {summary_method}\n"
                        f"Source: {result.source}\n\n"
                        f"SUMMARY\n{download_summary}\n\n"
                        f"KEYWORDS\n{', '.join(result.keywords)}\n\n"
                        f"ENTITIES\n"
                        + "\n".join(f"[{e.label}] {e.text}" for e in result.entities)
                    ),
                    file_name="sutra-brief.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    use_container_width=True,
                )
            with dl_col2:
                download_data = {
                    "language": result.language,
                    "summary": source_summary,
                    "translated_summary": result.summary if translation_language != result.language else None,
                    "translation_language": translation_language if translation_language != result.language else None,
                    "source": result.source,
                    "summary_method": summary_method,
                    "entity_method": entity_method,
                    "input_words": input_words,
                    "summary_words": summary_words,
                    "processing_time_s": result.processing_time,
                    "keywords": result.keywords,
                    "entities": [{"text": e.text, "label": e.label} for e in result.entities],
                }
                st.download_button(
                    "Export JSON",
                    json.dumps(download_data, ensure_ascii=False, indent=2),
                    file_name="sutra-analysis.json",
                    mime="application/json",
                    icon=":material/data_object:",
                    use_container_width=True,
                )

    with tab_compare:
        left_col, right_col = st.columns(2)
        with left_col:
            with st.container(border=True):
                st.subheader(f"Original article · {result.language}", anchor=False)
                st.caption(f"{input_words} words · {result.input_sentence_count} sentences")
                st.write(source_text)
        with right_col:
            with st.container(border=True):
                st.subheader("Generated summary", anchor=False)
                st.caption(
                    f"{summary_words} words"
                    + (f" · {compression_pct} reduction" if compression_pct != "—" else "")
                    + f" · {summary_method}"
                )
                st.write(source_summary)
                if translation_language != result.language:
                    st.divider()
                    st.caption(f"Translated → {translation_language}")
                    st.write(result.summary)

    st.write("")

    # -----------------------------------------------------------------------
    # Section 4 — Named entities
    # -----------------------------------------------------------------------
    st.header("4. Named entities", anchor=False)
    st.caption(
        f"Entities detected by the **{entity_method}** method, grouped by category.  \n"
        "Duplicate surface forms are deduplicated. Unknown labels are omitted."
    )

    grouped: dict[str, list[str]] = {label: [] for label in ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]}
    for entity in result.entities:
        if entity.label in grouped:
            grouped[entity.label].append(entity.text)

    entity_cols = st.columns(4)
    for col, (label, values) in zip(entity_cols, grouped.items()):
        with col:
            icon = ENTITY_ICONS[label]
            color = ENTITY_COLORS[label]
            with st.container(border=True):
                st.subheader(label.title(), anchor=False)
                st.caption(f"{icon} {len(values)} found")
                if values:
                    for value in values:
                        st.badge(value, color=color)
                else:
                    st.caption("None detected")

    st.write("")

    # -----------------------------------------------------------------------
    # Expanders — source article & warnings
    # -----------------------------------------------------------------------
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning, icon=":material/info:")

    with st.expander("View source article text", icon=":material/description:"):
        st.caption(f"Source: {result.source} · {input_words} words · {result.input_sentence_count} sentences")
        st.write(source_text)

    st.write("")

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    if st.button("Start over", icon=":material/refresh:", type="secondary"):
        for key in ("result", "source_text", "source_summary", "translation_language",
                    "translation_note", "article_input"):
            st.session_state.pop(key, None)
        st.rerun()
