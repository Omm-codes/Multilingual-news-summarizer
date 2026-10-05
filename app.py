import json
import logging

import streamlit as st

from news_summarizer.ingestion import extract_url_text
from news_summarizer.pipeline import analyze_text
from news_summarizer.translation import translate_summary

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Sutra | News intelligence",
    page_icon=":material/article:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SAMPLE_TEXT = (
    "The Maharashtra Government announced a new farming program in Mumbai on 12/03/2026. "
    "The Ministry said the program will support farmers across India. "
    "Officials expect the plan to reach more districts this year."
)


def load_sample() -> None:
    st.session_state["article_input"] = SAMPLE_TEXT


if "article_input" not in st.session_state:
    st.session_state["article_input"] = ""

st.title("Sutra", icon=":material/article:")
st.subheader("Multilingual Smart News & Text Summarizer with Entity Extraction")
st.write(
    "Detect language, summarize multilingual text, extract named entities, and optionally translate the generated brief."
)

st.space("small")
st.header("1. Input article", anchor=False)
st.caption("Choose a source, set the NLP options, and process one article at a time.")
input_mode = st.segmented_control(
    "Input source",
    ["Paste article", "Article URL"],
    default="Paste article",
    key="input_mode",
)

with st.container(border=True):
    if input_mode == "Paste article":
        st.subheader("Add an article", anchor=False)
        st.caption("Paste the article text below. Sutra keeps the analysis local and does not store your article.")
    else:
        st.subheader("Fetch an article", anchor=False)
        st.caption("Enter a public article URL. Publisher markup and paywalls can affect extraction.")

    with st.form("analysis_form", border=False):
        if input_mode == "Paste article":
            article_text = st.text_area(
                "Text input",
                height=220,
                placeholder="Paste an English, Hindi, or Marathi news article here...",
                key="article_input",
            )
        else:
            article_url = st.text_input(
                "URL input",
                placeholder="https://example.com/news/article",
                key="article_url",
            )
            article_text = ""

        control_col, language_col, model_col, action_col = st.columns([1, 1, 1.25, 1], vertical_alignment="bottom")
        with control_col:
            sentence_count = st.slider("Summary length", min_value=1, max_value=5, value=3)
        with language_col:
            output_language = st.selectbox(
                "Summary language",
                ["Same as article", "English", "Hindi", "Marathi"],
            )
        with model_col:
            model_mode = st.selectbox(
                "NLP engine",
                ["Local baseline", "Transformer (downloads models)"],
                help="Baseline mode is fast and offline. Transformer mode downloads multilingual models the first time it is used.",
            )
        with action_col:
            analyze = st.form_submit_button(
                "Process article",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

    if input_mode == "Paste article":
        st.button(
            "Load sample article",
            icon=":material/experiment:",
            on_click=load_sample,
            width="content",
        )

if analyze:
    try:
        with st.spinner("Reading and analyzing the article..."):
            source = "Pasted text"
            if input_mode == "Article URL":
                article_text, source = extract_url_text(article_url)
            model_method = "transformer" if model_mode.startswith("Transformer") else "baseline"
            result = analyze_text(
                article_text,
                sentence_count,
                source,
                summary_method=model_method,
                entity_method=model_method,
            )
            requested_language = result.language if output_language == "Same as article" else output_language
            source_summary = result.summary
            translated, translation_note = translate_summary(source_summary, requested_language, result.language)
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
        st.error("Something went wrong while processing this article. Try pasting the text directly.", icon=":material/error:")

result = st.session_state.get("result")
if result:
    source_text = st.session_state.get("source_text", "")
    source_summary = st.session_state.get("source_summary", result.summary)
    translation_language = st.session_state.get("translation_language", result.language)
    translation_note = st.session_state.get("translation_note", "Source language")
    summary_method, entity_method = result.mode.split(" + ", maxsplit=1)
    st.space("medium")
    st.header("2. Analysis overview", anchor=False)

    metric_cols = st.columns(4)
    metrics = [
        (result.language, "Detected language"),
        (len(source_summary.split()), "Summary words"),
        (len(result.entities), "Entities found"),
        (len(source_text.split()), "Source words"),
    ]
    for column, (value, label) in zip(metric_cols, metrics):
        with column:
            st.metric(label, value)

    method_cols = st.columns(2)
    with method_cols[0]:
        st.markdown(f"**Summarization method**  \n{summary_method}")
    with method_cols[1]:
        st.markdown(f"**NER method**  \n{entity_method}")

    status_cols = st.columns(2)
    with status_cols[0]:
        if "unavailable" in summary_method:
            st.warning("Transformer summarization was unavailable; the extractive baseline was used.")
        elif "Transformer" in summary_method:
            st.success("Transformer summarization completed successfully.")
        else:
            st.info("Local extractive summarization was used.")
    with status_cols[1]:
        if "unavailable" in entity_method:
            st.warning("Transformer NER was unavailable; the regex baseline was used.")
        elif "Transformer" in entity_method:
            st.success("Transformer named-entity extraction completed successfully.")
        else:
            st.info("Local regex entity extraction was used.")

    st.space("small")
    st.header("3. Generated summary", anchor=False)
    with st.container(border=True):
        st.subheader("Original-language brief", anchor=False)
        st.write(source_summary)
        st.caption(f"{result.language} · {summary_method} · Source: {result.source}")

        if translation_language != result.language:
            st.divider()
            st.subheader(f"Translated brief · {translation_language}", anchor=False)
            st.write(result.summary)
            st.caption(translation_note)

        with st.container(horizontal=True, horizontal_alignment="right"):
            download_summary = source_summary
            if translation_language != result.language:
                download_summary += f"\n\nTranslated ({translation_language}):\n{result.summary}"
            st.download_button(
                "Download brief",
                f"Language: {result.language}\n\n{download_summary}\n\nEntities: {', '.join(entity.text for entity in result.entities)}",
                file_name="sutra-brief.txt",
                mime="text/plain",
                icon=":material/download:",
                width="content",
            )
            download_data = {
                "language": result.language,
                "summary": result.summary,
                "source": result.source,
                "entities": [{"text": entity.text, "label": entity.label} for entity in result.entities],
            }
            st.download_button(
                "Export JSON",
                json.dumps(download_data, ensure_ascii=False, indent=2),
                file_name="sutra-analysis.json",
                mime="application/json",
                icon=":material/data_object:",
                width="content",
            )

    st.header("4. Named entities", anchor=False)
    st.caption("Entities returned by the selected NER method, grouped by category.")
    grouped = {label: [] for label in ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]}
    for entity in result.entities:
        grouped[entity.label].append(entity.text)
    entity_cols = st.columns(4)
    for column, (label, values) in zip(entity_cols, grouped.items()):
        with column:
            with st.container(border=True):
                st.subheader(label.title(), anchor=False)
                if values:
                    for value in values:
                        st.badge(value, icon=":material/sell:", color="blue")
                else:
                    st.caption("None detected")

    with st.expander("View source article", icon=":material/description:"):
        st.write(source_text)

    for warning in result.warnings:
        st.warning(warning, icon=":material/info:")

    st.space("small")
    if st.button("Start over", icon=":material/refresh:", type="secondary", width="content"):
        st.session_state.pop("result", None)
        st.session_state.pop("source_text", None)
        st.session_state.pop("source_summary", None)
        st.session_state.pop("translation_language", None)
        st.session_state.pop("translation_note", None)
        st.session_state.pop("article_input", None)
        st.rerun()
