import pytest

from news_summarizer.entities import extract_entities
from news_summarizer.ingestion import extract_url_text
from news_summarizer.pipeline import analyze_text
from news_summarizer.summarization import extract_keywords, summarize
from news_summarizer.text_processing import detect_language, normalize_text, split_sentences
from news_summarizer import model_adapters, translation


ENGLISH = "The Maharashtra Government announced a new farming program in Mumbai on 12/03/2026. The Ministry said the program will support farmers. Officials expect the plan to reach more districts this year."
HINDI = "महाराष्ट्र सरकार ने किसानों के लिए नई योजना की घोषणा की। मंत्रालय ने कहा कि यह योजना कृषि क्षेत्र को सहायता देगी।"
MARATHI = "महाराष्ट्र सरकारने शेतकऱ्यांसाठी नवीन योजनेची घोषणा केली. मंत्रालयाने सांगितले की या योजनेमुळे कृषी क्षेत्राला मदत मिळेल."


def test_normalize_text_removes_markup_and_whitespace():
    assert normalize_text(" <p>Hello</p>   world ") == "Hello world"


def test_split_sentences_supports_devanagari_punctuation():
    assert split_sentences("पहिले वाक्य। दुसरे वाक्य!") == ["पहिले वाक्य।", "दुसरे वाक्य!"]


@pytest.mark.parametrize("text, language", [(ENGLISH, "English"), (HINDI, "Hindi"), (MARATHI, "Marathi")])
def test_detect_language_supports_project_languages(text, language):
    assert detect_language(text)[0] == language


def test_short_text_is_rejected():
    with pytest.raises(ValueError, match="20 characters"):
        analyze_text("Too short")


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="enter an article"):
        analyze_text("   ")


def test_oversized_text_is_rejected_before_language_detection():
    with pytest.raises(ValueError, match="100,000 characters"):
        analyze_text("a" * 100_001)


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="Only English, Hindi, and Marathi"):
        analyze_text("これは日本語の記事です。ニュースを読みます。")


def test_summary_respects_sentence_count():
    assert len(summarize(ENGLISH, sentence_count=2).split(". ")) <= 2


def test_long_summary_covers_distinct_guidance_topics():
    text = (
        "This article explains how to evaluate statistical claims. "
        "First, notice your feelings before accepting evidence. "
        "Second, understand the claim and define what the numbers mean. "
        "Check the backstory to learn where the statistic came from. "
        "Put the number in perspective by comparing it with familiar quantities. "
        "Beware statistical significance when a result has little practical importance. "
        "Embrace imprecision and be curious about what is missing."
    )

    summary = summarize(text, sentence_count="Long")

    assert len(summary.split()) >= 20
    guidance_topics = ("feelings", "backstory", "perspective", "significance", "curious")
    assert sum(topic in summary for topic in guidance_topics) >= 3


def test_baseline_entities_prioritize_locations_and_organizations():
    text = (
        "Michael Pollan discussed Star Wars in New York. "
        "The National Rifle Association and The Guardian published reports. "
        "Researchers at Stony Brook University studied the issue."
    )

    entities = extract_entities(text)
    by_label = {(entity.text, entity.label) for entity in entities}

    assert ("Michael Pollan", "PERSON") in by_label
    assert ("New York", "LOCATION") in by_label
    assert ("Stony Brook", "LOCATION") in by_label
    assert ("National Rifle Association", "ORGANIZATION") in by_label
    assert ("The Guardian", "ORGANIZATION") in by_label
    assert ("Star Wars", "PERSON") not in by_label
    assert ("New York", "PERSON") not in by_label


def test_keywords_filter_function_words_and_keep_topic_terms():
    text = (
        "We have more claims, but statistics require evidence. "
        "The analysis examines bias, emotions, and perspective in public data."
    )

    keywords = {keyword.casefold() for keyword in extract_keywords(text, top_n=10)}

    assert not keywords.intersection({"we", "have", "more", "but", "the", "and"})
    assert {"claims", "statistics", "evidence", "bias"}.issubset(keywords)


def test_entity_extraction_returns_core_categories():
    entities = extract_entities(ENGLISH)
    labels = {entity.label for entity in entities}
    assert {"ORGANIZATION", "LOCATION", "DATE"}.issubset(labels)


def test_pipeline_returns_structured_result():
    result = analyze_text(ENGLISH, sentence_count=2)
    assert result.language == "English"
    assert result.summary
    assert result.source == "Pasted text"


def test_translation_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr(
        translation,
        "get_translator",
        lambda: (_ for _ in ()).throw(model_adapters.ModelUnavailableError("unavailable")),
    )

    summary, note = translation.translate_summary("A short brief.", "Hindi", "English")
    assert summary == "A short brief."
    assert "unavailable" in note



@pytest.mark.parametrize("url", ["", "example.com/article", "ftp://example.com/article"])
def test_url_input_requires_http_or_https(url):
    with pytest.raises(ValueError, match="article URL"):
        extract_url_text(url)


def test_url_input_rejects_pathologically_long_urls():
    with pytest.raises(ValueError, match="shorter article URL"):
        extract_url_text("https://example.com/" + "a" * 2_050)
