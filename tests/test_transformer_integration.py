import pytest

from news_summarizer import entities, model_adapters, summarization, translation
from news_summarizer.models import Entity


@pytest.fixture(autouse=True)
def clear_model_caches():
    model_adapters.get_summarizer.cache_clear()
    model_adapters.get_ner.cache_clear()
    model_adapters.get_translator.cache_clear()
    yield
    model_adapters.get_summarizer.cache_clear()
    model_adapters.get_ner.cache_clear()
    model_adapters.get_translator.cache_clear()


def test_model_adapters_load_lazily_and_cache_each_transformer(monkeypatch):
    loaded = []

    def fake_load_pipeline(task, model_name, **kwargs):
        loaded.append((task, model_name, kwargs))
        return object()

    monkeypatch.setattr(model_adapters, "_load_pipeline", fake_load_pipeline)

    assert model_adapters.get_summarizer() is model_adapters.get_summarizer()
    assert model_adapters.get_ner() is model_adapters.get_ner()
    assert model_adapters.get_translator("eng_Latn", "hin_Deva") is model_adapters.get_translator(
        "eng_Latn", "hin_Deva"
    )
    assert model_adapters.get_translator("hin_Deva", "mar_Deva") is model_adapters.get_translator(
        "hin_Deva", "mar_Deva"
    )

    assert len(loaded) == 4
    assert model_adapters.NER_MODEL == "ai4bharat/IndicNER"
    assert loaded[0] == ("text2text-generation", model_adapters.SUMMARY_MODEL, {})
    assert loaded[1] == (
        "token-classification",
        model_adapters.NER_MODEL,
        {"aggregation_strategy": "simple"},
    )
    assert loaded[2] == (
        "translation",
        model_adapters.TRANSLATION_MODEL,
        {"src_lang": "eng_Latn", "tgt_lang": "hin_Deva"},
    )
    assert loaded[3] == (
        "translation",
        model_adapters.TRANSLATION_MODEL,
        {"src_lang": "hin_Deva", "tgt_lang": "mar_Deva"},
    )


def test_transformer_summary_calls_cached_summarizer(monkeypatch):
    calls = []

    class FakeSummarizer:
        def __call__(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return [{"generated_text": "A generated brief."}]

    monkeypatch.setattr(summarization, "get_summarizer", lambda: FakeSummarizer())

    summary, mode = summarization.summarize_with_status(
        "The first sentence explains the event. The second sentence adds context.",
        sentence_count=2,
        method="transformer",
    )

    assert summary == "A generated brief."
    assert mode == "Transformer abstractive"
    assert calls[0][0].startswith("summarize: ")
    assert calls[0][1]["do_sample"] is False


def test_transformer_ner_maps_supported_labels(monkeypatch):
    monkeypatch.setattr(
        entities,
        "get_ner",
        lambda: lambda text: [
            {"entity_group": "PER", "word": "Asha"},
            {"entity_group": "ORG", "word": "Ministry"},
            {"entity_group": "LOC", "word": "Pune"},
            {"entity_group": "DATE", "word": "12/03/2026"},
            {"entity_group": "MISC", "word": "ignored"},
        ],
    )

    found, mode = entities.extract_entities_with_status("article", method="transformer")

    assert found == [
        Entity("Asha", "PERSON"),
        Entity("Ministry", "ORGANIZATION"),
        Entity("Pune", "LOCATION"),
        Entity("12/03/2026", "DATE"),
    ]
    assert mode == "Transformer NER"


def test_transformer_ner_reconstructs_devanagari_spans_from_offsets(monkeypatch):
    text = "नेहा शाह ने मंत्रालय को पुणे में बताया।"
    person_start = text.index("नेहा शाह")
    location_start = text.index("पुणे")
    monkeypatch.setattr(
        entities,
        "get_ner",
        lambda: lambda value: [
            {
                "entity_group": "PER",
                "word": "न##हा शाह",
                "start": person_start,
                "end": person_start + len("नेहा शाह"),
            },
            {
                "entity_group": "LOC",
                "word": "##पुणे",
                "start": location_start,
                "end": location_start + len("पुणे"),
            },
        ],
    )

    found = entities.extract_entities_transformer(text)

    assert found == [Entity("नेहा शाह", "PERSON"), Entity("पुणे", "LOCATION")]


def test_regex_entity_extraction_remains_unchanged():
    text = "The Maharashtra Government announced a program in Pune on 12/03/2026."

    assert entities.extract_entities_regex(text) == entities.extract_entities(text)


@pytest.mark.parametrize(
    ("source", "target", "source_code", "target_code"),
    [
        ("English", "Hindi", "eng_Latn", "hin_Deva"),
        ("Hindi", "Marathi", "hin_Deva", "mar_Deva"),
        ("Marathi", "English", "mar_Deva", "eng_Latn"),
    ],
)
def test_translation_uses_nllb_language_codes(
    monkeypatch, source, target, source_code, target_code
):
    calls = []

    def fake_get_translator(actual_source, actual_target):
        calls.append((actual_source, actual_target))
        return lambda summary, **kwargs: [{"translation_text": "अनुवादित मजकूर"}]

    monkeypatch.setattr(translation, "get_translator", fake_get_translator)

    translated, note = translation.translate_summary("A brief.", target, source)

    assert translated == "अनुवादित मजकूर"
    assert note == "Translated with NLLB-200"
    assert calls == [(source_code, target_code)]


def test_transformer_load_failures_use_existing_fallbacks(monkeypatch):
    error = model_adapters.ModelUnavailableError("unavailable")
    monkeypatch.setattr(summarization, "get_summarizer", lambda: (_ for _ in ()).throw(error))
    monkeypatch.setattr(entities, "get_ner", lambda: (_ for _ in ()).throw(error))
    monkeypatch.setattr(translation, "get_translator", lambda source, target: (_ for _ in ()).throw(error))

    summary, summary_mode = summarization.summarize_with_status(
        "The ministry announced a program.", method="transformer"
    )
    found, entity_mode = entities.extract_entities_with_status(
        "The ministry announced a program in Pune.", method="transformer"
    )
    translated, translation_note = translation.translate_summary("A brief.", "Hindi", "English")

    assert summary
    assert "transformer unavailable" in summary_mode
    assert entity_mode == "Regex baseline (transformer unavailable)"
    assert found
    assert translated == "A brief."
    assert "unavailable" in translation_note