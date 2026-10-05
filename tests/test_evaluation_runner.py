from news_summarizer.models import Entity

from evaluation import run_evaluation


def test_runner_reports_all_languages_and_metric_groups(monkeypatch):
    monkeypatch.setattr(run_evaluation, "summarize_transformer", lambda text, count: "model summary")
    monkeypatch.setattr(
        run_evaluation,
        "extract_entities_transformer",
        lambda text: [Entity("model entity", "PERSON")],
    )

    report = run_evaluation.evaluate_dataset()

    assert report["article_count"] == 15
    assert set(report["results"]) == {"en", "hi", "mr", "overall"}
    assert report["results"]["en"]["article_count"] == 5
    assert report["results"]["hi"]["article_count"] == 5
    assert report["results"]["mr"]["article_count"] == 5
    assert report["results"]["overall"]["article_count"] == 15

    for language in ("en", "hi", "mr", "overall"):
        section = report["results"][language]
        assert set(section["summarization"]) == {"baseline", "transformer"}
        assert set(section["ner"]) == {"regex", "indic_ner"}
        assert section["summarization"]["transformer"]["status"] == "completed"
        assert section["ner"]["indic_ner"]["status"] == "completed"
        assert set(section["summarization"]["baseline"]["metrics"]) == {
            "rouge1_f1",
            "rouge2_f1",
            "rougeL_f1",
        }
        assert set(section["ner"]["regex"]["metrics"]) == {"precision", "recall", "f1"}


def test_runner_does_not_replace_unavailable_transformers_with_baselines(monkeypatch):
    error = run_evaluation.ModelUnavailableError("model weights unavailable")
    monkeypatch.setattr(
        run_evaluation,
        "summarize_transformer",
        lambda text, count: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        run_evaluation,
        "extract_entities_transformer",
        lambda text: (_ for _ in ()).throw(error),
    )

    report = run_evaluation.evaluate_dataset()
    overall = report["results"]["overall"]

    assert overall["summarization"]["baseline"]["metrics"] is not None
    assert overall["ner"]["regex"]["metrics"] is not None
    assert overall["summarization"]["transformer"] == {
        "status": "unavailable",
        "error": "model weights unavailable",
        "metrics": None,
    }
    assert overall["ner"]["indic_ner"] == {
        "status": "unavailable",
        "error": "model weights unavailable",
        "metrics": None,
    }
    assert "unavailable" in run_evaluation.format_report(report)
