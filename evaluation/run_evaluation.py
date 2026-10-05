"""Evaluate Sutra baseline and transformer models on the curated dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from news_summarizer.entities import extract_entities_regex, extract_entities_transformer
from news_summarizer.evaluation import ner_scores, rouge_scores
from news_summarizer.model_adapters import ModelUnavailableError
from news_summarizer.summarization import summarize_extractive, summarize_transformer

DATASET_PATH = Path(__file__).with_name("dataset.json")
LANGUAGES = ("en", "hi", "mr")
SUMMARY_SENTENCE_COUNT = 3


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)

    if not isinstance(records, list) or len(records) != 15:
        raise ValueError("The evaluation dataset must contain exactly 15 records.")
    return records


def _mean_scores(scores: Iterable[dict[str, float]]) -> dict[str, float]:
    score_list = list(scores)
    if not score_list:
        return {}
    keys = score_list[0]
    return {key: sum(score[key] for score in score_list) / len(score_list) for key in keys}


def _group_results(
    records: list[dict[str, Any]], scorer: Callable[[dict[str, Any]], dict[str, float]]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[record["language"]].append(scorer(record))

    results = {language: _mean_scores(grouped[language]) for language in LANGUAGES}
    results["overall"] = _mean_scores(scorer(record) for record in records)
    return results


def _summary_scores(
    records: list[dict[str, Any]], summarizer: Callable[[str, int], str]
) -> dict[str, dict[str, float]]:
    return _group_results(
        records,
        lambda record: rouge_scores(
            summarizer(record["text"], SUMMARY_SENTENCE_COUNT), record["reference_summary"]
        ),
    )


def _entity_pairs(entities: Iterable[Any]) -> list[tuple[str, str]]:
    return [(entity.text, entity.label) for entity in entities]


def _ner_scores(
    records: list[dict[str, Any]], extractor: Callable[[str], list[Any]]
) -> dict[str, dict[str, float]]:
    return _group_results(
        records,
        lambda record: ner_scores(
            _entity_pairs(extractor(record["text"])),
            [(entity["text"], entity["type"]) for entity in record["entities"]],
        ),
    )


def _completed(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {"status": "completed", "metrics": metrics}


def _unavailable(error: ModelUnavailableError) -> dict[str, Any]:
    return {"status": "unavailable", "error": str(error), "metrics": None}


def evaluate_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    records = load_dataset(path)
    baseline_summary = _summary_scores(records, summarize_extractive)
    regex_ner = _ner_scores(records, extract_entities_regex)

    results: dict[str, dict[str, Any]] = {}
    for language in (*LANGUAGES, "overall"):
        results[language] = {
            "article_count": sum(record["language"] == language for record in records)
            if language != "overall"
            else len(records),
            "summarization": {"baseline": _completed(baseline_summary[language])},
            "ner": {"regex": _completed(regex_ner[language])},
        }

    try:
        transformer_summary = _completed(_summary_scores(records, summarize_transformer))
    except ModelUnavailableError as error:
        transformer_summary = _unavailable(error)

    try:
        indic_ner = _completed(_ner_scores(records, extract_entities_transformer))
    except ModelUnavailableError as error:
        indic_ner = _unavailable(error)

    for language in (*LANGUAGES, "overall"):
        results[language]["summarization"]["transformer"] = {
            "status": transformer_summary["status"],
            "error": transformer_summary.get("error"),
            "metrics": (
                transformer_summary["metrics"][language]
                if transformer_summary["metrics"] is not None
                else None
            ),
        }
        results[language]["ner"]["indic_ner"] = {
            "status": indic_ner["status"],
            "error": indic_ner.get("error"),
            "metrics": indic_ner["metrics"][language] if indic_ner["metrics"] is not None else None,
        }

    return {"article_count": len(records), "results": results}


def _format_metrics(metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return "unavailable"
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())


def format_report(report: dict[str, Any]) -> str:
    lines = [f"Sutra evaluation ({report['article_count']} articles)"]
    for language in (*LANGUAGES, "overall"):
        section = report["results"][language]
        title = "Overall" if language == "overall" else language
        lines.append(f"\n{title} ({section['article_count']} articles)")
        lines.append(
            "  Summarization baseline: "
            + _format_metrics(section["summarization"]["baseline"]["metrics"])
        )
        transformer = section["summarization"]["transformer"]
        lines.append(
            "  Summarization transformer: "
            + _format_metrics(transformer["metrics"])
            + f" [{transformer['status']}]"
        )
        lines.append("  NER regex: " + _format_metrics(section["ner"]["regex"]["metrics"]))
        indic_ner = section["ner"]["indic_ner"]
        lines.append(
            "  NER Indic-NER: "
            + _format_metrics(indic_ner["metrics"])
            + f" [{indic_ner['status']}]"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        report = evaluate_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, indent=2) if args.as_json else format_report(report))
    statuses = [
        report["results"]["overall"]["summarization"]["transformer"]["status"],
        report["results"]["overall"]["ner"]["indic_ner"]["status"],
    ]
    return 0 if all(status == "completed" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
