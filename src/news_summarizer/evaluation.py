from collections import Counter
import re
from collections.abc import Iterable


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097F]+", text.casefold(), flags=re.UNICODE)


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if not predicted or not reference or not overlap:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2 * precision * recall / (precision + recall)


def _rouge_n(prediction: list[str], reference: list[str], n: int) -> float:
    predicted_ngrams = Counter(tuple(prediction[index : index + n]) for index in range(len(prediction) - n + 1))
    reference_ngrams = Counter(tuple(reference[index : index + n]) for index in range(len(reference) - n + 1))
    overlap = sum((predicted_ngrams & reference_ngrams).values())
    return _f1(overlap, sum(predicted_ngrams.values()), sum(reference_ngrams.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    predicted_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    return {
        "rouge1_f1": _rouge_n(predicted_tokens, reference_tokens, 1),
        "rouge2_f1": _rouge_n(predicted_tokens, reference_tokens, 2),
        "rougeL_f1": _f1(lcs, len(predicted_tokens), len(reference_tokens)),
    }


def ner_scores(predicted: Iterable[tuple[str, str]], reference: Iterable[tuple[str, str]]) -> dict[str, float]:
    predicted_set = {(text.casefold().strip(), label) for text, label in predicted}
    reference_set = {(text.casefold().strip(), label) for text, label in reference}
    overlap = len(predicted_set & reference_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(reference_set) if reference_set else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(overlap, len(predicted_set), len(reference_set)),
    }
