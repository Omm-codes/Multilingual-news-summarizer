import re
from dataclasses import dataclass
from collections import Counter

from .model_adapters import ModelUnavailableError, get_summarizer
from .text_processing import split_sentences

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "more", "most", "not", "of",
    "on", "or", "our", "said", "she", "so", "some", "s", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "to", "too", "us",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
    "और", "का", "की", "के", "को", "में", "ने", "से", "है", "हैं", "था", "थे", "यह",
    "आणि", "चा", "ची", "चे", "ला", "मध्ये", "ने", "हे", "आहे", "आहेत", "होते",
}

_SUMMARY_SIGNALS = {
    "feelings", "emotion", "emotions", "understand the claim", "what does it mean",
    "causation", "backstory", "perspective", "statistical significance", "significant",
    "margin of error", "imprecision", "be curious", "curiosity", "what's missing",
    "what is missing", "check the", "ask whether", "six commandments",
}


@dataclass(frozen=True)
class SummaryLength:
    baseline_sentences: int
    target_ratio: float
    min_new_tokens: int
    max_new_tokens: int
    translation_max_length: int


SUMMARY_LENGTHS = {
    "Short": SummaryLength(2, 0.085, 32, 384, 512),
    "Medium": SummaryLength(3, 0.105, 64, 768, 768),
    "Long": SummaryLength(7, 0.255, 96, 1280, 1280),
}


def resolve_summary_length(value: str | int = "Medium") -> SummaryLength:
    if isinstance(value, str):
        return SUMMARY_LENGTHS.get(value, SUMMARY_LENGTHS["Medium"])
    sentence_count = max(1, min(value, 5))
    return SummaryLength(
        baseline_sentences=sentence_count,
        target_ratio=0.0,
        min_new_tokens=16,
        max_new_tokens=max(48, sentence_count * 32),
        translation_max_length=max(64, sentence_count * 48),
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w\u0900-\u097F]+", text, flags=re.UNICODE))


def _target_word_count(text: str, length: SummaryLength) -> int:
    source_words = _word_count(text)
    minimum = 40 if length.target_ratio >= 0.30 else 30 if length.target_ratio >= 0.20 else 20
    return max(minimum, round(source_words * length.target_ratio))


def _generation_budget(text: str, length: SummaryLength) -> int:
    if not length.target_ratio:
        return length.max_new_tokens
    target_tokens = round(_target_word_count(text, length) * 1.3)
    return min(length.max_new_tokens, max(length.min_new_tokens, target_tokens))


def _tokens(sentence: str) -> list[str]:
    return [
        token for token in re.findall(r"[\w\u0900-\u097F]+", sentence.lower(), flags=re.UNICODE)
        if token not in _STOPWORDS
    ]


def summarize_extractive(
    text: str, sentence_count: int = 3, target_words: int | None = None
) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    sentence_count = max(1, min(sentence_count, len(sentences)))
    frequencies = Counter(token for sentence in sentences for token in _tokens(sentence))
    if not frequencies:
        return " ".join(sentences[:sentence_count])
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = _tokens(sentence)
        score = sum(frequencies[token] for token in tokens) / max(len(tokens), 1)
        score += 1.5 * sum(signal in sentence.casefold() for signal in _SUMMARY_SIGNALS)
        score += max(0, len(sentences) - index) / len(sentences)
        scored.append((score, index, sentence))

    selected: list[tuple[float, int, str]] = []
    selected_tokens: set[str] = set()
    remaining = scored[:]
    selected_words = 0
    while remaining and (
        selected_words < target_words
        if target_words is not None
        else len(selected) < sentence_count
    ):
        best = max(
            remaining,
            key=lambda item: item[0] - 0.35 * len(set(_tokens(item[2])).intersection(selected_tokens)),
        )
        remaining.remove(best)
        selected.append(best)
        selected_tokens.update(_tokens(best[2]))
        selected_words += _word_count(best[2])
    if selected and all(item[1] != 0 for item in selected):
        selected[-1] = scored[0]
    return " ".join(sentence for _, _, sentence in sorted(selected, key=lambda item: item[1]))


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Return the top_n most frequent content tokens as keyword strings.

    Uses the same stopword list and tokeniser as the extractive summariser so
    no new dependencies are introduced.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    frequencies = Counter(token for sentence in sentences for token in _tokens(sentence))
    # Return original-case form by scanning first occurrence per token
    seen: dict[str, str] = {}
    for sentence in sentences:
        for raw in re.findall(r"[\w\u0900-\u097F]+", sentence, flags=re.UNICODE):
            key = raw.lower()
            if key in frequencies and key not in seen and key not in _STOPWORDS:
                seen[key] = raw
    ranked = sorted(
        seen.keys(),
        key=lambda k: (frequencies[k], len(k) > 2, len(k)),
        reverse=True,
    )
    return [seen[k] for k in ranked[:top_n]]


def summarize_transformer(text: str, sentence_count: str | int = "Medium") -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    length = resolve_summary_length(sentence_count)
    target_tokens = _generation_budget(text, length)
    total_words = max(1, _word_count(text))

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        if current and current_length + len(sentence) > 1_800:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += len(sentence)
    if current:
        chunks.append(" ".join(current))

    summarizer = get_summarizer()
    summaries = []
    for chunk in chunks:
        chunk_budget = target_tokens
        if length.target_ratio:
            chunk_budget = round(target_tokens * _word_count(chunk) / total_words * 1.25)
            chunk_budget = min(length.max_new_tokens, max(length.min_new_tokens, chunk_budget))
        min_tokens = min(max(16, chunk_budget // 4), chunk_budget - 1)
        generated = summarizer(
            f"summarize: {chunk}",
            max_new_tokens=chunk_budget,
            min_new_tokens=min_tokens,
            do_sample=False,
        )
        if generated and generated[0].get("generated_text"):
            summaries.append(generated[0]["generated_text"].strip())

    combined = " ".join(summaries)
    if len(summaries) > 1 and combined:
        min_tokens = min(max(16, target_tokens // 4), target_tokens - 1)
        final = summarizer(
            f"summarize: {combined}",
            max_new_tokens=target_tokens,
            min_new_tokens=min_tokens,
            do_sample=False,
        )
        if final and final[0].get("generated_text"):
            combined = final[0]["generated_text"].strip()
    return combined


def summarize(text: str, sentence_count: str | int = 3, method: str = "extractive") -> str:
    return summarize_with_status(text, sentence_count, method)[0]


def summarize_with_status(
    text: str, sentence_count: str | int = 3, method: str = "extractive"
) -> tuple[str, str]:
    if method == "transformer":
        try:
            return summarize_transformer(text, sentence_count), "Transformer abstractive"
        except ModelUnavailableError:
            length = resolve_summary_length(sentence_count)
            target_words = _target_word_count(text, length) if length.target_ratio else None
            return summarize_extractive(
                text, length.baseline_sentences, target_words
            ), "Extractive baseline (transformer unavailable)"
    if isinstance(sentence_count, str):
        length = resolve_summary_length(sentence_count)
        return summarize_extractive(
            text,
            length.baseline_sentences,
            _target_word_count(text, length),
        ), "Extractive word-frequency baseline"
    baseline_sentences = (
        sentence_count
    )
    return summarize_extractive(text, baseline_sentences), "Extractive word-frequency baseline"
