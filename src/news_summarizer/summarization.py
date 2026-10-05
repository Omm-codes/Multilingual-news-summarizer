import re
from collections import Counter

from .model_adapters import ModelUnavailableError, get_summarizer
from .text_processing import split_sentences

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
    "और", "का", "की", "के", "को", "में", "ने", "से", "है", "हैं", "था", "थे", "यह",
    "आणि", "चा", "ची", "चे", "ला", "मध्ये", "ने", "हे", "आहे", "आहेत", "होते",
}


def _tokens(sentence: str) -> list[str]:
    return [
        token for token in re.findall(r"[\w\u0900-\u097F]+", sentence.lower(), flags=re.UNICODE)
        if token not in _STOPWORDS
    ]


def summarize_extractive(text: str, sentence_count: int = 3) -> str:
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
        score += max(0, len(sentences) - index) / len(sentences)
        scored.append((score, index, sentence))
    selected = sorted(scored, reverse=True)[:sentence_count]
    return " ".join(sentence for _, _, sentence in sorted(selected, key=lambda item: item[1]))


def summarize_transformer(text: str, sentence_count: int = 3) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

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
        generated = summarizer(
            f"summarize: {chunk}",
            max_new_tokens=max(48, sentence_count * 32),
            min_new_tokens=16,
            do_sample=False,
        )
        if generated and generated[0].get("generated_text"):
            summaries.append(generated[0]["generated_text"].strip())

    combined = " ".join(summaries)
    if len(summaries) > 1 and combined:
        final = summarizer(
            f"summarize: {combined}",
            max_new_tokens=max(48, sentence_count * 32),
            min_new_tokens=16,
            do_sample=False,
        )
        if final and final[0].get("generated_text"):
            combined = final[0]["generated_text"].strip()
    return combined


def summarize(text: str, sentence_count: int = 3, method: str = "extractive") -> str:
    return summarize_with_status(text, sentence_count, method)[0]


def summarize_with_status(text: str, sentence_count: int = 3, method: str = "extractive") -> tuple[str, str]:
    if method == "transformer":
        try:
            return summarize_transformer(text, sentence_count), "Transformer abstractive"
        except ModelUnavailableError:
            return summarize_extractive(text, sentence_count), "Extractive baseline (transformer unavailable)"
    return summarize_extractive(text, sentence_count), "Extractive word-frequency baseline"
