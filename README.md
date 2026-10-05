# Sutra — Multilingual Smart News & Text Summarizer with Entity Extraction

Sutra is a Streamlit NLP application that turns English, Hindi, and Marathi news articles into concise briefs and extracts named entities (people, organisations, locations, dates). It supports two interchangeable NLP engines — a lightweight local baseline and optional HuggingFace transformer models — selectable at runtime without restarting the app.

## Features

| Feature | Baseline mode | Transformer mode |
|---|---|---|
| Language detection | `langdetect` (EN / HI / MR) | Same |
| Summarisation | TF-IDF extractive | mT5 / XL-Sum (abstractive) |
| Named-entity recognition | Regex + curated terms | IndicNER (BERT-based) |
| Translation | Explicit fallback message | NLLB-200 distilled 600M |
| Keyword extraction | Word-frequency (top 8) | Same |

- **Pipeline overview** panel with per-stage status indicators
- **Transformer warm-up** button to download and load selected models before analysis
- **Summary length presets** — Short, Medium, and Long with shared generation limits
- **Reading statistics** — input/summary word counts, sentence count, compression %, processing time
- **Side-by-side comparison** tab (original article vs. generated summary)
- **Entity display** grouped by PERSON / ORGANIZATION / LOCATION / DATE with per-category counts
- **"How it works" expander** — concise technical explainer suitable for live demonstrations
- **Structured export** — `.txt` brief and `.json` analysis with all metadata
- Input safeguards for empty text, short text, oversized articles, invalid URLs, and unsupported languages

## Setup

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[test,models]"
```

Run the application:

```powershell
streamlit run app.py
```

Run the test suite:

```powershell
pytest
```

### Optional: transformer models

The transformer NLP engine downloads HuggingFace models on first use (cached afterwards). In the app, choose **Transformer** and click **Download transformer models** before analysing an article. The summarizer and NER models are always warmed up; NLLB is included when a different output language is selected.

| Model | Purpose | Approx. size |
|---|---|---|
| `csebuetnlp/mT5_multilingual_XLSum` | Abstractive summarisation | ~1.2 GB |
| `ai4bharat/IndicNER` | Named-entity recognition (Indic) | ~400 MB |
| `facebook/nllb-200-distilled-600M` | Translation (EN ↔ HI ↔ MR) | ~2.4 GB |

IndicNER may require accepting Hugging Face access conditions for the repository. Use the **Local baseline** mode when models or GPU are unavailable — the app degrades gracefully with an explicit status indicator.

Summary length uses source-relative targets rather than fixed output percentages or exact sentence counts. The Transformer caps below are practical safety limits; actual output depends on the article and model:

| Length | Baseline | Transformer / translation budget |
|---|---:|---:|
| Short | approximately 8.5% | 384 / 512 tokens |
| Medium | approximately 10.5% | 768 / 768 tokens |
| Long | approximately 25.5% | 1,280 / 1,280 tokens |

Models run on CUDA when available and otherwise run on CPU. CPU inference can be slow even after the download is complete. Model failures now include the first loading error detail so network, access, cache, and memory problems can be diagnosed without silently retrying the full workflow.

## Architecture

```
app.py                          ← Streamlit UI (presentation only)
src/news_summarizer/
    pipeline.py                 ← Orchestrates steps, tracks timing
    text_processing.py          ← Normalisation, sentence splitting, language detection
    summarization.py            ← Extractive baseline + transformer wrapper + keyword extraction
    entities.py                 ← Regex NER baseline + IndicNER transformer wrapper
    translation.py              ← NLLB-200 wrapper with explicit fallback
    model_adapters.py           ← Lazy-loaded, LRU-cached model pipelines
    ingestion.py                ← URL text extraction (trafilatura)
    evaluation.py               ← ROUGE-1/2/L and NER precision/recall/F1 helpers
    models.py                   ← AnalysisResult and Entity dataclasses
evaluation/
    dataset.json                ← 15 manually curated multilingual articles
    run_evaluation.py           ← Evaluation runner (CLI, --json flag available)
tests/                          ← pytest suite (31 tests, all passing)
```

`app.py` is purely presentational. All NLP logic lives in `news_summarizer` so models can be improved or swapped without touching the UI.

## Evaluation

Run the benchmark on the curated 15-article dataset (5 articles per language):

```powershell
python evaluation/run_evaluation.py
# or for machine-readable output:
python evaluation/run_evaluation.py --json
```

Reports ROUGE-1, ROUGE-2, and ROUGE-L (F1) for baseline and mT5 summarisation, plus precision, recall, and F1 for regex and IndicNER entity extraction — grouped by language and overall.

Transformer model failures are reported as `unavailable` with clear error messages; the runner never silently substitutes baseline results for unavailable transformer results.

## Project notes

- The regex entity extractor is intentionally labelled as a **lightweight demonstration** strategy. It uses language-aware patterns and a curated location/organisation vocabulary — not a fully trained multilingual NER model. For production-quality Indic NER, the transformer mode with IndicNER is recommended.
- URL extraction depends on publisher markup; pasted text is the most reliable demonstration path for live sessions.
- The evaluation dataset was manually written for this project; no real articles were scraped or used.
