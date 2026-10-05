# Sutra: Multilingual Smart News Summarizer

Sutra is a Streamlit NLP project that turns English, Hindi, and Marathi news articles into concise briefs and extracts people, organizations, locations, and dates.

## Features

- Pasted article text or optional URL extraction
- Language detection for English, Hindi, and Marathi
- Lightweight extractive summarization with adjustable length
- Heuristic entity extraction with four visible categories
- Optional output-language control with a transparent fallback when translation is not configured
- Responsive Streamlit dashboard with native controls, KPI metrics, and export actions
- Input safeguards for oversized article text, downloaded content, and URLs

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,models]"
```

Run the app:

```powershell
streamlit run app.py
```

Run tests:

```powershell
pytest
```

## Architecture

`app.py` handles presentation. The `news_summarizer` package keeps ingestion, text processing, summarization, entity extraction, and translation separate so stronger models can be added without rewriting the UI.

The default summarizer is extractive and local-friendly. Entity extraction is intentionally labeled as a lightweight demo strategy: it uses language-aware patterns and curated terms, not a fully trained multilingual NER model. For production-quality Hindi and Marathi NER, replace `entities.py` with a validated Indic or multilingual model adapter and test it against labeled examples.

URL extraction requires network access and depends on publisher markup. Pasted text is the most reliable demonstration path.

The current pipeline accepts articles up to 100,000 characters. The recommended transformer mode uses `csebuetnlp/mT5_multilingual_XLSum` for summarization, `ai4bharat/IndicNER` for named-entity recognition, and `facebook/nllb-200-distilled-600M` for translation. Models are downloaded lazily from Hugging Face on first use. The IndicNER repository may require accepting its Hugging Face access conditions. Use the lightweight baseline mode when model dependencies or weights are unavailable.

The baseline remains available for comparison: word-frequency extractive summarization, regex entity extraction, and explicit translation fallback. Evaluation helpers in `evaluation.py` calculate ROUGE-1/2/L and exact-match NER precision, recall, and F1 for labeled experiments.
