"""Multilingual news summarizer package."""

from .models import AnalysisResult, Entity
from .pipeline import analyze_text

__all__ = ["AnalysisResult", "Entity", "analyze_text"]
