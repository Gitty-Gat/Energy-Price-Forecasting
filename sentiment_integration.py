"""
Utilities for integrating sentiment analysis into the forecasting pipeline.

Sentiment signals derived from news and social media have been shown
to improve predictive performance in financial markets, including
energy stocks. A recent study using the FinBERT model – a
Transformer pre‑trained on financial text – demonstrated that
transformer‑based sentiment analysis of business news can enhance
stock market prediction in the energy industry, with news content
being more informative than headlines【955594989788747†L140-L146】.

This module provides a placeholder interface for computing
FinBERT‑based sentiment scores from raw text. For demonstration
purposes, a simple sentiment scoring approach using HuggingFace
transformers is provided. In practice you might cache results or
offload sentiment scoring to a separate service.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # type: ignore


def compute_finbert_sentiment(texts: Iterable[str]) -> pd.Series:
    """Compute FinBERT sentiment scores for a collection of texts.

    Parameters
    ----------
    texts : Iterable[str]
        An iterable of news headlines or articles.

    Returns
    -------
    pd.Series
        Series of sentiment scores (positive minus negative probability).

    Notes
    -----
    If the ``transformers`` library or the required model is not
    available, this function returns zeros for all input texts. The
    FinBERT model assigns probabilities to ``positive``, ``negative``
    and ``neutral`` classes; here we collapse the probabilities to a
    single score by subtracting negative from positive.
    """
    if pipeline is None:
        # transformers is not installed; return zero scores
        return pd.Series([0.0] * len(list(texts)))

    # Load FinBERT sentiment pipeline (may download model weights)
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    scores: List[float] = []
    for text in texts:
        result = sentiment_pipeline(text, truncation=True)[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)
    return pd.Series(scores)
