#!/usr/bin/env python3
"""
exog_sentiment_pipeline.py
=========================

This script builds a daily sentiment index for Natural Gas and Oil from
your news archive produced by ``news_pipeline.py``.  It supports two
sentiment backends:

  1. A finance‑domain BERT model (``ProsusAI/finbert``) via the
     ``transformers`` library.  If available, this model will be used
     automatically.
  2. A lightweight lexicon fallback that requires no external
     dependencies.  When ``transformers`` is not installed or the
     model fails to load, the script falls back to counting positive
     and negative keywords.

The output is a CSV file ``sentiment_exog.csv`` with columns::

    date,sentiment_ng,sentiment_ol

that can be merged with your other exogenous data sources.  Dates are
aligned to a business‑day calendar and missing values are forward
filled for a few days to smooth gaps.

Example usage::

    python exog_sentiment_pipeline.py \
        --news-root data/news/docs \
        --output data/sentiment_exog.csv \
        --days 90

The ``--days`` flag limits the news considered to the specified
number of trailing days.  Omit it to process the entire archive.

Dependencies
------------

* ``pandas`` and ``numpy`` are required.
* ``transformers`` and ``torch`` (optional) enable the FinBERT model.

If you wish to override the default model, set the environment
variable ``SENTIMENT_MODEL`` to a Hugging Face model name.  The
script will attempt to download and cache it automatically.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def find_parquet_files(root: Path) -> List[Path]:
    """Recursively find all Parquet files under the given root directory."""
    return [p for p in root.rglob("*.parquet")]


def _infer_commodity(row: pd.Series) -> Optional[str]:
    """Infer commodity tag ("ng" or "oil") from text if missing."""
    text = ((row.get("title") or "") + " " + (row.get("text") or "")).lower()
    # Keywords for natural gas
    ng_keywords = [
        "natural gas", "natgas", "henry hub", "lng", "ng ", "nat gas",
        "ammonia", "feedgas", "dry gas", "pipeline capacity",
    ]
    # Keywords for oil
    oil_keywords = [
        "crude", "wti", "brent", "oil market", "refinery", "opec", "oil prices",
        "refined product", "diesel", "gasoline", "barrel", "petroleum",
    ]
    for kw in ng_keywords:
        if any(kw in text for kw in ["natural gas","natgas","henry hub","lng","lng export","pipeline gas","gas storage","hh","ng "]):
            return "ng"
    for kw in oil_keywords:
        if any(kw in text for kw in ["crude","wti","brent","oil market","refinery","opec","spr release","upstream oil","downstream oil","diesel","gasoline"]):
            return "oil"
    return None


def load_news(root: Path, days: Optional[int] = None) -> pd.DataFrame:
    """Load news Parquet files and return a DataFrame with required columns.

    Parameters
    ----------
    root : pathlib.Path
        Directory under which to search for Parquet files.
    days : int, optional
        Number of trailing days of news to load.  If None, all news
        will be returned.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing columns ``published_utc`` (datetime),
        ``title``, ``text``, ``commodity_tag`` and ``blob``.  Rows
        with missing publication dates or empty text are dropped.
    """
    files = find_parquet_files(root)
    if not files:
        logging.warning("No Parquet files found under %s", root)
        return pd.DataFrame(columns=["published_utc", "title", "text", "commodity_tag", "blob"])
    frames = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            logging.warning("Failed to read %s: %s", fp, e)
            continue
        # Determine available columns
        cols = df.columns
        # Choose whichever columns are present
        title_col = "title" if "title" in cols else None
        text_col = "text" if "text" in cols else None
        date_col = "published_utc" if "published_utc" in cols else None
        tag_col = "commodity_tag" if "commodity_tag" in cols else None
        required = [c for c in [date_col, title_col, text_col, tag_col] if c is not None]
        if not required:
            continue
        sub = df[required].copy()
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["published_utc", "title", "text", "commodity_tag", "blob"])
    news = pd.concat(frames, ignore_index=True)
    # Parse datetime
    if "published_utc" in news.columns:
        news["published_utc"] = pd.to_datetime(news["published_utc"], errors="coerce", utc=True)
        news = news.dropna(subset=["published_utc"])
    else:
        # Without a timestamp, we cannot sort or filter by days
        news["published_utc"] = pd.Timestamp.utcnow()
    # Filter to last N days
    if days is not None and not news.empty:
        cutoff = news["published_utc"].max() - pd.Timedelta(days=days)
        news = news[news["published_utc"] >= cutoff]
    # Fill missing columns
    if "title" not in news.columns:
        news["title"] = ""
    else:
        news["title"] = news["title"].fillna("")
    if "text" not in news.columns:
        news["text"] = ""
    else:
        news["text"] = news["text"].fillna("")
    if "commodity_tag" not in news.columns:
        news["commodity_tag"] = None
    news["commodity_tag"] = news["commodity_tag"].fillna(news.apply(_infer_commodity, axis=1))
    # Build blob for sentiment analysis (truncate long text)
    news["blob"] = (news["title"].astype(str) + "\n" + news["text"].astype(str)).str.slice(0, 2000)
    # Extract date for daily aggregation
    news["date"] = news["published_utc"].dt.tz_convert("UTC").dt.date
    return news[["date", "commodity_tag", "blob"]]


# Optional heavy dependency for FinBERT
_TRANSFORMERS_OK = False
try:
    from transformers import (  # type: ignore
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TextClassificationPipeline,
    )
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False


def load_finbert() -> Optional['TextClassificationPipeline']:
    """Attempt to load the FinBERT sentiment model.

    Returns
    -------
    transformers.TextClassificationPipeline or None
        A Hugging Face pipeline if successful, otherwise None.
    """
    if not _TRANSFORMERS_OK:
        return None
    model_id = os.environ.get("SENTIMENT_MODEL", "ProsusAI/finbert")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True, max_length=512)
        return pipe
    except Exception as e:
        logging.warning("Failed to load FinBERT model %s: %s", model_id, e)
        return None


def finbert_score(pipe: 'TextClassificationPipeline', texts: List[str]) -> np.ndarray:
    """Compute sentiment scores using a FinBERT pipeline.

    Parameters
    ----------
    pipe : transformers.TextClassificationPipeline
        Hugging Face pipeline for sentiment classification.
    texts : list of str
        Text strings to score.

    Returns
    -------
    numpy.ndarray
        Sentiment scores between -1 and 1 (positive minus negative).
    """
    out_scores = []
    for result in pipe(texts):
        # result is a list of dicts: {"label": label, "score": score}
        scores = {r["label"].lower(): float(r["score"]) for r in result}
        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        out_scores.append(pos - neg)
    return np.array(out_scores, dtype=float)


# Lexicon fallback
_POSITIVE = set(
    "beat upbeat optimistic surge bullish rally increase growth expand expansion rose rising strengthened improved strong exceeded positive momentum".split()
)
_NEGATIVE = set(
    "miss downgrade pessimistic slump bearish drop decrease fall falling weakened deteriorated weak disappointed negative slowdown".split()
)


def lexicon_score(texts: List[str]) -> np.ndarray:
    """Compute sentiment scores using a simple keyword lexicon.

    Scores are normalized by the length of the document to prevent
    extremely short texts from producing extreme values.

    Parameters
    ----------
    texts : list of str
        Text strings to score.

    Returns
    -------
    numpy.ndarray
        Sentiment scores between -1 and 1.
    """
    scores = []
    for t in texts:
        tokens = [w.strip(".,;:!?()[]{}\"'").lower() for w in str(t).split()]
        if not tokens:
            scores.append(0.0)
            continue
        pos = sum(1 for w in tokens if w in _POSITIVE)
        neg = sum(1 for w in tokens if w in _NEGATIVE)
        length = max(len(tokens), 20)
        scores.append((pos - neg) / length)
    return np.array(scores, dtype=float)


def compute_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["sentiment_ng", "sentiment_oil", "sentiment_power"], index=pd.Index([], name="date"))
    
    texts = news["blob"].tolist()
    pipe = load_finbert()
    if pipe is not None:
        logging.info("Using FinBERT for sentiment scoring")
        scores = finbert_score(pipe, texts)
    else:
        logging.info("Using lexicon fallback for sentiment scoring")
        scores = lexicon_score(texts)
    
    news = news.copy()
    news["score"] = scores
    
    # Handle multi-tags: Split and explode
    news['commodity_tag'] = news['commodity_tag'].astype(str).str.split(',')
    news = news.explode('commodity_tag')
    news['commodity_tag'] = news['commodity_tag'].str.strip()
    
    # Aggregate by date and commodity_tag
    agg = news.groupby(["date", "commodity_tag"], dropna=False)["score"].mean().unstack("commodity_tag")
    
    # Ensure all columns exist (rename 'ol' to 'oil' if typo)
    for col in ["ng", "oil", "power"]:
        if col not in agg.columns:
            agg[col] = np.nan
    
    df = agg.rename(columns={"ng": "sentiment_ng", "oil": "sentiment_oil", "power": "sentiment_power", "ol": "sentiment_oil"})  # Handle potential typo
    
    # Reindex to business days
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.asfreq("B")
    
    # Forward fill a limited window
    df = df.ffill(limit=3)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily sentiment exogenous features from a news archive.")
    parser.add_argument(
        "--news-root",
        type=str,
        required=True,
        help="Folder containing Parquet docs from news_pipeline.py (e.g., data/news/docs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sentiment_exog.csv",
        help="Path to write sentiment CSV (default: data/sentiment_exog.csv)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Limit to the most recent N days of news (default: unlimited)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    news_root = Path(args.news_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Scanning news files under %s ...", news_root)
    news_df = load_news(news_root, days=args.days)
    logging.info("Loaded %d news items", len(news_df))
    logging.info("Computing sentiment ...")
    sentiment_df = compute_sentiment(news_df)
    # Prepare for CSV output
    sentiment_df = sentiment_df.reset_index().rename(columns={"index": "date"})
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    sentiment_df.to_csv(out_path, index=False)
    logging.info("Sentiment exogenous saved to %s (%d rows)", out_path, len(sentiment_df))


if __name__ == "__main__":
    main()