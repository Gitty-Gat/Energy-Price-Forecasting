"""
news_pipeline.py
==================

This script implements an end‑to‑end workflow for ingesting publicly available
energy‑related news and regulatory filings, normalising them into a common
schema, persisting the raw and cleaned records to disk and a relational
catalogue, and laying the groundwork for downstream natural language
processing (NLP) tasks such as sentiment modelling.  The goal is to provide a
reproducible, zero‑cost data pipeline that can be scheduled on a daily or
hourly basis to keep your local corpus up to date.

The pipeline pulls data from three primary sources:

1. **EIA RSS feeds** – The U.S. Energy Information Administration publishes
   several RSS feeds including *Today in Energy*, *Press Releases* and
   *What's New*.  These feeds contain short articles and announcements.
   The documentation page for `feedparser` notes that it can handle
   numerous RSS and Atom versions and exposes a simple `parse` function
   that accepts a URL【605110887165993†L32-L45】.  For environments where
   `feedparser` is unavailable, a fallback XML parser using Python's
   standard library is provided.

2. **SEC EDGAR submissions API** – The Securities and Exchange Commission
   hosts unauthenticated JSON APIs on `data.sec.gov` that return a
   company's recent submissions in real time.  The EDGAR API page
   explains that the submissions API exposes each filer's filing history
   as a compact JSON file【780644219454658†L182-L207】.  This script
   demonstrates how to download a filer's recent forms, filter for
   Form 8‑K filings (commonly used for material events), and download
   their primary documents for extraction.

3. **GDELT DOC 2.0 API** – GDELT's DOC 2.0 API provides a keyword
   searchable index of global news coverage.  The API supports JSON
   output and can search across a rolling three month window of
   coverage【530671770496115†L17-L31】.  While calls to the API cannot
   be executed in this offline environment, the code shows how to
   construct a query and process the returned JSON structure.

The script organises data into three tiers:

* **Raw archive** – HTML pages and original metadata JSONs are stored in
  `data/news/raw/YYYY/MM/DD/` to preserve the original sources.
* **Normalised documents** – A unified Parquet file under
  `data/news/docs/YYYY/MM/docs_YYYYMMDD.parquet` contains one record per
  article with fields such as `doc_id`, `source`, `published_utc`, `title`
  and `text`.
* **Catalogue database** – A lightweight SQLite database
  (`news_catalog.db`) holds the same records for fast querying and
  indexing.  Indices on `published_utc`, `source` and `commodity_tag`
  accelerate common lookups.

To run this script you will need a Python environment with the
following packages installed: `requests`, `beautifulsoup4`, `pandas` and
`pyarrow` (for Parquet output).  If available, `feedparser` and
`trafilatura` provide higher quality feed parsing and text extraction.

The script is heavily commented to explain each step.  See the `main()`
function at the bottom for an example workflow.

Note: This code does not execute in the restricted container used by
this exercise because external network access is disabled.  It is
intended to be run in your own environment (e.g. a scheduled job on a
machine with internet access).
"""

import os
import re
import uuid
import json
import time
import hashlib
import logging
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple

try:
    import feedparser  # type: ignore
except ImportError:
    feedparser = None  # fallback to built‑in XML parser

try:
    import trafilatura  # type: ignore
except ImportError:
    trafilatura = None

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime, date
from email.utils import parsedate_to_datetime
import datetime as dt
from bs4.element import Tag as _BSTag
from urllib.parse import urljoin, urlparse
from pathlib import Path




# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


###############################################################################
# Utility functions
###############################################################################

def absolutize(url: str, base: str) -> str | None:
    """Return absolute URL built from url and base. Handles //host and /path cases."""
    if not url:
        return None
    url = url.strip()
    # protocol-relative (e.g., //www.eia.gov/pressroom/...)
    if url.startswith("//"):
        return "https:" + url
    # already absolute
    if urlparse(url).scheme:
        return url
    # relative → join with base
    return urljoin(base, url)

def json_sanitize(obj):
    """Recursively convert objects to JSON-serializable types."""
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # datetime/date -> ISO
    if isinstance(obj, (dt.datetime, dt.date)):
        if isinstance(obj, dt.datetime) and obj.tzinfo is None:
            obj = obj.replace(tzinfo=dt.timezone.utc)
        return obj.astimezone(dt.timezone.utc).isoformat()

    # struct_time -> ISO (UTC) via epoch
    if isinstance(obj, time.struct_time):
        return dt.datetime.fromtimestamp(time.mktime(obj), tz=dt.timezone.utc).isoformat()

    # bytes -> utf-8 (fallback to repr)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        try:
            return bytes(obj).decode("utf-8", "replace")
        except Exception:
            return repr(obj)

    # sets/tuples -> lists
    if isinstance(obj, (set, tuple, list)):
        return [json_sanitize(x) for x in obj]

    # bs4 tag -> text
    if _BSTag is not None and isinstance(obj, _BSTag):
        return obj.get_text(" ", strip=True)

    # dict-like (FeedParserDict is dict subclass) -> sanitize values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # drop callables (methods/functions)
            if callable(v):
                continue
            out[str(k)] = json_sanitize(v)
        return out

    # objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return json_sanitize(vars(obj))
        except Exception:
            return str(obj)

    # fallback
    return str(obj)

def to_iso8601(value):
    """Return ISO-8601 string from many date representations (datetime, epoch, str)."""
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date)):
        # ensure timezone-aware UTC for consistency
        if isinstance(value, dt.datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc).isoformat()
    if isinstance(value, (int, float)):  # epoch seconds
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).isoformat()
    if isinstance(value, str):
        s = value.strip()
        # ISO first
        try:
            return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc).isoformat()
        except Exception:
            pass
        # RFC-822 (common in RSS)
        try:
            return parsedate_to_datetime(s).astimezone(dt.timezone.utc).isoformat()
        except Exception:
            pass
        # Give back the original string if unparseable (still stored, won’t crash)
        return s
    return None

def parse_any_dt(value):
    """Return a timezone-aware UTC datetime from many inputs; None if impossible."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time(0), tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        for fn in (
            lambda x: dt.datetime.fromisoformat(x),
            lambda x: parsedate_to_datetime(x),
        ):
            try:
                out = fn(s)
                return out if out.tzinfo else out.replace(tzinfo=dt.timezone.utc)
            except Exception:
                pass
    return None

def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)

import time
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # Built-in with requests

def parse_rss(feed_url: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    # Setup session with retries (max 3, backoff 1-2s)
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        logging.info("Downloading feed: %s", feed_url)
        response = session.get(feed_url, timeout=30)
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("Content-Type", "").lower()
        
        # Early bail if not XML-ish
        if "xml" not in content_type and "rss" not in content_type and "atom" not in content_type:
            logging.warning("Non-XML Content-Type '%s' for %s", content_type, feed_url)
            return entries
            
    except Exception as exc:
        logging.error("Failed to download feed %s: %s", feed_url, exc)
        return entries

    if feedparser is not None:
        parsed = feedparser.parse(content)
        if parsed.bozo:  # feedparser error flag
            logging.warning("Feedparser warnings for %s: %s", feed_url, parsed.get("bozo_exception", "Unknown"))
        for entry in parsed.entries or []:
            # Normalize keys (Atom uses 'summary', RSS 'description')
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            published = getattr(entry, "published", None) or getattr(entry, "updated", None)
            description = getattr(entry, "summary", getattr(entry, "description", "")).strip()
            
            # Validate: Skip if no title/link
            if not title or not link:
                continue
            
            entries.append({
                "title": title,
                "link": link,
                "published": published,
                "description": description,
            })
    else:
        # Enhanced fallback: Handle RSS 2.0 and basic Atom
        import xml.etree.ElementTree as ET
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}  # For Atom
        try:
            tree = ET.fromstring(content)
            # RSS: channel/item
            items = tree.findall(".//item") or tree.findall(".//atom:entry", namespaces)
            for item in items:
                title_el = item.find("title") or item.find("atom:title", namespaces)
                link_el = item.find("link") or item.find("atom:link[@rel='alternate']/@href", namespaces)
                pub_el = item.find("pubDate") or item.find("atom:published", namespaces)
                desc_el = item.find("description") or item.find("atom:summary", namespaces)
                
                title = (title_el.text if title_el is not None else "").strip()
                link = (link_el.text if link_el is not None and link_el.text else 
                        link_el.attrib.get("href", "") if hasattr(link_el, "attrib") else "").strip()
                published = (pub_el.text if pub_el is not None else None).strip()
                description = (desc_el.text if desc_el is not None else "").strip()
                
                if not title or not link:
                    continue
                
                entries.append({
                    "title": title,
                    "link": link,
                    "published": published,
                    "description": description,
                })
        except ET.ParseError as exc:
            logging.error("XML parse failed for %s: %s", feed_url, exc)
            return entries
    
    logging.info("Parsed %d valid entries from %s", len(entries), feed_url)
    time.sleep(2)  # Rate limit
    return entries

def fetch_url(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[str]:
    """
    Fetch a URL and return decoded text only for HTML/XML pages.
    Returns None for non-HTML content (PDF, images, octet-stream, etc.).
    """
    if not url:
        return None

    base_headers = {
        "User-Agent": "Sean Slattery STAT429 Pipeline (contact: sean@example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    if headers:
        base_headers.update(headers)

    for attempt in range(2):  # small retry
        try:
            resp = requests.get(url, headers=base_headers, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()

            ctype = (resp.headers.get("Content-Type") or "").lower()
            # Allow HTML or XML-ish types only
            if ("text/html" in ctype) or ("application/xhtml" in ctype) or ("xml" in ctype):
                resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
                text = resp.text
                # quick sanity check to avoid passing junk to extractor
                if text and any(tag in text[:1024].lower() for tag in ("<html", "<!doctype", "<head", "<body")):
                    return text
                # If it's XML (RSS/Atom) you probably don't want to extract article text
                if "xml" in ctype:
                    return None
                # anything else: probably not real HTML
                return None
            else:
                logging.info("Non-HTML Content-Type '%s' at %s — skipping extraction", ctype, url)
                return None

        except Exception as e:
            if attempt == 0:
                time.sleep(0.5)
                continue
            logging.warning("Error downloading %s: %s", url, e)
            return None

    return None

def fetch_binary(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[bytes]:
    base_headers = {
        "User-Agent": "Sean Slattery STAT429 Pipeline (contact: seants@illinois.edu)",
        "Accept": "*/*",
        "Accept-Encoding": "identity",  # easier for raw save
    }
    if headers:
        base_headers.update(headers)
    try:
        r = requests.get(url, headers=base_headers, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logging.warning("Binary download failed for %s: %s", url, e)
        return None


import sys
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def _looks_like_html(s: Optional[str]) -> bool:
    if not s or len(s) < 50:
        return False
    head = s.lstrip()[:1024].lower()
    return ("<html" in head) or ("<!doctype" in head) or ("<head" in head) or ("<body" in head)

def extract_text_and_meta(html: Optional[str]) -> Dict[str, Any]:
    """
    Extract main text, title, date, authors from an HTML page.
    Returns empty text if the input is None or not HTML-like.
    """
    out: Dict[str, Any] = {"text": "", "title": None, "date": None, "authors": None}

    if not _looks_like_html(html):
        return out

    # --- Try Trafilatura if available ---
    try:
        import trafilatura
        downloaded = html  # we already fetched the text; no need to fetch again
        meta = trafilatura.extract(downloaded, include_comments=False, include_tables=False, output="json")
        if meta:
            try:
                j = json.loads(meta)
                out["text"]    = (j.get("text") or "").strip()
                out["title"]   = j.get("title")
                out["date"]    = to_iso8601(j.get("date")) if j.get("date") else None
                out["authors"] = j.get("author") or j.get("authors")
                if out["text"]:
                    return out
            except Exception:
                pass
    except Exception:
        # Trafilatura not installed or error inside
        pass

    # --- Fallback: BeautifulSoup plain-text extraction ---
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml") if "lxml" in sys.modules else BeautifulSoup(html, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Title
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        # Date candidates (very heuristic)
        date_str = None
        for tag in soup.find_all(["time", "meta"]):
            try:
                if tag.name == "time" and tag.get("datetime"):
                    date_str = tag.get("datetime")
                    break
                if tag.name == "meta" and tag.get("property") in ("article:published_time", "og:updated_time"):
                    date_str = tag.get("content")
                    break
            except Exception:
                continue
        text = soup.get_text(" ", strip=True)
        out["text"] = text
        out["title"] = title
        out["date"] = to_iso8601(date_str) if date_str else None
        out["authors"] = None
    except Exception:
        # Give up: return empty text; caller will skip it
        pass

    return out



import hashlib
import json
from typing import Union


def compute_sha256(content: Union[str, bytes, dict, list]) -> str:
    """
    Return SHA-256 hex digest for strings, bytes, or JSON-serializable objects.
    - str  -> UTF-8 encode
    - bytes -> as-is
    - other -> json.dumps after json_sanitize
    """
    if isinstance(content, str):
        data = content.encode("utf-8")
    elif isinstance(content, (bytes, bytearray, memoryview)):
        data = bytes(content)
    else:
        data = json.dumps(json_sanitize(content), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def classify_commodity_improved(title: str, text: str) -> Optional[str]:
    """
    Improved heuristic with multi-tagging: Returns comma-separated tags if multiple score above threshold.
    """
    combined = f"{title} {text}".lower()
    scores = {"ng": 0, "oil": 0, "power": 0}
    
    # Natural Gas (weight 1)
    gas_kws = ["natural gas", "natgas", "henry hub", "lng", "shale gas", "gas price", "gas demand", "gas market", "gas storage", "pipeline gas"]
    for kw in gas_kws:
        scores["ng"] += combined.count(kw)
    
    # Oil (weight 1.5 for balance)
    oil_kws = ["crude", "wti", "brent", "oil", "oil market", "refinery", "opec", "petroleum", "diesel", "gasoline", "jet fuel", "fuel oil", "oil price", "oil demand", "oil production"]
    for kw in oil_kws:
        scores["oil"] += combined.count(kw) * 1.5
    
    # Power (weight 1.2 to boost)
    power_kws = ["electricity", "power", "grid", "renewable", "solar", "wind", "electric grid", "battery", "energy storage"]
    for kw in power_kws:
        scores["power"] += combined.count(kw) * 1.2
    
    # Multi-tag: Collect tags above threshold (e.g., score > 0); sort for consistency
    threshold = 0  # Adjust higher (e.g., 1) to require stronger matches
    tags = sorted([k for k, v in scores.items() if v > threshold])
    if tags:
        return ','.join(tags)  # e.g., 'ng,oil' or 'oil,power'
    return None


def parse_date(date_str: Optional[str]) -> Optional[str]:
    """Parse a date string into ISO 8601 format (UTC) if possible."""
    if not date_str:
        return None
    # Try multiple common date formats used in RSS/Atom feeds and web pages
    # Example: 'Tue, 28 Oct 2025 07:00:00 EST'
    date_str = date_str.strip()
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%d %b %Y",
    ]:
        try:
            dt_obj = dt.datetime.strptime(date_str, fmt)
            # Convert to UTC if timezone aware; otherwise treat as UTC
            return to_iso8601(dt_obj.astimezone(dt.timezone.utc))
        except Exception:
            continue
    return None


def update_sqlite(db_path: str, records: List[Dict[str, Any]]) -> None:
    """
    Insert document records into a SQLite database and create indices if necessary.
    """
    if not records:
        return
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id TEXT PRIMARY KEY,
                source TEXT,
                url TEXT,
                published_utc TEXT,
                ingested_utc TEXT,
                title TEXT,
                text TEXT,
                tickers TEXT,  -- Added for SEC/related tickers (e.g., 'XOM,CVX')
                commodity_tag TEXT,
                language TEXT,
                source_type TEXT,
                checksum TEXT,
                meta TEXT
            )
            """
        )
        for rec in records:
            cur.execute(
                """
                INSERT OR REPLACE INTO docs (doc_id, source, url, published_utc, ingested_utc,
                    title, text, tickers, commodity_tag, language, source_type, checksum, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["doc_id"], rec["source"], rec["url"], rec["published_utc"], rec["ingested_utc"],
                    rec["title"], rec["text"], rec.get("tickers"), rec.get("commodity_tag"),
                    rec.get("language"), rec["source_type"], rec["checksum"], json.dumps(rec.get("meta", {}))
                ),
            )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_published ON docs (published_utc)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_source ON docs (source)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_tag ON docs (commodity_tag)")
        conn.commit()
    finally:
        conn.close()
def process_eia_feeds(feed_urls: Dict[str, str], storage_root: str) -> List[Dict[str, Any]]:
    """
    Accepts either:
      A) {"todayinenergy": "https://www.eia.gov/rss/todayinenergy.xml", ...}
      B) {"https://www.eia.gov/rss/todayinenergy.xml": "https://www.eia.gov/", ...}

    and normalizes to (name, rss_url, base_url).
    """
    documents: List[Dict[str, Any]] = []

    # --- normalize feed specs to a list of tuples: (name, rss_url, base_url) ---
    norm_specs: List[Tuple[str, str, str]] = []
    for k, v in feed_urls.items():
        if k.lower().endswith(".xml"):
            # Style B: {rss_url: base_url}
            rss_url, base_url = k, (v or "")
            name = Path(rss_url).stem  # e.g., todayinenergy
        else:
            # Style A: {name: rss_url}
            name, rss_url = k, v
            base_url = ""
        # derive base_url from rss_url if missing
        try:
            parsed = urlparse(rss_url)
            if not base_url and parsed.scheme and parsed.netloc:
                base_url = f"{parsed.scheme}://{parsed.netloc}/"
        except Exception:
            pass
        if not base_url:
            base_url = "https://www.eia.gov/"
        norm_specs.append((name, rss_url, base_url))

    # --- process each feed ---
    for name, rss_url, base_url in norm_specs:
        if not isinstance(rss_url, str) or ".xml" not in rss_url.lower():
            logging.error("Skipping non-RSS URL for feed '%s': %s", name, rss_url)
            continue

        logging.info("Downloading feed: %s", rss_url)
        entries = parse_rss(rss_url)
        logging.info("Parsed %d entries from feed '%s'", len(entries), name)

        for entry in entries:
            # Absolute article URL
            raw_link = entry.get("link") or entry.get("id") or entry.get("href")
            article_url = absolutize(raw_link, base_url)
            if not article_url and entry.get("links"):
                for lk in entry["links"]:
                    if isinstance(lk, dict):
                        article_url = absolutize(lk.get("href"), base_url)
                        if article_url:
                            break
            if not article_url:
                logging.warning("Skipping entry with no resolvable URL: %s", entry.get("title"))
                continue

            html = fetch_url(article_url)
            if not html:
                continue

            article_meta = extract_text_and_meta(html)
            text = (article_meta.get("text") or "").strip()
            title = article_meta.get("title") or entry.get("title") or ""
            if not text:
                continue

            # Deterministic ID on URL prevents duplicates across runs
            doc_id = compute_sha256(article_url.encode("utf-8"))

            published_utc = (
                parse_date(entry.get("published"))
                or parse_date(entry.get("updated"))
                or parse_date(article_meta.get("date"))
            )
            ingested_utc = to_iso8601(dt.datetime.now(dt.timezone.utc))
            checksum = compute_sha256(text)
            commodity_tag = classify_commodity_improved(title, text)

            pub_dt = parse_any_dt(published_utc) or dt.datetime.now(dt.timezone.utc)
            y, m, d = pub_dt.year, f"{pub_dt.month:02d}", f"{pub_dt.day:02d}"
            raw_dir = os.path.join(storage_root, "raw", str(y), m, d)
            ensure_directory(raw_dir)

            html_path = os.path.join(raw_dir, f"{doc_id}.html")
            json_path = os.path.join(raw_dir, f"{doc_id}.json")

            # If you want to skip re-downloads: uncomment this guard
            # if os.path.exists(html_path) and os.path.exists(json_path):
            #     logging.info("Already have %s, skipping", article_url)
            #     continue

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            lean_entry = {k: entry.get(k) for k in ("title", "link", "published", "updated", "summary", "id") if k in entry}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feed": name,
                        "rss_url": rss_url,
                        "base_url": base_url,
                        "article_url": article_url,
                        "entry": json_sanitize(lean_entry),
                        "extraction": json_sanitize(article_meta),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            rec = {
                "doc_id": doc_id,
                "source": f"eia_{name}",
                "url": article_url,
                "published_utc": published_utc,
                "ingested_utc": ingested_utc,
                "title": article_meta.get("title") or entry.get("title"),
                "text": text,
                "tickers": None,
                "commodity_tag": commodity_tag,
                "language": None,
                "source_type": "rss",
                "checksum": checksum,
                "meta": {
                    "authors": article_meta.get("authors"),
                    "raw_description": entry.get("description"),
                },
            }
            documents.append(rec)
            logging.info("Processed EIA article '%s' (%s)", rec["title"], doc_id)

        time.sleep(1)  # politeness

    return documents




def process_sec_filings(ciks: List[str], storage_root: str) -> List[Dict[str, Any]]:
    """
    Download and extract recent Form 8-K filings for a list of CIKs, using the
    SEC submissions JSON endpoint and the EDGAR Archives for primary documents.
    """
    records: List[Dict[str, Any]] = []

    # IMPORTANT: Use a real identifying UA per SEC guidelines
    sec_headers = {
        "User-Agent": "Sean Slattery Commodities Analysis Project (seants2@illinois.edu)",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Host": "data.sec.gov",
    }

    for cik in ciks:
        # Normalize CIK → 10-digit with leading zeros for the submissions endpoint
        cik_str = str(cik).lstrip().replace("CIK", "").lstrip("0")
        cik10 = f"{int(cik_str):010d}"
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"

        logging.info("Fetching submissions for CIK %s", cik10)
        try:
            resp = requests.get(submissions_url, headers=sec_headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logging.error("Failed to fetch submissions for CIK %s: %s", cik10, exc)
            continue

        recent = (data.get("filings") or {}).get("recent") or {}
        forms              = list(recent.get("form") or [])
        accession_numbers  = list(recent.get("accessionNumber") or [])
        primary_documents  = list(recent.get("primaryDocument") or [])
        filing_dates       = list(recent.get("filingDate") or [])

        # Zip defensively to avoid index errors if arrays are uneven
        for form, accession, primary_doc, filing_date in zip(forms, accession_numbers, primary_documents, filing_dates):
            if not isinstance(form, str):
                continue
            if not form.upper().startswith("8-K"):  # catches "8-K" and "8-K/A"
                continue
            if not accession or not primary_doc:
                continue

            # Build the document URL: /Archives/edgar/data/{cik_no_leading_zeros}/{accessionnodashes}/{primary_doc}
            try:
                cik_int = int(cik_str)
            except Exception:
                # Fallback if somehow cik_str can't be parsed (shouldn't happen after formatting above)
                cik_int = int(cik10)

            accession_no = str(accession).replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no}/{primary_doc}"

            # Fetch the filing document (HTML) with a polite UA (SEC domain, not data.sec.gov)
            html_headers = {
                "User-Agent": sec_headers["User-Agent"],
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Host": "www.sec.gov",
            }
            html = fetch_url(filing_url, headers=html_headers)
            if not html:
                continue

            # Extract text and metadata
            meta = extract_text_and_meta(html)
            text = (meta.get("text") or "").strip()
            title = meta.get("title") or form
            if not text:
                continue

            # Deterministic doc_id so re-runs overwrite the same file instead of duplicating
            doc_id = compute_sha256(filing_url.encode("utf-8"))
            published_utc = parse_date(filing_date) or to_iso8601(filing_date)
            ingested_utc = to_iso8601(dt.datetime.now(dt.timezone.utc))
            checksum = compute_sha256(text)
            
            commodity_tag = classify_commodity_improved(title, text)

            # Choose YYYY/MM/DD based on filing/published date (fallback = now UTC)
            pub_dt = parse_any_dt(published_utc) or dt.datetime.now(dt.timezone.utc)
            y, m, d = pub_dt.year, f"{pub_dt.month:02d}", f"{pub_dt.day:02d}"
            raw_dir = os.path.join(storage_root, "raw", str(y), m, d)
            ensure_directory(raw_dir)

            html_path = os.path.join(raw_dir, f"{doc_id}.html")
            json_path = os.path.join(raw_dir, f"{doc_id}.json")

            # If you prefer skipping when already present, uncomment:
            # if os.path.exists(html_path) and os.path.exists(json_path):
            #     logging.info("Already have filing: %s (doc_id=%s)", filing_url, doc_id)
            #     continue

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cik": cik10,
                        "form": form,
                        "accession": accession,
                        "filing_date": to_iso8601(filing_date),
                        "filing_url": filing_url,
                        "extraction": json_sanitize(meta),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            rec = {
                "doc_id": doc_id,
                "source": "sec_edgar",
                "url": filing_url,
                "published_utc": published_utc,
                "ingested_utc": ingested_utc,
                "title": meta.get("title") or form,
                "text": text,
                "tickers": None,
                "commodity_tag": commodity_tag,
                "language": None,
                "source_type": "web",
                "checksum": checksum,
                "meta": {
                    "authors": meta.get("authors"),
                    "sec_form": form,
                    "filing_date": to_iso8601(filing_date),
                    "accession": accession,
                    "primary_document": primary_doc,
                    "cik": cik10,
                },
            }
            records.append(rec)
            logging.info("Processed SEC filing %s for CIK %s", accession, cik10)

            # Be polite to SEC servers
            time.sleep(0.5)

        # Small pause between companies
        time.sleep(1.0)

    return records



def process_gdelt(query: str, max_records: int, storage_root: str) -> List[Dict[str, Any]]:
    """
    Search GDELT DOC 2.0 API for articles and process results.
    Handles non-JSON errors gracefully.
    """
    records: List[Dict[str, Any]] = []
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "hybridrel",
        "timespan": "1m",  # Last month for recency
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    try:
        logging.info("Querying GDELT with '%s'", query)
        resp = requests.get(url, params=params, timeout=60)
        logging.info("GDELT Status: %d, Content-Type: %s", resp.status_code, resp.headers.get('Content-Type'))
        
        if resp.status_code != 200:
            logging.error("GDELT non-200 status: %d - %s", resp.status_code, resp.text[:500])
            return records
        
        content_type = resp.headers.get('Content-Type', '').lower()
        if 'application/json' not in content_type:
            logging.error("GDELT non-JSON response: %s - Preview: %s", content_type, resp.text[:500])
            return records
        
        data = resp.json()
        
    except requests.exceptions.RequestException as exc:
        logging.error("GDELT request failed: %s", exc)
        return records
    except Exception as exc:  # Includes JSONDecodeError
        logging.error("GDELT parse failed: %s - Response: %s", exc, resp.text[:500] if 'resp' in locals() else "No response")
        return records
    
    articles = data.get("articles", [])
    logging.info("GDELT returned %d articles", len(articles))
    
    for art in articles:
        article_url = art.get("url")
        if not article_url:
            continue
        title = art.get("title", "")
        date_str = art.get("seendate")  # UTC timestamp (e.g., "20251030T123456")
        html = fetch_url(article_url)
        if not html:
            continue
        meta = extract_text_and_meta(html)
        text = (meta.get("text") or "").strip()
        if not text:
            continue
        doc_id = compute_sha256(article_url)
        published_utc = to_iso8601(date_str)  # Convert timestamp to ISO
        ingested_utc = to_iso8601(dt.datetime.now(dt.timezone.utc))
        checksum = compute_sha256(text)
        commodity_tag = classify_commodity_improved(title, text)  # Ensure title passed
        
        # Raw storage (by pub date)
        pub_dt = parse_any_dt(published_utc) or dt.datetime.now(dt.timezone.utc)
        y, m, d = pub_dt.year, f"{pub_dt.month:02d}", f"{pub_dt.day:02d}"
        raw_dir = os.path.join(storage_root, "raw", str(y), m, d)
        ensure_directory(raw_dir)
        html_path = os.path.join(raw_dir, f"{doc_id}.html")
        json_path = os.path.join(raw_dir, f"{doc_id}.json")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"gdelt": art, "extraction": meta}, f, ensure_ascii=False, indent=2)
        
        rec = {
            "doc_id": doc_id,
            "source": "gdelt",
            "url": article_url,
            "published_utc": published_utc,
            "ingested_utc": ingested_utc,
            "title": meta.get("title") or title,
            "text": text,
            "commodity_tag": commodity_tag,
            "language": None,
            "source_type": "api",
            "checksum": checksum,
            "meta": {
                "authors": meta.get("authors"),
                "gdelt_source": art.get("sourceCommonName"),
                "gdelt_country": art.get("sourceCountry"),
                "score": art.get("socialscore"),
            },
        }
        records.append(rec)
        logging.info("Processed GDELT article '%s'", title)
    
    return records


def save_parquet(records: List[Dict[str, Any]], storage_root: str) -> Optional[str]:
    """
    Save normalised records to a Parquet file and return the file path.

    Parameters
    ----------
    records : list of dict
        Normalised document records.
    storage_root : str
        Root directory for storing the Parquet files.

    Returns
    -------
    Optional[str]
        Path to the Parquet file if records exist, else None.
    """
    if not records:
        return None
    df = pd.DataFrame(records)
    # Determine path based on current date
    now = dt.datetime.utcnow()
    year, month, day = now.year, f"{now.month:02d}", f"{now.day:02d}"
    docs_dir = os.path.join(storage_root, "docs", str(year), month)
    ensure_directory(docs_dir)
    parquet_path = os.path.join(docs_dir, f"docs_{year}{month}{day}.parquet")
    df.to_parquet(parquet_path, index=False)
    logging.info("Saved %d records to Parquet at %s", len(records), parquet_path)
    return parquet_path


import os
from typing import List, Dict, Any
from urllib.parse import urlparse

def process_congress_hearings(committee_codes: List[str], storage_root: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch recent hearings via Congress.gov v3 API, extract transcripts, and normalize.
    Supports Senate (S...) and House (H...) codes automatically.
    """
    records: List[Dict[str, Any]] = []
    base_url = "https://api.congress.gov/v3/hearing"
    headers = {"User-Agent": "NewsPipeline/1.0 (your.email@example.com)"}  # Polite UA
    
    for code in committee_codes:
        params = {
            "congress": 119,  # 2025-2026 session
            "committee": code,
            "limit": 20,
            "startDate": "2025-01-01",  # Per page
            "format": "json",
            "api_key": api_key,
        }
        # No chamber param—API infers from code (S/H prefix)
        offset = 0
        while True:  # Pagination
            params["offset"] = offset
            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=30)
                if resp.status_code != 200:
                    logging.error("Congress Status %d for %s: %s", resp.status_code, code, resp.text[:500])
                    break
                content_type = resp.headers.get('Content-Type', '').lower()
                if 'application/json' not in content_type:
                    logging.error("Congress non-JSON for %s: %s", code, resp.text[:500])
                    break
                data = resp.json()
                hearings = data.get("hearings", [])
                if not hearings:
                    break
                
                for hearing in hearings:
                    title = hearing.get("title", "")
                    hearing_date = hearing.get("hearingDate", "")
                    transcript_url = hearing.get("formatUrl", {}).get("html", "")
                    if not transcript_url:
                        logging.info("Skipping '%s' (no transcript): %s", title, hearing_date)
                        continue
                    
                    html = fetch_url(transcript_url, headers=headers)
                    if not html:
                        continue
                    
                    meta = extract_text_and_meta(html)
                    text = meta.get("text", "").strip()
                    if not text:
                        continue
                    
                    doc_id = compute_sha256(transcript_url)
                    published_utc = to_iso8601(hearing_date)
                    ingested_utc = to_iso8601(dt.datetime.now(dt.timezone.utc))
                    checksum = compute_sha256(text)
                    title_final = meta.get("title") or title
                    commodity_tag = classify_commodity_improved(title_final, text)
                    
                    # Raw storage
                    pub_dt = parse_any_dt(published_utc) or dt.datetime.now(dt.timezone.utc)
                    y, m, d = pub_dt.year, f"{pub_dt.month:02d}", f"{pub_dt.day:02d}"
                    raw_dir = os.path.join(storage_root, "raw", str(y), m, d)
                    ensure_directory(raw_dir)
                    html_path = os.path.join(raw_dir, f"{doc_id}.html")
                    json_path = os.path.join(raw_dir, f"{doc_id}.json")
                    
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "api": "congress_v3",
                            "committee": code,
                            "hearing": hearing,
                            "extraction": meta,
                        }, f, ensure_ascii=False, indent=2)
                    
                    rec = {
                        "doc_id": doc_id,
                        "source": f"congress_{code}",
                        "url": transcript_url,
                        "published_utc": published_utc,
                        "ingested_utc": ingested_utc,
                        "title": title_final,
                        "text": text,
                        "commodity_tag": commodity_tag,
                        "language": None,
                        "source_type": "api",
                        "checksum": checksum,
                        "meta": {
                            "authors": meta.get("authors"),
                            "congress": 119,
                            "committee": code,
                            "hearing_date": hearing_date,
                        },
                    }
                    records.append(rec)
                    logging.info("Processed hearing '%s' (%s)", title_final, doc_id)
                
                if len(hearings) < params["limit"]:
                    break
                offset += params["limit"]
                time.sleep(0.5)
                
            except Exception as exc:
                logging.error("Congress API failed for %s (offset %d): %s", code, offset, exc)
                break
        
        time.sleep(1)
    
    return records

def main() -> None:
    """Example workflow for the news ingestion pipeline."""
    from datetime import datetime, timezone  # Import timezone separately
    from typing import List, Dict, Any
    import argparse
    import os
    import logging
    import pandas as pd
    from datetime import datetime as dt
    from pathlib import Path
    from urllib.parse import urlparse  # For base_url if needed in process_eia_feeds

    # Configure logging early
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="News Ingestion Pipeline")
    parser.add_argument("--test", action="store_true", help="Test feed parsing without full ingestion")
    parser.add_argument("--api-key", required=False, help="Congress.gov API key (or set CONGRESS_API_KEY env var)")
    args = parser.parse_args()

    # Define storage root EARLY, before any conditional blocks
    storage_root = os.path.join(os.path.dirname(__file__), "data", "news")
    ensure_directory(storage_root)  # Assuming defined

    # Get Congress API key (env fallback)
    api_key = args.api_key or os.getenv('CONGRESS_API_KEY') or "fdZfN78AsqgwAJwC5LO1uxmCifyTvFWAo5vC9tbG"
    if not api_key:
        logging.warning("No Congress API key provided—skipping Congress hearings.")

    # EIA RSS feeds: {name: rss_url} (base_url derived as f"{scheme}://{netloc}/" if needed)
    eia_feeds = {
        "todayinenergy": "https://www.eia.gov/rss/todayinenergy.xml",
        "whatsnew": "https://www.eia.gov/about/new/WNtest3.php",  # PHP-generated RSS
        "testimony": "https://www.eia.gov/rss/testimony.xml",
        "presentations": "https://www.eia.gov/rss/presentations.xml",
        "petroleum_gasdiesel": "https://www.eia.gov/petroleum/gasdiesel/includes/gas_diesel_rss.xml",
        "petroleum_hopu": "https://www.eia.gov/petroleum/heatingoilpropane/includes/hopu_rss.xml",
    }

    # DOE/FECM RSS feeds: Official URLs from https://www.energy.gov/fecm/fossil-energy-rss-feeds (Oct 30, 2025)
    # These use ?view=rss and deliver valid RSS 2.0; if process_eia_feeds skips, remove any .xml check
    doe_feeds = {
        "all_fossil_energy": "https://www.energy.gov/fecm/fe/techlines?view=rss",
        "clean_coal": "https://www.energy.gov/fecm/fe/clean-coal-news?view=rss",
        "carbon_capture": "https://www.energy.gov/fecm/fe/carbon-capture-storage-news?view=rss",
        "oil_natural_gas": "https://www.energy.gov/fecm/fe/oil-natural-gas-news?view=rss",
        "petroleum_reserves": "https://www.energy.gov/fecm/fe/petroleum-reserves-news?view=rss",
        "fossil_energy_blog": "https://www.energy.gov/fecm/fe/blog?view=rss",
    }

    # Corrected CIKs for SEC (ExxonMobil: 0000034088, Chevron: 0000093410)
    ciks = ["0000034088", "0000093410"]

    if args.test:
        print("Running in test mode: Parsing RSS feeds without full ingestion...")
        all_rss_feeds = {**eia_feeds, **doe_feeds}
        total_items = 0
        for name, rss_url in all_rss_feeds.items():
            try:
                items = parse_rss(rss_url)
                count = len(items)
                print(f"Feed '{name}' ({rss_url}): {count} items")
                total_items += count
            except Exception as exc:
                print(f"Feed '{name}' ({rss_url}): Error - {exc}")
        print(f"Total RSS items across feeds: {total_items}")
        print("Non-RSS sources (Congress, SEC, GDELT) skipped in test mode.")
        return

    # Normal mode: Process all sources
    all_records: List[Dict[str, Any]] = []

    # EIA
    print("Processing EIA feeds...")
    eia_records = process_eia_feeds(eia_feeds, storage_root)
    all_records.extend(eia_records)

    # DOE/FECM
    print("Processing DOE/FECM feeds...")
    doe_records = process_eia_feeds(doe_feeds, storage_root)
    all_records.extend(doe_records)

    # Congress
    # print("Processing Congress hearings...")
    # congress_records: List[Dict[str, Any]] = []
    #if api_key:
     #   congress_codes = ["SSEN", "HSII", "HSII16", "HSIF"]
     #   congress_records = process_congress_hearings(congress_codes, storage_root, api_key)
    #else:
     #   logging.warning("Congress skipped: No API key.")
    #all_records.extend(congress_records)

    # SEC (with correct CIKs)
    print("Processing SEC filings...")
    sec_records = process_sec_filings(ciks, storage_root)
    all_records.extend(sec_records)

    # GDELT (ensure params["timespan"] = "1m" in process_gdelt for recent focus)
    print("Processing GDELT...")
    gdelt_records = process_gdelt("(energy OR oil OR \"natural gas\" OR petroleum)", max_records=50, storage_root=storage_root)
    all_records.extend(gdelt_records)

    if not all_records:
        logging.warning("No records retrieved. Check network/feed URLs.")
        return

    # Save Parquet (by latest pub date)
    print("Saving Parquet...")
    latest_pub = max(parse_any_dt(r["published_utc"]) or datetime.now(tz=timezone.utc) for r in all_records)
    out_dir = Path(storage_root) / "docs" / f"{latest_pub.year}" / f"{latest_pub.month:02d}"
    ensure_directory(str(out_dir))
    out_path = out_dir / f"docs_{latest_pub.year}{latest_pub.month:02d}{latest_pub.day:02d}.parquet"
    df = pd.DataFrame(all_records)
    df.to_parquet(out_path, index=False)
    logging.info("Saved %d records to Parquet at %s", len(df), out_path)

    # SQLite
    print("Updating SQLite...")
    db_path = os.path.join(storage_root, "news_catalog.db")
    update_sqlite(db_path, all_records)
    logging.info("Pipeline complete. Parquet: %s, SQLite: %s", out_path, db_path)

if __name__ == "__main__":
    main()