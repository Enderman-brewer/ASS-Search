#!/usr/bin/env python3
# A.S.S Search v2.8 — Merged ranking (TF-IDF + BM25 + quality + recency)
# Python 3.12+
from __future__ import annotations
import re
import time
import random
import heapq
import threading
import sqlite3
import math
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
import queue
import signal
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set
from urllib.parse import urlparse, urljoin, urlunparse, parse_qs, urlencode
import requests
import urllib.robotparser
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string, jsonify

import config
from exit_manager import ExitManager

# --- NLTK for advanced tokenization (optional) ---
try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk_available = True
except ImportError:
    nltk_available = False

# -----------------------------------------------

# --- Logger setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = Flask(__name__)
Path(config.SAVE_DIR).mkdir(exist_ok=True)

# --- Stemmer & Stopwords Setup ---
if nltk_available:
    stemmer = PorterStemmer()
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        STOP_WORDS = set()
else:
    stemmer = None
    STOP_WORDS = set()

# --- Concurrency Primitives ---
VISITED_LOCK = threading.RLock()
VISITED: Set[str] = set()
PER_HOST_LAST: Dict[str, float] = {}
PER_HOST_COUNT: Dict[str, int] = {}
PER_HOST_LOCKS: Dict[str, threading.Lock] = {}

# --- Enqueue tracking / Priority Queue ---
cr_queue_lock = threading.RLock()
CR_QUEUE_COND = threading.Condition(cr_queue_lock)
_cr_counter = 0
_cr_heap: List[Tuple[float, int, str]] = []

ENQUEUED_LOCK = threading.RLock()
ENQUEUED: Set[str] = set()

# Cache for observed redirects: source_url -> (target_url, status_code, timestamp)
REDIRECT_CACHE: Dict[str, Tuple[str, int, float]] = {}

# Shutdown event for graceful termination
SHUTDOWN_EVENT = threading.Event()

# Queue for batching database writes
INDEX_QUEUE = queue.Queue()

# --- HTTP Session ---
SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=config.REQUEST_POOL, pool_maxsize=config.REQUEST_POOL, max_retries=1)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)
SESSION.max_redirects = 8

ROBOTS: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}




def cr_count_high_priority() -> int:
    with cr_queue_lock:
        return sum(1 for pri, _, _ in _cr_heap if pri < 0)

# -------------------------
# Tokenization
# -------------------------
TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    if nltk_available and stemmer:
        try:
            words = word_tokenize(text)
            return [stemmer.stem(w) for w in words if w not in STOP_WORDS and TOKEN_RE.fullmatch(w)]
        except Exception:
            pass
    return TOKEN_RE.findall(text)

# -------------------------
# Content quality heuristics
# -------------------------
MIN_USEFUL_WORDS = config.MIN_USEFUL_WORDS  # be less strict — allow smaller pages through
HARD_SKIP_WORDS = config.HARD_SKIP_WORDS   # only very tiny pages should be treated as hard-skip
SPAM_KEYWORDS = config.SPAM_KEYWORDS


def extract_text_from_soup(soup: BeautifulSoup) -> str:
    # Remove non-content tags to clean up the source
    for tag in soup(['nav', 'header', 'footer', 'aside', 'script', 'style', 'form', 'button', 'iframe', 'img']):
        tag.decompose()

    # A more comprehensive list of text-rich tags
    text_tags = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section", "main", "pre", "code", "td", "th", "blockquote"]
    
    # Try to find a main content area
    main_content = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"}) or soup.find("div", {"id": "main"}) or soup.find("div", {"id": "content"})
    search_area = main_content if main_content else soup

    # Get text from the identified area
    parts = [el.get_text(" ", strip=True) for el in search_area.find_all(text_tags) if el.get_text(" ", strip=True)]
    
    # If the targeted search yields little, fall back to broader text extraction from the whole body
    if len(" ".join(parts)) < MIN_USEFUL_WORDS:
        all_text_parts = [text for text in soup.body.stripped_strings] if soup.body else []
        # Combine and deduplicate while preserving order
        combined = list(dict.fromkeys(parts + all_text_parts))
        return " ".join(combined)

    return " ".join(parts)


def assess_content_quality(html: str, scraped_fields: Dict[str, Any], soup: Optional[BeautifulSoup] = None) -> Dict[str, Any]:
    try:
        s = soup if soup is not None else BeautifulSoup(html or "", "html.parser")
    except Exception:
        s = BeautifulSoup("", "html.parser")

    text = extract_text_from_soup(s)
    word_count = len(tokenize(text))
    html_len = len(html or "")
    text_len = len(text)

    if word_count <= 0:
        base = 0.0
    else:
        base = min(1.0, word_count / float(max(1, MIN_USEFUL_WORDS)))

    hard_flag = word_count < HARD_SKIP_WORDS

    ratio = (text_len / html_len) if html_len > 0 else 0.0
    ratio_penalty = 0.0
    if ratio < 0.05:
        ratio_penalty = 0.5
    elif ratio < 0.15:
        ratio_penalty = 0.2

    lower_html = (html or "").lower()
    spam_hits = sum(1 for k in SPAM_KEYWORDS if k in lower_html)
    spam_penalty = 0.0
    if spam_hits >= 2:
        spam_penalty = 0.5
    elif spam_hits == 1:
        spam_penalty = 0.2

    script_count = len(s.find_all('script'))
    iframe_count = len(s.find_all('iframe'))
    heavy_media = 1.0
    if script_count + iframe_count > 20:
        heavy_media = 0.5
    elif script_count + iframe_count > 5:
        heavy_media = 0.8

    score = base * heavy_media * (1.0 - ratio_penalty) * (1.0 - spam_penalty)
    score = max(0.0, min(1.0, score))

    flags = {
        'word_count': word_count,
        'text_len': text_len,
        'html_len': html_len,
        'text_html_ratio': ratio,
        'spam_hits': spam_hits,
        'script_count': script_count,
        'iframe_count': iframe_count,
        'hard_flag': hard_flag,
    }

    return {'quality': score, 'flags': flags}

# -------------------------
# Persistent Search Index Class (SQLite)
# -------------------------
class SearchIndex:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self._setup_db()

    def _setup_db(self):
        cursor = self._conn.cursor()
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                url TEXT PRIMARY KEY,
                title TEXT,
                meta_description TEXT,
                snippet TEXT,
                fetched_at TEXT,
                is_placeholder INTEGER DEFAULT 0
            )
        ''')
        # ensure compatibility with older DBs: add column if missing
        try:
            cursor.execute("PRAGMA table_info(documents)")
            cols = [r[1] for r in cursor.fetchall()]
            if 'is_placeholder' not in cols:
                cursor.execute('ALTER TABLE documents ADD COLUMN is_placeholder INTEGER DEFAULT 0')
        except Exception:
            # if any error, ignore - best-effort migration
            pass

        # store token -> url mapping with raw counts and doc_len to support multiple ranking algos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inverted_index (
                token TEXT NOT NULL,
                url TEXT NOT NULL,
                term_frequency REAL NOT NULL,
                raw_count INTEGER NOT NULL,
                doc_len INTEGER NOT NULL,
                FOREIGN KEY(url) REFERENCES documents(url) ON DELETE CASCADE
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_token ON inverted_index(token)')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doc_stats (
                url TEXT PRIMARY KEY,
                doc_len INTEGER NOT NULL,
                fetched_at TEXT
            )
        ''')
        self._conn.commit()

    def document_exists(self, url: str) -> bool:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1 FROM documents WHERE url=?", (url,))
            return cursor.fetchone() is not None

    def get_document(self, url: str) -> Optional[Dict[str, Any]]:
        """Return stored document row for url or None."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT url, title, meta_description, snippet, fetched_at, is_placeholder FROM documents WHERE url=?", (url,))
            row = cursor.fetchone()
            if not row:
                return None
            return {"url": row[0], "title": row[1], "meta_description": row[2], "snippet": row[3], "fetched_at": row[4], "is_placeholder": int(row[5] or 0)}

    def index_document(self, url: str, fields: Dict[str, Any]):
        with self._lock:
            # Build meta_description and persist quality if present so search can read it later.
            meta_desc = (fields.get("meta_description") or "").strip()
            if "quality" in fields:
                try:
                    qv = float(fields.get("quality") or 0.0)
                    # append quality marker in a stable format for later parsing
                    meta_desc = (meta_desc + " ").strip() + f" quality:{qv:.4f}"
                except Exception:
                    pass

            text_content = " ".join([fields.get("title", ""), meta_desc, fields.get("snippet", "")])
            tokens = tokenize(text_content)
            doc_len = max(1, len(tokens))
            token_counts = Counter(tokens)

            # determine placeholder flag (default to 0)
            is_placeholder = 1 if fields.get('is_placeholder') else 0

            cursor = self._conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents (url, title, meta_description, snippet, fetched_at, is_placeholder)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (url, fields.get('title', ''), meta_desc, fields.get('snippet', ''), datetime.now(timezone.utc).isoformat(), is_placeholder))

            cursor.execute("DELETE FROM inverted_index WHERE url=?", (url,))
            if tokens:
                index_data = []
                for token, count in token_counts.items():
                    # store normalized tf (count/doc_len) and keep raw_count/doc_len for BM25
                    tf = (count / doc_len)
                    index_data.append((token, url, tf, int(count), int(doc_len)))
                cursor.executemany("INSERT INTO inverted_index (token, url, term_frequency, raw_count, doc_len) VALUES (?, ?, ?, ?, ?)", index_data)

            cursor.execute('INSERT OR REPLACE INTO doc_stats (url, doc_len, fetched_at) VALUES (?, ?, ?)', (url, int(doc_len), datetime.now(timezone.utc).isoformat()))

            self._conn.commit()

    def index_batch(self, batch: List[Tuple[str, Dict[str, Any]]]):
        with self._lock:
            cursor = self._conn.cursor()
            
            docs_to_insert = []
            tokens_to_delete = []
            index_data_to_insert = []
            stats_to_insert = []

            for url, fields in batch:
                meta_desc = (fields.get("meta_description") or "").strip()
                if "quality" in fields:
                    try:
                        qv = float(fields.get("quality") or 0.0)
                        meta_desc = (meta_desc + " ").strip() + f" quality:{qv:.4f}"
                    except Exception:
                        pass

                text_content = " ".join([fields.get("title", ""), meta_desc, fields.get("snippet", "")])
                tokens = tokenize(text_content)
                doc_len = max(1, len(tokens))
                token_counts = Counter(tokens)
                is_placeholder = 1 if fields.get('is_placeholder') else 0

                docs_to_insert.append((url, fields.get('title', ''), meta_desc, fields.get('snippet', ''), datetime.now(timezone.utc).isoformat(), is_placeholder))
                tokens_to_delete.append((url,))

                if tokens:
                    for token, count in token_counts.items():
                        tf = (count / doc_len)
                        index_data_to_insert.append((token, url, tf, int(count), int(doc_len)))
                
                stats_to_insert.append((url, int(doc_len), datetime.now(timezone.utc).isoformat()))

            cursor.executemany('''
                INSERT OR REPLACE INTO documents (url, title, meta_description, snippet, fetched_at, is_placeholder)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', docs_to_insert)
            
            cursor.executemany("DELETE FROM inverted_index WHERE url=?", tokens_to_delete)
            
            if index_data_to_insert:
                cursor.executemany("INSERT INTO inverted_index (token, url, term_frequency, raw_count, doc_len) VALUES (?, ?, ?, ?, ?)", index_data_to_insert)

            cursor.executemany('INSERT OR REPLACE INTO doc_stats (url, doc_len, fetched_at) VALUES (?, ?, ?)', stats_to_insert)

            self._conn.commit()

    def search(self, query: str, algorithm: str = 'merged'):
        """
        Supported algorithms:
          - tfidf
          - bm25
          - quality
          - hybrid   (kept for backward compatibility)
          - recent
          - merged   (default): blends tfidf + bm25 + quality + recent using config.RANKING_WEIGHTS
          - merged_simple: blends tfidf + bm25 + quality + recent + url_len using config.RANKING_WEIGHTS_SIMPLE
          - merged_advanced: blends tfidf + bm25 + quality + recent + url_len using config.RANKING_WEIGHTS_ADVANCED
        """
        
        # Query parsing
        excluded_tokens = {term.lower() for term in re.findall(r'-(\w+)', query)}
        phrases = [p.lower() for p in re.findall(r'"([^"]+)"', query)]
        
        # Remove excluded terms and phrases from the main query
        query_cleaned = re.sub(r'-(\w+)', '', query)
        query_cleaned = re.sub(r'"([^"]+)"', '', query_cleaned)
        
        query_tokens = list(set(tokenize(query_cleaned)))
        if not query_tokens and not phrases:
            return [], 0

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            if total_docs == 0:
                return [], 0

            # Fetch all matching rows for tokens and phrases
            all_tokens_to_search = query_tokens + [token for phrase in phrases for token in tokenize(phrase)] + list(excluded_tokens)
            placeholders = ','.join('?' for _ in all_tokens_to_search)
            cursor.execute(f"SELECT token, url, term_frequency, raw_count, doc_len FROM inverted_index WHERE token IN ({placeholders})", all_tokens_to_search)
            matching_rows = cursor.fetchall()
            if not matching_rows:
                return [], 0

            # collect rows by token
            token_to_urls: Dict[str, Set[str]] = defaultdict(set)
            token_rows: Dict[str, List[Tuple[str, float, int, int]]] = defaultdict(list)
            for token, url, tf, raw_count, doc_len in matching_rows:
                token_to_urls[token].add(url)
                token_rows[token].append((url, float(tf), int(raw_count), int(doc_len)))

            # prepare score maps
            tfidf_scores: Dict[str, float] = defaultdict(float)
            bm25_scores: Dict[str, float] = defaultdict(float)
            quality_scores: Dict[str, float] = {}
            recent_scores: Dict[str, float] = {}
            url_len_scores: Dict[str, float] = {}

            # TF-IDF style base scoring (uses stored term_frequency)
            for token, rows in token_rows.items():
                df = len(token_to_urls.get(token, set()))
                idf = math.log((total_docs + 1) / (1 + df))
                for url, tf, raw_count, doc_len in rows:
                    tfidf_scores[url] += tf * idf

            # BM25 score
            k1 = 1.5
            b = 0.75
            cursor.execute('SELECT AVG(doc_len) FROM doc_stats')
            avgdl_row = cursor.fetchone()
            avgdl = float(avgdl_row[0]) if avgdl_row and avgdl_row[0] else 1.0
            for token, rows in token_rows.items():
                n_q = len(token_to_urls[token])
                idf = math.log((total_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
                for url, tf_norm, raw_count, doc_len in rows:
                    raw = raw_count
                    score = idf * ((raw * (k1 + 1.0)) / (raw + k1 * (1.0 - b + b * (doc_len / avgdl))))
                    bm25_scores[url] += score

            # get candidate URLs set
            candidate_urls = set(list(tfidf_scores.keys()) + list(bm25_scores.keys()))

            if not candidate_urls:
                return [], 0
                
            # Filter out URLs with excluded tokens
            if excluded_tokens:
                excluded_urls = set()
                for token in excluded_tokens:
                    excluded_urls.update(token_to_urls.get(token, set()))
                candidate_urls -= excluded_urls

            # Filter out URLs that don't contain the exact phrases
            if phrases:
                phrase_urls = set()
                for phrase in phrases:
                    phrase_tokens = tokenize(phrase)
                    if not phrase_tokens:
                        continue
                    
                    # Get URLs that contain all tokens in the phrase
                    urls_with_all_tokens = set.intersection(*(token_to_urls.get(token, set()) for token in phrase_tokens))
                    phrase_urls.update(urls_with_all_tokens)
                
                candidate_urls &= phrase_urls


            if not candidate_urls:
                return [], 0

            # fetch per-doc meta (meta_description, snippet, fetched_at) in one query
            placeholders = ','.join('?' for _ in candidate_urls)
            cursor.execute(f"SELECT url, meta_description, snippet, fetched_at FROM documents WHERE url IN ({placeholders})", list(candidate_urls))
            rows = cursor.fetchall()
            now_ts = time.time()
            for u, meta, snip, fetched_at in rows:
                # parse quality: pattern quality:0.xxx if present
                q = 0.5
                try:
                    if meta:
                        m = re.search(r"quality[:=]\s*([0-9]+(?:\.[0-9]+)?)", meta)
                        if m:
                            q = float(m.group(1))
                except Exception:
                    q = 0.5
                quality_scores[u] = max(0.0, min(1.0, q))

                # recency: use inverse age to create a recency score
                rec_score = 0.0
                try:
                    if fetched_at:
                        fetched_ts = datetime.fromisoformat(fetched_at).timestamp()
                        age = max(1.0, now_ts - fetched_ts)
                        # smaller age -> higher recency value; scale by a simple function (age in days)
                        rec_score = 1.0 / (1.0 + (age / (60.0 * 60.0 * 24.0)))
                except Exception:
                    rec_score = 0.0
                recent_scores[u] = rec_score
                
                # url length score
                url_len_scores[u] = 1.0 / (1.0 + len(u))


            # Normalise helper
            def normalize_map(m: Dict[str, float]) -> Dict[str, float]:
                if not m:
                    return {}
                vals = list(m.values())
                minv = min(vals)
                maxv = max(vals)
                if maxv == minv:
                    return {k: 1.0 for k in m}
                return {k: (v - minv) / (maxv - minv) for k, v in m.items()}

            tfidf_norm = normalize_map(tfidf_scores)
            bm25_norm = normalize_map(bm25_scores)
            quality_norm = normalize_map(quality_scores)
            recent_norm = normalize_map(recent_scores)
            url_len_norm = normalize_map(url_len_scores)

            # If user requested a single algorithm explicitly, honour it for backwards compatibility
            if algorithm in ('tfidf', 'bm25', 'quality', 'recent', 'hybrid'):
                if algorithm == 'tfidf':
                    final_scores = tfidf_norm
                elif algorithm == 'bm25':
                    final_scores = bm25_norm
                elif algorithm == 'quality':
                    final_scores = quality_norm
                elif algorithm == 'recent':
                    final_scores = recent_norm
                elif algorithm == 'hybrid':
                    # simple hybrid (tfidf + quality)
                    final_scores = {}
                    for u in candidate_urls:
                        final_scores[u] = 0.6 * tfidf_norm.get(u, 0.0) + 0.4 * quality_norm.get(u, 0.0)
            elif algorithm == 'merged_simple':
                # merged / auto: blend all signals using config.RANKING_WEIGHTS
                w = config.RANKING_WEIGHTS_SIMPLE
                final_scores = {}
                for u in candidate_urls:
                    final_scores[u] = (
                        w.get("tfidf", 0.0) * tfidf_norm.get(u, 0.0)
                        + w.get("bm25", 0.0) * bm25_norm.get(u, 0.0)
                        + w.get("quality", 0.0) * quality_norm.get(u, 0.0)
                        + w.get("recent", 0.0) * recent_norm.get(u, 0.0)
                        + w.get("url_len", 0.0) * url_len_norm.get(u, 0.0)
                    )
            elif algorithm == 'merged_advanced':
                # merged / auto: blend all signals using config.RANKING_WEIGHTS
                w = config.RANKING_WEIGHTS_ADVANCED
                final_scores = {}
                for u in candidate_urls:
                    final_scores[u] = (
                        w.get("tfidf", 0.0) * tfidf_norm.get(u, 0.0)
                        + w.get("bm25", 0.0) * bm25_norm.get(u, 0.0)
                        + w.get("quality", 0.0) * quality_norm.get(u, 0.0)
                        + w.get("recent", 0.0) * recent_norm.get(u, 0.0)
                        + w.get("url_len", 0.0) * url_len_norm.get(u, 0.0)
                    )
            else:
                # merged / auto: blend all signals using config.RANKING_WEIGHTS
                w = config.RANKING_WEIGHTS
                final_scores = {}
                for u in candidate_urls:
                    final_scores[u] = (
                        w.get("tfidf", 0.0) * tfidf_norm.get(u, 0.0)
                        + w.get("bm25", 0.0) * bm25_norm.get(u, 0.0)
                        + w.get("quality", 0.0) * quality_norm.get(u, 0.0)
                        + w.get("recent", 0.0) * recent_norm.get(u, 0.0)
                    )

            if not final_scores:
                return [], 0

            ranked_urls = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
            top_urls = [url for url, score in ranked_urls[:config.TOP_RESULTS]]
            if not top_urls:
                return [], 0

            placeholders = ','.join('?' for _ in top_urls)
            cursor.execute(f"SELECT url, title, meta_description, snippet FROM documents WHERE url IN ({placeholders})", top_urls)
            doc_details = {row[0]: {'title': row[1], 'meta': row[2], 'snip': row[3]} for row in cursor.fetchall()}

            results = []
            for url, score in ranked_urls[:config.TOP_RESULTS]:
                details = doc_details.get(url)
                if details:
                    results.append({"url": url, **details})

            return results, len(final_scores)

# --- Global Index Instance ---
INDEX = SearchIndex(config.DB_FILE)

# -------------------------
# Utilities
# -------------------------
def normalize_url(u: str, base: Optional[str] = None) -> str:
    if not u:
        return ""
    u = u.strip()
    
    # Join with base if relative URL
    if base and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        try:
            u = urljoin(base, u)
        except Exception:
            return "" # Return empty on bad join

    # Scheme normalization and fixing
    u = re.sub(r'^(https?)[:/\\]+', r'\1://', u, flags=re.I)
    p = urlparse(u)
    if not p.scheme:
        u = "https://" + u
        p = urlparse(u)
    
    scheme = p.scheme.lower()
    if scheme not in ("http", "https"):
        return ""
    if scheme == "http":
        scheme = "https"

    # Netloc normalization (lowercase, remove default ports)
    netloc = p.netloc.lower()
    if (scheme == 'http' and netloc.endswith(':80')) or \
       (scheme == 'https' and netloc.endswith(':443')):
        netloc = netloc.rsplit(':', 1)[0]

    # Path normalization (remove common index files, ensure trailing slash)
    path = p.path or "/"
    if path.endswith(('/index.html', '/index.htm', '/index.php', '/default.html', '/default.htm')):
        path = path.rsplit('/', 1)[0] + '/'
    if not path:
        path = "/"
    
    # Query parameter normalization
    query_params = parse_qs(p.query)
    # Common tracking parameters to remove
    tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid', 'gclid', 'mc_cid', 'mc_eid', 'hsCtaTracking']
    
    filtered_params = {k: v for k, v in query_params.items() if k.lower() not in tracking_params}
    
    # Sort remaining parameters
    sorted_params = urlencode(sorted(filtered_params.items()), doseq=True)

    # Reconstruct the URL, removing fragment
    return urlunparse((scheme, netloc, path, p.params, sorted_params, ""))


def get_hostname(url: str) -> str:
    return urlparse(url).netloc


def extract_fields(base_url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else urlparse(base_url).netloc
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag.get("content").strip()

    blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["p", "li"], limit=8) if el.get_text(" ", strip=True)]
    snippet = " ".join(blocks)[:1200]

    canonical_tag = soup.find("link", rel="canonical")
    canonical = normalize_url(canonical_tag.get("href", ""), base=base_url) if canonical_tag else normalize_url(base_url)

    return {"title": title, "meta_description": meta_desc, "snippet": snippet, "canonical": canonical}


def force_search_and_index(url: str):
    """
    Force-fetches a URL, assesses its content, and indexes it immediately.
    This bypasses the regular crawl queue for immediate, high-priority indexing.
    """
    normalized_url = normalize_url(url)
    if not normalized_url:
        logger.error(f"[Force-Index] Invalid URL provided: {url}")
        return

    logger.info(f"[Force-Index] Starting immediate indexing for: {normalized_url}")

    try:
        # 1. Fetch the content
        fetched_url, scraped_fields, html_text = aggressive_fetch(normalized_url, extra_aggressive=True)

        if scraped_fields.get("redirect"):
            # If it's a redirect, we can index the redirect record and enqueue the target
            INDEX.index_document(normalized_url, scraped_fields)
            redirect_target = scraped_fields.get("redirect_target")
            if redirect_target:
                logger.info(f"[Force-Index] URL is a redirect to {redirect_target}. Enqueuing target.")
                cr_enqueue(redirect_target, high_priority=True, source='user')
            return

        # 2. Assess content quality
        soup = BeautifulSoup(html_text or "", "html.parser")
        quality_info = assess_content_quality(html_text or "", scraped_fields, soup)
        scraped_fields['quality'] = quality_info['quality']
        scraped_fields['content_flags'] = quality_info['flags']

        # 3. Determine the final URL to index (respecting canonical links)
        final_url_to_index = scraped_fields.get('canonical', fetched_url)

        # 4. Index the document
        INDEX.index_document(final_url_to_index, scraped_fields)
        logger.info(f"[Force-Index] Successfully indexed: {final_url_to_index} (Quality: {scraped_fields['quality']:.2f})")

    except Exception as e:
        logger.error(f"[Force-Index] Failed to index {normalized_url}: {repr(e)}")
        # Optionally, add a minimal error entry so the URL is at least known
        error_fields = {
            "title": f"Failed to fetch: {urlparse(normalized_url).netloc}",
            "meta_description": f"Error during force-index: {repr(e)}",
            "snippet": "",
            "canonical": normalized_url,
            "quality": 0.0,
            "is_placeholder": 1, # Mark as placeholder so it might be retried
        }
        try:
            INDEX.index_document(normalized_url, error_fields)
        except Exception as db_e:
            logger.error(f"[Force-Index] Could not even save an error placeholder for {normalized_url}: {repr(db_e)}")


def allowed_by_robots(host: str, user_agent: str, path: str = "/") -> bool:
    rp = ROBOTS.get(host)
    if rp is None:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.set_url(f"https://{host}/robots.txt")
            rp.read()
            ROBOTS[host] = rp
        except Exception:
            ROBOTS[host] = None
    if ROBOTS.get(host) is None:
        return True
    return ROBOTS[host].can_fetch(user_agent, path)

# -------------------------
# Aggressive fetch with redirect handling
# -------------------------
REDIRECT_STATUS_SET = {301, 302, 303, 307, 308}


def aggressive_fetch(url: str, extra_aggressive: bool = False) -> Tuple[str, Dict[str, Any], str]:
    url_norm = normalize_url(url)
    if not url_norm:
        raise Exception(f"Invalid URL passed to aggressive_fetch: {url!r}")

    cached = REDIRECT_CACHE.get(url_norm)
    if cached:
        target, status_code, ts = cached
        fields = {
            "title": urlparse(url_norm).netloc,
            "meta_description": "",
            "snippet": "",
            "canonical": target,
            "redirect": True,
            "redirect_target": target,
            "redirect_status": status_code,
            "cached_redirect": True,
        }
        return url_norm, fields, ""

    headers_list = [config.DEFAULT_HEADERS, config.BROWSER_HEADERS, config.GOOGLEBOT_HEADERS]
    if extra_aggressive:
        headers_list.append(config.GENERIC_HEADERS)
        
    last_exc = None

    for headers in headers_list:
        # If we are being extra aggressive, we don't check robots.txt
        if not extra_aggressive:
            host = get_hostname(url_norm)
            path = urlparse(url_norm).path or "/"
            user_agent = headers.get("User-Agent", "*")
            if not allowed_by_robots(host, user_agent, path):
                last_exc = Exception(f"Disallowed by robots.txt for user-agent: {user_agent}")
                continue

        try:
            r = SESSION.get(url_norm, headers=headers, timeout=config.REQUEST_TIMEOUT, allow_redirects=False)
            status = r.status_code

            if status in REDIRECT_STATUS_SET:
                loc = r.headers.get("location")
                if loc:
                    target = normalize_url(loc, base=url_norm)
                    if target:
                        REDIRECT_CACHE[url_norm] = (target, status, time.time())
                        fields = {
                            "title": urlparse(url_norm).netloc,
                            "meta_description": "",
                            "snippet": "",
                            "canonical": target,
                            "redirect": True,
                            "redirect_target": target,
                            "redirect_status": status,
                        }
                        return url_norm, fields, ""
                last_exc = Exception(f"Redirect ({status}) with no usable Location for {url_norm}")
                continue

            if 200 <= status < 300:
                content_type = r.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    r2 = SESSION.get(url_norm, headers=headers, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
                    r2.raise_for_status()
                    fetched_url = normalize_url(r2.url)
                    fields = extract_fields(fetched_url, r2.text)
                    return fetched_url, fields, r2.text
                else:
                    fetched_url = normalize_url(r.url)
                    fields = {
                        "title": urlparse(url_norm).netloc,
                        "meta_description": f"Non-HTML content-type: {content_type}",
                        "snippet": "",
                        "canonical": fetched_url,
                    }
                    return fetched_url, fields, ""
            else:
                last_exc = Exception(f"HTTP {status} for {url_norm}")
                continue

        except requests.RequestException as e:
            last_exc = e
            continue

    raise Exception(f"Failed to fetch {url_norm} -> {repr(last_exc)}")

# -------------------------
# Crawler: enqueue / dequeue
# -------------------------
def cr_enqueue(url: str, high_priority: bool = False, source: str = "system"):
    norm = normalize_url(url)
    if not norm:
        logger.warning(f"[Enqueue-SKIP] invalid url: {url}")
        return

    with ENQUEUED_LOCK:
        if norm in ENQUEUED or norm in VISITED:
            logger.debug(f"[Enqueue-SKIP] already enqueued/visited: {norm}")
            return
        ENQUEUED.add(norm)

    global _cr_counter
    with CR_QUEUE_COND:
        if len(_cr_heap) >= config.MAX_QUEUE_SIZE:
            logger.warning(f"[Enqueue-SKIP] queue full ({len(_cr_heap)}) - dropping: {norm}")
            return
        _cr_counter += 1
        
        is_new = not INDEX.document_exists(norm)

        if high_priority:
            pri = -1.0  # Highest priority
        elif is_new:
            # High-ish priority for new links, with some randomness
            pri = random.uniform(-0.5, 0.5)
        else:
            # Normal priority for existing links
            pri = random.random()

        heapq.heappush(_cr_heap, (pri, _cr_counter, norm))
        CR_QUEUE_COND.notify()

        if source == 'user':
            logger.info(f"[Enqueued by USER{'(HIGH)' if high_priority else ''}] {norm}")
        else:
            logger.info(f"[Enqueued{'(HIGH)' if high_priority else ''}{' (NEW)' if is_new and not high_priority else ''}] {norm}")


def cr_dequeue(timeout: Optional[float] = None) -> Optional[Tuple[float, str]]:
    with CR_QUEUE_COND:
        end = None if timeout is None else (time.time() + timeout)
        while True:
            if _cr_heap:
                pri, _, url = heapq.heappop(_cr_heap)
                return pri, url
            remaining = None if end is None else (end - time.time())
            if remaining is not None and remaining <= 0:
                return None
            CR_QUEUE_COND.wait(timeout=remaining)


def cr_size() -> int:
    with cr_queue_lock:
        return len(_cr_heap)

# -------------------------
# Crawler Worker (now accepts config args)
# -------------------------
crawl_total = 0
crawl_total_lock = threading.Lock()

LOW_QUALITY_ENQUEUE_PROB = 0.20  # follow more links from low-quality pages
PENALTY_SKIP_THRESHOLD = 0.10     # fewer pages are considered "too low quality" to follow


def crawler_worker(worker_id: int, max_retries: int = 3, stability: str = 'lenient', proxies: Optional[Dict[str, str]] = None, profile: Optional[Dict[str, Any]] = None):
    """Worker loop. Supports an optional `profile` dict which can override per-worker settings such as headers,
    per_host_delay and max pages per host. This allows multiple crawler "profiles" to run concurrently.
    """
    global crawl_total, SESSION

    # local configuration (profile overrides global defaults)
    local_per_host_delay = profile.get('per_host_delay', config.PER_HOST_DELAY) if profile else config.PER_HOST_DELAY
    local_max_per_host = profile.get('max_per_host', config.CRAWL_MAX_PER_HOST) if profile else config.CRAWL_MAX_PER_HOST
    local_headers = profile.get('headers') if profile and profile.get('headers') else config.DEFAULT_HEADERS

    if proxies:
        SESSION.proxies = proxies

    while not SHUTDOWN_EVENT.is_set():
        dequeued = cr_dequeue(timeout=1.0)
        if not dequeued:
            continue

        pri, candidate_raw = dequeued
        candidate_norm = normalize_url(candidate_raw)
        if not candidate_norm:
            logger.warning(f"[Worker-{worker_id}] Dequeued invalid URL, skipping: {candidate_raw}")
            continue

        logger.info(f"[Worker-{worker_id}] Dequeued: {candidate_norm} {'(PRIORITY)' if pri < 0 else ''}")

        with VISITED_LOCK:
            if candidate_norm in VISITED:
                logger.debug(f"[Worker-{worker_id}] Skip - already visited: {candidate_norm}")
                continue
            VISITED.add(candidate_norm)

        with ENQUEUED_LOCK:
            ENQUEUED.discard(candidate_norm)

        already_indexed_before = INDEX.document_exists(candidate_norm)



        is_placeholder = False
        if already_indexed_before:
            try:
                doc = INDEX.get_document(candidate_norm)
                if doc and int(doc.get('is_placeholder', 0)) == 1:
                    is_placeholder = True
            except Exception:
                is_placeholder = False

        if already_indexed_before and not is_placeholder:
            logger.debug(f"[Worker-{worker_id}] Skip - already indexed before worker started: {candidate_norm}")
            with crawl_total_lock:
                crawl_total += 1
            continue

        host = get_hostname(candidate_norm)
        if PER_HOST_COUNT.get(host, 0) > local_max_per_host:
            logger.warning(f"[Worker-{worker_id}] Skip - host reached max: {host}")
            continue

        path = urlparse(candidate_norm).path or "/"
        user_agent = local_headers.get("User-Agent", "*")
        if not allowed_by_robots(host, user_agent, path):
            logger.warning(f"[Worker-{worker_id}] Skip - disallowed by robots.txt: {candidate_norm}")
            continue

        host_lock = PER_HOST_LOCKS.setdefault(host, threading.Lock())
        with host_lock:
            last = PER_HOST_LAST.get(host, 0.0)
            now = time.time()
            wait = local_per_host_delay - (now - last)
            if wait > 0:
                time.sleep(wait)
            PER_HOST_LAST[host] = time.time()

        try:
            fetched_url, scraped_fields, html_text = aggressive_fetch(candidate_norm)

            if scraped_fields.get("redirect"):
                INDEX_QUEUE.put((candidate_norm, scraped_fields))
                rt = scraped_fields.get("redirect_target")
                if rt:
                    cr_enqueue(rt, high_priority=True)
                logger.info(f"[Worker-{worker_id}] [Crawled-Redirect] {candidate_norm} -> {scraped_fields.get('redirect_target')}")
                with crawl_total_lock:
                    crawl_total += 1
                PER_HOST_COUNT[host] = PER_HOST_COUNT.get(host, 0) + 1
                continue

            soup = BeautifulSoup(html_text or "", "html.parser")
            quality_info = assess_content_quality(html_text or "", scraped_fields, soup)
            scraped_fields['quality'] = quality_info['quality']
            scraped_fields['content_flags'] = quality_info['flags']

            final_url_to_index = scraped_fields.get('canonical', fetched_url)
            scraped_fields.pop('is_placeholder', None)
            INDEX_QUEUE.put((final_url_to_index, scraped_fields))
            logger.info(f"[Worker-{worker_id}] [Crawled] {final_url_to_index} (quality={scraped_fields['quality']:.2f})")
            with crawl_total_lock:
                crawl_total += 1
            PER_HOST_COUNT[host] = PER_HOST_COUNT.get(host, 0) + 1

            if not html_text:
                continue

            soup = BeautifulSoup(html_text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith(("mailto:", "javascript:")):
                    continue
                new_url = normalize_url(href, base=final_url_to_index)
                if new_url and get_hostname(new_url):
                    logger.debug(f"[Worker-{worker_id}] [Found] {new_url}  (from {final_url_to_index})")
                    with VISITED_LOCK:
                        already_seen = (new_url in VISITED)
                    with ENQUEUED_LOCK:
                        already_enq = (new_url in ENQUEUED)
                    if not already_seen and not already_enq:
                        q = scraped_fields.get('quality', 1.0)
                        if q < PENALTY_SKIP_THRESHOLD:
                            if random.random() < LOW_QUALITY_ENQUEUE_PROB:
                                cr_enqueue(new_url, high_priority=False)
                                logger.debug(f"[Worker-{worker_id}] [LowQ-Enqueued] {new_url} (source quality={q:.2f})")
                        else:
                            cr_enqueue(new_url, high_priority=False)
            
            # Token-based URL generation
            text_content = extract_text_from_soup(soup)
            tokens = tokenize(text_content)
            if len(tokens) >= 3:
                for i in range(len(tokens) - 2):
                    # Create 3-token combinations
                    tri_gram = "".join(tokens[i:i+3])
                    if len(tri_gram) > 5 and len(tri_gram) < 20: # Basic sanity check for domain length
                        generated_url = f"https://{tri_gram}.com"
                        cr_enqueue(generated_url, source='generated')


        except Exception as e:
            logger.error(f"[Worker-{worker_id}] Exception handling {candidate_norm} -> {repr(e)}")
            fields = {
                "title": urlparse(candidate_norm).netloc,
                "meta_description": f"Fetch attempts failed: {repr(e)}",
                "snippet": "",
                "canonical": candidate_norm,
            }
            try:
                INDEX_QUEUE.put((candidate_norm, fields))
                logger.warning(f"[Worker-{worker_id}] [Cached-Error] {candidate_norm}")
            except Exception as ie:
                logger.error(f"[Worker-{worker_id}] Failed to cache error entry for {candidate_norm} -> {repr(ie)}")
            with crawl_total_lock:
                crawl_total += 1
            PER_HOST_COUNT[host] = PER_HOST_COUNT.get(host, 0) + 1
            continue

# -------------------------
# Database Writer Thread
# -------------------------
def database_writer():
    batch = []
    last_write_time = time.time()
    while not SHUTDOWN_EVENT.is_set() or not INDEX_QUEUE.empty():
        try:
            # Use a short timeout to remain responsive
            item = INDEX_QUEUE.get(timeout=0.5)
            batch.append(item)
            
            # Write if batch is full or if some time has passed
            if len(batch) >= config.DB_BATCH_SIZE or (time.time() - last_write_time > config.DB_BATCH_TIMEOUT and batch):
                INDEX.index_batch(batch)
                logger.info(f"[DBWriter] Wrote batch of {len(batch)} items to DB.")
                batch = []
                last_write_time = time.time()
        except queue.Empty:
            # If queue is empty, write any remaining items
            if batch:
                INDEX.index_batch(batch)
                logger.info(f"[DBWriter] Wrote final batch of {len(batch)} items to DB.")
                batch = []
                last_write_time = time.time()
    
    # Final write after shutdown signal
    if batch:
        INDEX.index_batch(batch)
        logger.info(f"[DBWriter] Wrote final batch of {len(batch)} items to DB.")


# -------------------------
# Status Printer (console)
# -------------------------
def status_printer():
    while not SHUTDOWN_EVENT.is_set():
        with crawl_total_lock:
            total = crawl_total
        q_size = cr_size()
        high_waiting = cr_count_high_priority()
        
        logger.info("--- [Crawler Status] ---")
        logger.info(f"Queue: {q_size} | Total Fetched: {total} | Visited URLs: {len(VISITED)} | Enqueued: {len(ENQUEUED)} | Redirects cached: {len(REDIRECT_CACHE)} | High-priority waiting: {high_waiting}")
        
        top_hosts = sorted(PER_HOST_COUNT.items(), key=lambda i: i[1], reverse=True)[:5]
        if top_hosts:
            logger.info("Top hosts:")
            for h, c in top_hosts:
                logger.info(f"  {h:<30} | Count: {c}")
        logger.info("-" * 60)
        time.sleep(10.0)

# -------------------------
# Search UI & Endpoints
# -------------------------
HTML_UI = """<!doctype html>
<html lang="en-GB">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A.S.S Search v2.8</title>
<style>
body{font-family:system-ui,-apple-system,Roboto,Arial;margin:1.2rem;color:#f0f0f0;background-color:#1a1a1a;}
form{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem;align-items:center}
input[type=text]{flex:1 1 420px;padding:.5rem .7rem;border:1px solid #555;border-radius:8px;font-size:1rem;background-color:#2a2a2a;color:#f0f0f0;}
select{padding:.4rem .6rem;border-radius:8px;border:1px solid #555;background-color:#2a2a2a;color:#f0f0f0;}
button{padding:.5rem 1rem;border-radius:8px;background:#2b6cb0;color:#fff;border:0;cursor:pointer}
.result{padding:.8rem 0;border-bottom:1px solid #333}
.title{font-size:1.2rem;font-weight:600;} .title a{color:#8ab4f8;text-decoration:none;} .title a:hover{text-decoration:underline}
.snip{color:#ccc;margin-top:.2rem}
.meta{color:#888;font-size:.9rem;margin-top:.2rem;word-break:break-all}
</style></head>
<body>
<h1>A.S.S Search v2.8</h1>
<form method="get" action="/">
  <input type="text" name="q" value="{{query|e}}" placeholder="Search..." autocomplete="off" autocorrect="off" autocapitalize="off" autofocus>
  <label for="alg">Algorithm:</label>
  <select id="alg" name="alg">
    <option value="merged_advanced" {% if alg=='merged_advanced' %}selected{% endif %}>Merged (Advanced)</option>
    <option value="merged" {% if alg=='merged' %}selected{% endif %}>Merged (auto)</option>
    <option value="merged_simple" {% if alg=='merged_simple' %}selected{% endif %}>Merged (Simple)</option>
    <option value="tfidf" {% if alg=='tfidf' %}selected{% endif %}>TF-IDF</option>
    <option value="bm25" {% if alg=='bm25' %}selected{% endif %}>BM25</option>
    <option value="quality" {% if alg=='quality' %}selected{% endif %}>Quality</option>
    <option value="recent" {% if alg=='recent' %}selected{% endif %}>Recent</option>
  </select>
  <button type="submit">Search</button>
</form>
<p>Results: {{total_results}}</p>
{% for item in results %}
  <div class="result">
    <div class="title"><a href="{{ item['url'] }}" target="_blank" rel="noopener noreferrer">{{ item['title'] }}</a></div>
    {% if item['snip'] %}<div class="snip">{{ item['snip']|e }}</div>{% endif %}
    <div class="meta">{{ item['url'] }}</div>
  </div>
{% endfor %}
</body>
</html>"""

@app.route("/", methods=["GET"])
def ui_search():
    q = (request.args.get("q", "") or "").strip()
    alg = (request.args.get("alg", "merged_advanced") or "merged_advanced").lower()
    if q:
        results, total = INDEX.search(q, algorithm=alg)
        # Token-based URL generation from search query
        tokens = tokenize(q)
        if tokens:
            random_token = random.choice(tokens)
            generated_url = f"https://{random_token}.com"
            cr_enqueue(generated_url, source='generated')
    else:
        results, total = [], 0
    return render_template_string(HTML_UI, query=q, results=results, total_results=total, alg=alg)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/api/search", methods=["GET"])
def api_search():
    q = (request.args.get("q", "") or "").strip()
    alg = (request.args.get("alg", "merged_advanced") or "merged_advanced").lower()
    if not q:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    
    results, total = INDEX.search(q, algorithm=alg)
    return jsonify({
        "query": q,
        "algorithm": alg,
        "total_results": total,
        "results": results
    })


# API endpoint to add a new crawl target dynamically
@app.route("/add_target", methods=["POST"])
def add_target():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"status": "error", "message": "Missing 'url' in request body."}), 400

    url_to_add = data["url"]
    normalized_url = normalize_url(url_to_add)
    if not normalized_url:
        return jsonify({"status": "error", "message": "Invalid URL provided."}), 400

    # Run force-indexing in a separate thread so the API can respond immediately
    force_thread = ForceIndexThread(normalized_url)
    force_thread.start()
    
    logger.info(f"[API] Queued force-indexing for target: {normalized_url}")
    return jsonify({
        "status": "ok",
        "message": f"URL '{normalized_url}' has been queued for immediate indexing."
    }), 202  # 202 Accepted

# Optional API: dump redirect cache (for debugging)
@app.route("/_redirects", methods=["GET"])
def dump_redirects():
    return jsonify({k: {"target": v[0], "status": v[1], "ts": v[2]} for k, v in REDIRECT_CACHE.items()})

# -------------------------
# Force Index Thread
# -------------------------
class ForceIndexThread(threading.Thread):
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        # Force this thread into the main thread group so it does proper forks and searches
        self.daemon = False

    def run(self):
        try:
            force_search_and_index(self.url)
        except Exception as e:
            logger.error(f"[ForceIndexThread] Unhandled exception for {self.url}: {repr(e)}")

from exit_manager import ExitManager
# -------------------------
# Boot & Seeding
# -------------------------
def _signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    SHUTDOWN_EVENT.set()


def main():
    global SESSION, INDEX, _cr_heap, VISITED, ENQUEUED, REDIRECT_CACHE, PER_HOST_COUNT

    parser = argparse.ArgumentParser(description='A.S.S Search v2.8 — crawler + merged ranking algorithms')
    parser.add_argument('--workers', '-w', type=int, default=config.CRAWLER_WORKER_COUNT, help='Number of crawler worker threads')
    parser.add_argument('--no-crawl', action='store_true', help='Start server without launching crawler workers')
    parser.add_argument('--seeds', type=str, default=','.join(config.DEMO_SEEDS), help='Comma-separated seed URLs to enqueue at startup')
    parser.add_argument('--db-file', type=str, default=config.DB_FILE, help='SQLite DB file for persistence')
    parser.add_argument('--exit-db-file', type=str, default='exit.db', help='SQLite DB file for exit state')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries for fetching URLs')
    parser.add_argument('--per-host-delay', type=float, default=config.PER_HOST_DELAY, help='Delay between requests to same host')
    parser.add_argument('--max-per-host', type=int, default=config.CRAWL_MAX_PER_HOST, help='Maximum pages to fetch per host')
    parser.add_argument('--max-queue', type=int, default=config.MAX_QUEUE_SIZE, help='Maximum queue size')
    parser.add_argument('--stability', choices=['lenient', 'strict'], default='lenient', help='Stability mode for fetch retry behaviour')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--clear-db', action='store_true', help='Remove the DB file at startup (clear index)')
    parser.add_argument('--proxy', type=str, default=None, help='Optional proxy (format: http://host:port)')
    parser.add_argument('--add', type=str, help='Add a new URL to the crawl queue and exit')
    parser.add_argument('--add-seeds', type=str, help='Add a comma-separated list of new seed URLs to the crawl queue and exit')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    exit_manager = ExitManager(args.exit_db_file)
    
    if not args.clear_db:
        # Load state from exit.db
        state = exit_manager.load_state()
        if state:
            _cr_heap.extend(state.get("crawl_queue", []))
            heapq.heapify(_cr_heap)
            VISITED.update(state.get("visited", []))
            ENQUEUED.update(state.get("enqueued", []))
            REDIRECT_CACHE.update(state.get("redirect_cache", {}))
            PER_HOST_COUNT.update(state.get("per_host_count", {}))

    if args.add:
        db_path = args.db_file or config.DB_FILE
        if args.clear_db and Path(db_path).exists():
            try:
                Path(db_path).unlink()
                logger.info(f"Cleared DB file: {db_path}")
            except Exception as e:
                logger.error(f"Failed to remove DB file: {e}")

        INDEX = SearchIndex(db_path)
        url_to_add = normalize_url(args.add)
        if not url_to_add:
            logger.error("Invalid URL provided.")
            exit(1)
        try:
            force_search_and_index(url_to_add)
            logger.info(f"Successfully force-indexed '{url_to_add}'.")
        except Exception as e:
            logger.error(f"Failed to force-index URL: {e}")
            exit(1)
        exit(0)

    if args.add_seeds:
        db_path = args.db_file or config.DB_FILE
        if args.clear_db and Path(db_path).exists():
            try:
                Path(db_path).unlink()
                logger.info(f"Cleared DB file: {db_path}")
            except Exception as e:
                logger.error(f"Failed to remove DB file: {e}")

        INDEX = SearchIndex(db_path)
        urls_to_add = [s.strip() for s in args.add_seeds.split(',') if s.strip()]
        for url in urls_to_add:
            normalized_url = normalize_url(url)
            if not normalized_url:
                logger.error(f"Invalid URL provided: {url}")
                continue
            try:
                force_search_and_index(normalized_url)
                logger.info(f"Successfully force-indexed '{normalized_url}'.")
            except Exception as e:
                logger.error(f"Failed to force-index URL: {e}")
        exit(0)

    config.CRAWLER_WORKER_COUNT = max(1, args.workers)
    config.PER_HOST_DELAY = float(args.per_host_delay)
    config.CRAWL_MAX_PER_HOST = int(args.max_per_host)
    config.MAX_QUEUE_SIZE = int(args.max_queue)
    config.DB_FILE = args.db_file

    if args.clear_db and Path(config.DB_FILE).exists():
        try:
            Path(config.DB_FILE).unlink()
            logger.info(f"Cleared DB file: {config.DB_FILE}")
        except Exception as e:
            logger.error(f"Failed to remove DB file: {e}")

    # recreate index object with chosen DB
    INDEX = SearchIndex(config.DB_FILE)

    adapter = requests.adapters.HTTPAdapter(pool_connections=config.REQUEST_POOL, pool_maxsize=config.REQUEST_POOL, max_retries=0)
    SESSION.mount("http://", adapter)
    SESSION.mount("https://", adapter)
    SESSION.max_redirects = 8

    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    for s in seeds:
        n = normalize_url(s)
        if n:
            cr_enqueue(n)
            logger.info(f"[Seed] {n}")

    if not args.no_crawl:
        # Start the database writer thread
        db_writer_thread = threading.Thread(target=database_writer)
        db_writer_thread.start()

        crawler_threads = []
        for i in range(config.CRAWLER_WORKER_COUNT):
            thread = threading.Thread(target=crawler_worker, args=(i, args.max_retries, args.stability, proxies))
            thread.start()
            crawler_threads.append(thread)
        
        status_thread = threading.Thread(target=status_printer, daemon=True)
        status_thread.start()

    logger.info(f"A.S.S Search v2.8 starting — workers={config.CRAWLER_WORKER_COUNT}, persistence={config.DB_FILE}")

    # Run Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host=config.HOST, port=config.PORT, debug=False))
    flask_thread.daemon = True
    flask_thread.start()

    try:
        # Keep the main thread alive, waiting for a shutdown signal
        while not SHUTDOWN_EVENT.is_set():
            time.sleep(1)  # Main loop can sleep, threads do the work
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        SHUTDOWN_EVENT.set()
    finally:
        logger.info("Shutdown initiated, waiting for threads to complete...")
        exit_manager.save_state(
            crawl_queue=_cr_heap,
            visited=VISITED,
            enqueued=ENQUEUED,
            redirect_cache=REDIRECT_CACHE,
            per_host_count=PER_HOST_COUNT
        )
        
        if not args.no_crawl:
            # Wait for the DB writer to finish processing its queue
            if 'db_writer_thread' in locals() and db_writer_thread.is_alive():
                db_writer_thread.join()
                logger.info("DB writer thread finished.")
            
            # Wait for crawler threads to finish their current tasks
            if 'crawler_threads' in locals():
                for thread in crawler_threads:
                    if thread.is_alive():
                        thread.join()
                        logger.info("Crawler thread finished.")

        logger.info("Shutdown complete.")




if __name__ == "__main__":
    main()
