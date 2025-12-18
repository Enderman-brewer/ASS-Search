#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for A.S.S Search
"""

# --- Server ---
HOST = "0.0.0.0"
PORT = 8090

# --- Database ---
DB_FILE = "ass_search.db"
DB_BATCH_SIZE = 20
DB_BATCH_TIMEOUT = 5.0

# --- Crawler ---
CRAWLER_WORKER_COUNT = 20
REQUEST_TIMEOUT = 7.0
REQUEST_POOL = 40
PER_HOST_DELAY = 0.001
CRAWL_MAX_PER_HOST = 20000
MAX_QUEUE_SIZE = 1000000000

# --- Ranking ---
NEEDS_OVERRIDE_THRESHOLD = 30
TOP_RESULTS = 100
RANKING_WEIGHTS = {
    "tfidf": 0.40,
    "bm25":  0.30,
    "quality": 0.20,
    "recent": 0.10,
}

RANKING_WEIGHTS_SIMPLE = {
    "tfidf": 0.35,
    "bm25":  0.25,
    "quality": 0.15,
    "recent": 0.10,
    "url_len": 0.15,
}

RANKING_WEIGHTS_ADVANCED = {
    "tfidf": 0.30,
    "bm25":  0.30,
    "quality": 0.20,
    "recent": 0.10,
    "url_len": 0.10,
}

# --- Content Quality ---
MIN_USEFUL_WORDS = 60
HARD_SKIP_WORDS = 10
SPAM_KEYWORDS = ["adsbygoogle", "sponsored", "advertisement", "buy now", "click here", "subscribe", "promo", "shop now"]

# --- Saved Pages ---
SAVE_DIR = "saved_pages"
SAVED_PATTERNS = [r"youtube\.com/watch\?"]

# --- User Agents ---
DEFAULT_HEADERS = {"User-Agent": "ASS-Search/2.7 (+http://localhost)"}
GOOGLEBOT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36"}
GENERIC_HEADERS = {"User-Agent": "Mozilla/5.0"}

# --- Seeds ---
DEMO_SEEDS = ["https://example.com", "https://github.com", "https://duckduckgo.com", "https://google.com", "https://bing.com", "https://microsoft.com/", "https://jackbox.tv/", "https://wikipedia.org/", "https://shop.hasbro.com/en-us"]



if __name__ == "__main__":
    print("Please import to use config")
elif __name__ != "config":
    print("Use original name")