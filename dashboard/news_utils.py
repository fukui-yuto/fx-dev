"""
dashboard/news_utils.py

USD/JPY 関連ニュースの取得とセンチメント分析。
- RSSフィード並列取得（ThreadPoolExecutor）
- 翻訳・分析も並列処理（初回でも高速）
- 結果は5分間キャッシュ
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import feedparser
import streamlit as st

# ============================================================
# ニュースソース
# ============================================================

NEWS_FEEDS = [
    ("ロイター",       "https://feeds.reuters.com/reuters/JPBusinessNews"),
    ("ロイター(国際)", "https://feeds.reuters.com/reuters/businessNews"),
    ("ForexLive",      "https://www.forexlive.com/feed/"),
    ("FXStreet",       "https://www.fxstreet.com/rss/news"),
    ("Yahoo Finance",  "https://finance.yahoo.com/rss/headline?s=USDJPY%3DX"),
]

# USD/JPY 関連フィルタリングキーワード（日英）
_RELEVANT = [
    "dollar", "yen", "usd", "jpy", "usdjpy", "usd/jpy",
    "fed", "federal reserve", "boj", "bank of japan",
    "interest rate", "inflation", "cpi", "nfp", "payroll",
    "japan", "tariff", "trade", "intervention", "treasury",
    "monetary policy", "yield", "forex", "currency",
    "gdp", "employment", "jobs", "recession",
    "ドル", "円", "為替", "ドル円", "日銀", "連邦準備",
    "金利", "インフレ", "利上げ", "利下げ", "雇用",
    "貿易", "介入", "景気", "経済", "物価", "政策",
]

# ============================================================
# キーワードベースのセンチメント
# ============================================================

_BULLISH: dict[str, int] = {
    "rate hike": 3, "hawkish": 2, "tightening": 2,
    "strong jobs": 2, "beats expectations": 2,
    "dollar rally": 2, "dollar rises": 2, "dollar surge": 2,
    "yen falls": 2, "yen drops": 2, "weak yen": 2,
    "boj maintain": 2, "easy monetary": 1, "risk-on": 1,
    "利上げ": 3, "ドル高": 2, "円安": 2, "ドル買い": 2,
    "日銀緩和": 2, "緩和維持": 2, "金利上昇": 2,
    "強い雇用": 2, "経済成長": 1, "リスクオン": 1,
}

_BEARISH: dict[str, int] = {
    "rate cut": 3, "dovish": 2, "easing": 1,
    "weak jobs": 2, "misses expectations": 2,
    "dollar falls": 2, "dollar weakens": 2,
    "yen rises": 2, "strong yen": 2, "yen rally": 2,
    "boj hike": 3, "boj raise": 3, "recession": 2, "risk-off": 1,
    "利下げ": 3, "ドル安": 2, "円高": 2, "円買い": 2,
    "日銀利上げ": 3, "金融引き締め": 2, "景気後退": 2,
    "弱い雇用": 2, "リスクオフ": 1, "ドル売り": 2,
}

_REASON_JA: dict[str, str] = {
    "rate hike": "利上げ観測", "hawkish": "タカ派姿勢",
    "tightening": "金融引き締め", "dollar rally": "ドル上昇",
    "yen falls": "円安進行", "boj maintain": "日銀緩和維持",
    "rate cut": "利下げ観測", "dovish": "ハト派姿勢",
    "dollar falls": "ドル下落", "yen rises": "円高進行",
    "boj hike": "日銀利上げ", "recession": "景気後退懸念",
    "risk-on": "リスクオン", "risk-off": "リスクオフ",
    "利上げ": "利上げ観測", "円安": "円安進行",
    "利下げ": "利下げ観測", "円高": "円高進行",
    "日銀利上げ": "日銀利上げ観測",
}


def _keyword_sentiment(title: str, summary: str) -> dict:
    text = (title + " " + summary).lower()
    bull_hits = [(kw, w) for kw, w in _BULLISH.items() if kw.lower() in text]
    bear_hits = [(kw, w) for kw, w in _BEARISH.items() if kw.lower() in text]
    bull = sum(w for _, w in bull_hits)
    bear = sum(w for _, w in bear_hits)
    net  = bull - bear

    if net >= 2:
        direction, hits, summary_text = "up",      bull_hits, "ドル円に上昇示唆"
    elif net <= -2:
        direction, hits, summary_text = "down",    bear_hits, "ドル円に下落示唆"
    else:
        direction, hits, summary_text = "neutral", [],        "USD/JPYへの影響は限定的"

    impact  = min(5, max(1, abs(net)))
    reasons = [_REASON_JA.get(kw, kw) for kw, _ in hits[:2]]
    reason  = "、".join(reasons) if reasons else "明確なシグナルなし"

    return {"direction": direction, "impact": impact,
            "summary": summary_text, "reason": reason}


# ============================================================
# 日本語判定・翻訳
# ============================================================

def _is_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u9FFF]", text))


def _translate_ja(text: str) -> str:
    """Google翻訳で英→日（無料・API不要）。失敗時は原文を返す。"""
    if not text or _is_japanese(text):
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="ja").translate(text) or text
    except Exception:
        return text


# ============================================================
# RSS 1件取得（並列用）
# ============================================================

def _fetch_one_feed(source_name: str, url: str) -> list[dict]:
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "FXDashboard/1.0"})
        results = []
        for entry in feed.entries[:20]:
            title         = getattr(entry, "title", "") or ""
            summary_raw   = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            link          = getattr(entry, "link", "") or ""
            summary_clean = re.sub(r"<[^>]+>", "", summary_raw)[:400]

            if not any(kw.lower() in (title + " " + summary_clean).lower() for kw in _RELEVANT):
                continue

            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            results.append({
                "source": source_name, "title": title,
                "summary": summary_clean, "link": link, "published": published,
            })
        return results
    except Exception:
        return []


# ============================================================
# メイン：並列取得＆分析（5分キャッシュ）
# ============================================================

@st.cache_data(ttl=300)
def fetch_and_analyze_news() -> list[dict]:
    """
    RSSの並列取得 → 翻訳・センチメント分析を並列実行。
    全体をまとめてキャッシュするため、2回目以降は瞬時に表示。
    """
    # Step 1: 全フィードを並列取得
    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(NEWS_FEEDS)) as ex:
        futures = {ex.submit(_fetch_one_feed, name, url): name for name, url in NEWS_FEEDS}
        for f in as_completed(futures):
            raw.extend(f.result())

    # 重複除去 → 新しい順
    seen:   set[str]   = set()
    unique: list[dict] = []
    for a in raw:
        key = a["title"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(a)
    unique.sort(
        key=lambda x: x["published"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    unique = unique[:30]

    # Step 2: 翻訳＋センチメントを並列処理
    def _process(art: dict) -> dict:
        analysis           = _keyword_sentiment(art["title"], art["summary"])
        analysis["title_ja"] = _translate_ja(art["title"])
        return {**art, "analysis": analysis}

    with ThreadPoolExecutor(max_workers=10) as ex:
        processed = list(ex.map(_process, unique))

    return processed


# ============================================================
# ユーティリティ
# ============================================================

def time_ago(published: datetime | None) -> str:
    if not published:
        return "日時不明"
    diff = (datetime.now(timezone.utc) - published).total_seconds()
    if diff < 60:
        return "たった今"
    if diff < 3600:
        return f"{int(diff / 60)}分前"
    if diff < 86400:
        return f"{int(diff / 3600)}時間前"
    return f"{int(diff / 86400)}日前"
