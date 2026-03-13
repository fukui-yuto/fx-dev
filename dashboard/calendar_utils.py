"""
dashboard/calendar_utils.py

ForexFactory非公式APIから経済指標カレンダーを取得するモジュール。
- 今週・来週のイベントを取得
- 5分キャッシュ
- 高インパクトイベントをLightweightChartsマーカー形式に変換
"""

from __future__ import annotations

import re as _re
from datetime import datetime, timezone, timedelta

import streamlit as st

_FF_URLS = {
    "thisweek": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "nextweek":  "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
}

_IMPACT_COLORS = {
    "High":   "#ef5350",
    "Medium": "#ff9800",
    "Low":    "#9e9e9e",
}

_IMPACT_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Non-Economic": 3, "Holiday": 4}

# 低いほど良い逆指標（actual < forecast → 強気）
_INVERTED_INDICATORS: frozenset[str] = frozenset({
    "Unemployment Rate",
    "Unemployment Change",
    "Claimant Count Change",
    "Initial Jobless Claims",
    "Continuing Jobless Claims",
    "Jobless Claims 4-Week Avg",
})

# 強弱判定の表示設定
_DIRECTION_COLORS = {
    "bullish": "#26a69a",
    "bearish": "#ef5350",
    "neutral": "#9e9e9e",
}
_DIRECTION_SHAPES = {
    "bullish": "arrowUp",
    "bearish": "arrowDown",
    "neutral": "circle",
}

_VALUE_MULTIPLIERS: dict[str, float] = {
    "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000
}


def _parse_value(raw: str) -> float | None:
    """経済指標の文字列値を float に変換する。
    例: "185K" → 185000.0 / "4.2%" → 4.2 / "-0.3M" → -300000.0
    パース不能な場合は None を返す。
    """
    if not raw or raw.strip() in ("-", "", "N/A"):
        return None
    s = raw.strip().replace(",", "").rstrip("%")
    suffix = s[-1].upper() if s and s[-1].upper() in _VALUE_MULTIPLIERS else ""
    if suffix:
        s = s[:-1]
    try:
        return float(s) * _VALUE_MULTIPLIERS.get(suffix, 1.0)
    except (ValueError, TypeError):
        return None


def _compute_direction(title: str, actual_str: str, forecast_str: str) -> str:
    """actual vs forecast を比較して強弱を判定する。
    戻り値: "bullish" | "bearish" | "neutral"
    """
    actual   = _parse_value(actual_str)
    forecast = _parse_value(forecast_str)
    if actual is None or forecast is None:
        return "neutral"
    is_inverted = any(kw.lower() in title.lower() for kw in _INVERTED_INDICATORS)
    if abs(actual - forecast) < max(abs(forecast) * 0.001, 1e-9):
        return "neutral"
    if actual > forecast:
        return "bearish" if is_inverted else "bullish"
    return "bullish" if is_inverted else "bearish"


# ============================================================
# データ取得
# ============================================================

def _fetch_raw(url: str) -> tuple[list, str]:
    """URLからJSONを取得する。戻り値: (data, error_kind)
    error_kind: "" | "ratelimit" | "error"
    """
    import urllib.request
    import urllib.error
    import json as _json
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.forexfactory.com/",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            return _json.loads(resp.read().decode("utf-8")), ""
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return [], "ratelimit"
        return [], "error"
    except Exception:
        return [], "error"


def _parse_raw(raw: list) -> list[dict]:
    """生JSONリストをイベント辞書リストに変換する。"""
    events = []
    for ev in raw:
        date_str = ev.get("date", "")
        if not date_str:
            continue
        try:
            dt_utc = datetime.fromisoformat(date_str).astimezone(timezone.utc)
        except Exception:
            continue
        impact_raw   = ev.get("impact", "Low") or "Low"
        actual_val   = ev.get("actual")   or "-"
        forecast_val = ev.get("forecast") or "-"
        events.append({
            "title":     ev.get("title", ""),
            "country":   (ev.get("country") or "").upper(),
            "impact":    impact_raw.capitalize(),
            "forecast":  forecast_val,
            "previous":  ev.get("previous") or "-",
            "actual":    actual_val,
            "dt_utc":    dt_utc,
            "direction": _compute_direction(ev.get("title", ""), actual_val, forecast_val),
        })
    events.sort(key=lambda e: e["dt_utc"])
    return events


def _do_fetch() -> tuple[list[dict], str]:
    """今週＋来週を並列取得して結合する。戻り値: (data, error_kind)"""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_this = ex.submit(_fetch_raw, _FF_URLS["thisweek"])
        f_next = ex.submit(_fetch_raw, _FF_URLS["nextweek"])
        raw_this, err_this = f_this.result()
        raw_next, err_next = f_next.result()

    error_kind = "ratelimit" if "ratelimit" in (err_this, err_next) else \
                 "error"     if "error"     in (err_this, err_next) else ""

    all_events = _parse_raw(raw_this) + _parse_raw(raw_next)
    seen: set[tuple] = set()
    merged = []
    for ev in all_events:
        key = (ev["dt_utc"], ev["title"], ev["country"])
        if key not in seen:
            seen.add(key)
            merged.append(ev)
    merged.sort(key=lambda e: e["dt_utc"])
    return merged, error_kind


class _CalendarCache:
    """成功・失敗・レートリミットで TTL を切り替えるシングルトンキャッシュ。
    成功: 10分 / 一般エラー: 60秒 / 429レートリミット: 10分
    """
    _TTL_OK        = 600   # 10分
    _TTL_FAIL      = 60    # 1分
    _TTL_RATELIMIT = 600   # 10分（429は長めに待つ）

    def __init__(self) -> None:
        self._data: list[dict] = []
        self._fetched_at: float = 0.0
        self._error: str = ""   # "" | "ratelimit" | "error"

    def get(self) -> tuple[list[dict], str]:
        """(data, error_kind) を返す。"""
        import time
        now = time.monotonic()
        if self._error == "ratelimit":
            ttl = self._TTL_RATELIMIT
        elif self._data:
            ttl = self._TTL_OK
        else:
            ttl = self._TTL_FAIL

        if now - self._fetched_at >= ttl:
            self._fetched_at = now
            fresh, err = _do_fetch()
            self._error = err
            if fresh:
                self._data = fresh  # 成功時のみ上書き

        return self._data, self._error

    def invalidate(self) -> None:
        """手動更新ボタン用: 429 以外なら即座に再取得する。"""
        import time
        # レートリミット中は残り待機時間を短縮（30秒残す）
        if self._error == "ratelimit":
            self._fetched_at = time.monotonic() - self._TTL_RATELIMIT + 30
        else:
            self._fetched_at = 0.0


@st.cache_resource
def _calendar_cache() -> _CalendarCache:
    return _CalendarCache()


def fetch_both_weeks() -> list[dict]:
    """後方互換インターフェース。データのみ返す（エラー情報が不要な場合）。"""
    data, _ = _calendar_cache().get()
    return data


def fetch_both_weeks_with_status() -> tuple[list[dict], str]:
    """(data, error_kind) を返す。カレンダーページ用。"""
    return _calendar_cache().get()


def fetch_calendar_events(week: str = "thisweek") -> list[dict]:
    """後方互換用。"""
    all_ev = fetch_both_weeks()
    if week == "thisweek":
        cutoff = datetime.now(timezone.utc) + timedelta(days=7)
        return [e for e in all_ev if e["dt_utc"] <= cutoff]
    return all_ev


# ============================================================
# チャートマーカー変換
# ============================================================

def events_to_markers(
    events: list[dict],
    jst_offset: int,
    impacts: list[str] | None = None,
) -> list[dict]:
    """
    イベントリストをLightweightCharts setMarkers() 形式に変換する。

    Parameters:
        events:     fetch_both_weeks() の返却値
        jst_offset: chart_utils.JST_OFFSET (= 6 * 3600)
        impacts:    フィルタするインパクト（None → ["High"] のみ）
    """
    if impacts is None:
        impacts = ["High"]

    markers = []
    for ev in events:
        if ev["impact"] not in impacts:
            continue
        # 実績値がないイベントはチャートに表示しない
        if ev.get("actual", "-") == "-":
            continue
        t         = int(ev["dt_utc"].timestamp()) + jst_offset
        direction = ev.get("direction", "neutral")
        color     = _DIRECTION_COLORS[direction]
        shape     = _DIRECTION_SHAPES[direction]
        label     = f"{ev['country']} {ev['title'][:20]}"
        markers.append({
            "time":     t,
            "position": "aboveBar",
            "color":    color,
            "shape":    shape,
            "text":     label,
        })
    return markers


def get_high_impact_for_symbols(
    symbols: list[str],
    jst_offset: int,
) -> list[dict]:
    """
    通貨ペアリスト（例: ["USDJPY"]）に関連する通貨の
    High インパクトイベントのマーカーリストを返す。
    USDJPY → ["USD", "JPY"] に分解する。
    """
    currencies: set[str] = set()
    for sym in symbols:
        # 6文字の通貨ペアを3文字ずつ分割
        sym = sym.upper().replace("/", "")
        if len(sym) >= 6:
            currencies.add(sym[:3])
            currencies.add(sym[3:6])

    try:
        all_events = fetch_both_weeks()
    except Exception:
        return []

    filtered = [ev for ev in all_events if ev["country"] in currencies]
    return events_to_markers(filtered, jst_offset, impacts=["High", "Medium"])
