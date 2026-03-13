"""
dashboard/ai_utils.py

Claude API を使ったチャート解説モジュール。
- ANTHROPIC_API_KEY が .env に設定されている場合のみ有効
- claude-haiku-4-5-20251001 で 100〜150字の日本語分析
- 5分キャッシュ（同じデータなら再利用）
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st

# .env 読み込み
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def is_ai_available() -> bool:
    """ANTHROPIC_API_KEY が設定されている場合のみ True を返す。"""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def prepare_chart_inputs(df, indicators_data: dict) -> dict:
    """
    DataFrameと indicators_data から analyze_chart() の引数を準備する。
    DataFrame は @st.cache_data に直接渡せないため tuple/str に変換する。

    Returns:
        {
          "df_tail_hash":       str,
          "close_prices":       tuple,
          "high_prices":        tuple,
          "low_prices":         tuple,
          "volume_prices":      tuple,
          "indicators_summary": str,
        }
    """
    n = min(20, len(df))
    tail = df.iloc[-n:]

    close  = tuple(round(float(v), 5) for v in tail["close"])
    high   = tuple(round(float(v), 5) for v in tail["high"])
    low    = tuple(round(float(v), 5) for v in tail["low"])
    volume = tuple(int(v) for v in tail["volume"])

    hash_src     = ",".join(f"{v:.5f}" for v in close[-10:])
    df_tail_hash = hashlib.md5(hash_src.encode()).hexdigest()[:12]

    parts = []
    for key, ind in indicators_data.items():
        data = ind.get("data") or []
        if not data:
            continue
        last_val = data[-1].get("value")
        if last_val is None:
            continue
        if ind.get("type") == "overlay":
            parts.append(f"{key}={last_val:.3f}")
        elif key == "RSI":
            parts.append(f"RSI={last_val:.1f}")
        if len(parts) >= 5:
            break

    return {
        "df_tail_hash":       df_tail_hash,
        "close_prices":       close,
        "high_prices":        high,
        "low_prices":         low,
        "volume_prices":      volume,
        "indicators_summary": ", ".join(parts),
    }


def _build_prompt(
    symbol: str,
    timeframe: str,
    close_prices: tuple,
    high_prices: tuple,
    low_prices: tuple,
    indicators_summary: str,
) -> str:
    recent_close = close_prices[-10:]
    price_str    = ", ".join(f"{p:.3f}" for p in recent_close)
    h_max        = max(high_prices[-10:])
    l_min        = min(low_prices[-10:])

    return (
        f"FXトレーダー向けに {symbol} の {timeframe} 足チャートを簡潔に分析してください。\n\n"
        f"直近10本の終値: {price_str}\n"
        f"直近高値: {h_max:.3f} / 安値: {l_min:.3f}\n"
        f"インジケーター: {indicators_summary or 'なし'}\n\n"
        "100〜150字の日本語で、現在のトレンド方向・強さ・注目すべき価格水準を\n"
        "簡潔にまとめてください。接頭辞（「AI分析:」等）は不要です。"
    )


@st.cache_data(ttl=300)
def analyze_chart(
    symbol: str,
    timeframe: str,
    df_tail_hash: str,
    close_prices: tuple,
    high_prices: tuple,
    low_prices: tuple,
    volume_prices: tuple,
    indicators_summary: str,
) -> str | None:
    """
    Claude Haiku でチャートを分析し日本語テキストを返す。
    エラー時・API未設定時は None を返す。
    """
    if not is_ai_available():
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        prompt = _build_prompt(
            symbol, timeframe, close_prices, high_prices, low_prices, indicators_summary
        )
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        return None
