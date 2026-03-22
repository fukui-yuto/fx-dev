"""
dashboard/signal_score.py

エントリースコアリングと通貨強弱メーター。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# エントリースコアリング
# ============================================================

def calc_entry_score(df: pd.DataFrame) -> dict:
    """
    複数インジケーターを組み合わせてエントリースコア(0-100)と方向性を返す。

    スコア内訳（合計100点）:
        EMA20方向       : 20点
        RSIゾーン       : 25点
        MACDヒストグラム : 20点
        BB位置          : 15点
        ATRボラ         : 20点

    Returns:
        {
            "score": int (0-100),
            "direction": "long" | "short" | "neutral",
            "long_score": int,
            "short_score": int,
            "details": dict  # {"項目": (説明, long点, short点), ...}
        }
    """
    if len(df) < 50:
        return {"score": 50, "direction": "neutral", "long_score": 50,
                "short_score": 50, "details": {}}

    close = df["close"]
    details: dict = {}
    long_pts: list[int] = []
    short_pts: list[int] = []

    # ---- 1. EMA20 方向性 (重み: 20) ----
    ema20 = close.ewm(span=20, adjust=False).mean()
    # 短期足（1M/5M）では2本差でも十分。長期足は多少広く見る
    _slope_bars = 2 if len(close) >= 200 else min(2, len(close) - 1)
    slope = float(ema20.iloc[-1]) - float(ema20.iloc[-1 - _slope_bars])
    price_vs_ema = float(close.iloc[-1]) - float(ema20.iloc[-1])
    if slope > 0 and price_vs_ema > 0:
        l, s = 20, 0;  desc = "↑ 上昇"
    elif slope < 0 and price_vs_ema < 0:
        l, s = 0, 20;  desc = "↓ 下降"
    else:
        l, s = 10, 10; desc = "→ 中立"
    long_pts.append(l); short_pts.append(s)
    details["EMA20"] = (desc, l, s)

    # ---- 2. RSI (重み: 25) ----
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
    if rsi < 30:
        l, s = 25, 0;  desc = f"{rsi:.0f} 売られすぎ"
    elif rsi > 70:
        l, s = 0, 25;  desc = f"{rsi:.0f} 買われすぎ"
    elif rsi < 45:
        l, s = 18, 7;  desc = f"{rsi:.0f} やや売られ"
    elif rsi > 55:
        l, s = 7, 18;  desc = f"{rsi:.0f} やや買われ"
    else:
        l, s = 12, 12; desc = f"{rsi:.0f} 中立"
    long_pts.append(l); short_pts.append(s)
    details["RSI"] = (desc, l, s)

    # ---- 3. MACD ヒストグラム (重み: 20) ----
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig
    h_now  = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2]) if len(hist) >= 2 else 0.0
    if h_now > 0 and h_now > h_prev:
        l, s = 20, 0;  desc = "↑ 上昇加速"
    elif h_now > 0:
        l, s = 15, 5;  desc = "↗ 上昇減速"
    elif h_now < 0 and h_now < h_prev:
        l, s = 0, 20;  desc = "↓ 下降加速"
    elif h_now < 0:
        l, s = 5, 15;  desc = "↘ 下降減速"
    else:
        l, s = 10, 10; desc = "→ 中立"
    long_pts.append(l); short_pts.append(s)
    details["MACD"] = (desc, l, s)

    # ---- 4. ボリンジャーバンド位置 (重み: 15) ----
    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    bb_up2  = bb_mid + 2 * bb_std
    bb_lo2  = bb_mid - 2 * bb_std
    rng     = float(bb_up2.iloc[-1]) - float(bb_lo2.iloc[-1])
    bb_pos  = (float(close.iloc[-1]) - float(bb_lo2.iloc[-1])) / rng if rng > 0 else 0.5
    if bb_pos < 0.2:
        l, s = 15, 0;  desc = "下バンド付近"
    elif bb_pos > 0.8:
        l, s = 0, 15;  desc = "上バンド付近"
    elif bb_pos < 0.4:
        l, s = 10, 5;  desc = "下寄り"
    elif bb_pos > 0.6:
        l, s = 5, 10;  desc = "上寄り"
    else:
        l, s = 7, 7;   desc = "中心付近"
    long_pts.append(l); short_pts.append(s)
    details["BB"] = (desc, l, s)

    # ---- 5. ATRボラティリティ (重み: 15) ----
    h = df["high"].values; lv = df["low"].values; c = df["close"].values
    tr = np.maximum.reduce([
        h[1:] - lv[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(lv[1:] - c[:-1]),
    ])
    atr14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else 0.0
    atr50 = float(np.mean(tr[-50:])) if len(tr) >= 50 else atr14
    ratio = atr14 / atr50 if atr50 > 0 else 1.0
    if ratio > 1.5:
        l, s = 5, 5;   desc = "高ボラ注意"
    elif ratio < 0.7:
        l, s = 5, 5;   desc = "閑散相場"
    else:
        l, s = 15, 15; desc = "適正ボラ"
    long_pts.append(l); short_pts.append(s)
    details["ボラ"] = (desc, l, s)

    # ---- 6. Stochastic %K (重み: 20, スキャルプ逆張り検出) ----
    _hi5 = pd.Series(df["high"].values).rolling(5).max()
    _lo5 = pd.Series(df["low"].values).rolling(5).min()
    _rng = (_hi5 - _lo5).replace(0, float("nan"))
    stoch_k_s = ((close - _lo5) / _rng * 100).rolling(3).mean()
    sk = float(stoch_k_s.iloc[-1]) if len(stoch_k_s) >= 5 and not pd.isna(stoch_k_s.iloc[-1]) else 50.0
    sk1 = float(stoch_k_s.iloc[-2]) if len(stoch_k_s) >= 6 and not pd.isna(stoch_k_s.iloc[-2]) else sk
    if sk < 20 and sk > sk1:          # 売られすぎからの反転
        l, s = 20, 0;  desc = f"{sk:.0f} 売られすぎ反転"
    elif sk > 80 and sk < sk1:        # 買われすぎからの反転
        l, s = 0, 20;  desc = f"{sk:.0f} 買われすぎ反転"
    elif sk < 30:
        l, s = 15, 5;  desc = f"{sk:.0f} OS圏"
    elif sk > 70:
        l, s = 5, 15;  desc = f"{sk:.0f} OB圏"
    else:
        l, s = 10, 10; desc = f"{sk:.0f} 中立"
    long_pts.append(l); short_pts.append(s)
    details["Stoch%K"] = (desc, l, s)

    # ---- スコア計算 ----
    total_long  = sum(long_pts)
    total_short = sum(short_pts)
    diff = total_long - total_short

    if diff > 20:
        score     = min(100, 50 + int(diff * 1.5))
        direction = "long"
    elif diff < -20:
        score     = min(100, 50 + int(-diff * 1.5))
        direction = "short"
    else:
        score     = 50
        direction = "neutral"

    return {
        "score":       score,
        "direction":   direction,
        "long_score":  total_long,
        "short_score": total_short,
        "details":     details,
    }


# ============================================================
# 通貨強弱メーター
# ============================================================

# 強弱計算に使う主要ペア
_STRENGTH_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
]

_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]


def calc_currency_strength(
    timeframe: str = "1H",
    lookback: int = 14,
) -> dict[str, float]:
    """
    主要通貨ペアのROCを使って各通貨の相対強弱スコアを計算する。

    Returns:
        {"USD": 1.5, "JPY": -0.8, ...}  正が強い、負が弱い（降順ソート済み）
    """
    from dashboard.sample_data import get_ohlcv_dataframe

    pair_roc: dict[str, float] = {}
    for pair in _STRENGTH_PAIRS:
        try:
            df, _ = get_ohlcv_dataframe(pair, timeframe, count=lookback + 5)
            if len(df) < lookback + 1:
                continue
            c   = df["close"]
            roc = (float(c.iloc[-1]) - float(c.iloc[-lookback - 1])) / float(c.iloc[-lookback - 1])
            pair_roc[pair] = roc
        except Exception:
            pass

    if not pair_roc:
        return {}

    strength: dict[str, float] = {c: 0.0 for c in _CURRENCIES}
    count:    dict[str, int]   = {c: 0    for c in _CURRENCIES}

    for pair, roc in pair_roc.items():
        base  = pair[:3]
        quote = pair[3:]
        if base in strength:
            strength[base] += roc
            count[base]    += 1
        if quote in strength:
            strength[quote] -= roc
            count[quote]    += 1

    for cur in _CURRENCIES:
        if count[cur] > 0:
            strength[cur] = round(strength[cur] / count[cur] * 100, 2)

    return dict(sorted(strength.items(), key=lambda x: -x[1]))
