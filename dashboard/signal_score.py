"""
dashboard/signal_score.py

エントリースコアリングと通貨強弱メーター。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# HTF マッピング（上位時間足）
# ============================================================
_HIGHER_TF: dict[str, str] = {
    "M1":  "M15",
    "M5":  "H1",
    "M15": "H4",
    "H1":  "D1",
    "H4":  "W1",
    "D1":  "W1",
    # エイリアス
    "1M":  "15M",
    "5M":  "1H",
    "15M": "4H",
    "1H":  "D1",
    "4H":  "W1",
}


def _hurst_latest(close: pd.Series, window: int = 100) -> float:
    """
    R/S 分析でハースト指数を推定する（最新 window 本を使用）。
    H > 0.55 → トレンド相場
    H < 0.45 → 平均回帰相場
    """
    s = close.dropna().iloc[-window:]
    n = len(s)
    if n < 20:
        return 0.5
    try:
        log_vals: list[float] = []
        log_ns:   list[float] = []
        for sub in [n // 4, n // 3, n // 2, n]:
            if sub < 8:
                continue
            chunk = s.iloc[-sub:].values.astype(float)
            mean  = chunk.mean()
            devs  = np.cumsum(chunk - mean)
            rs    = (devs.max() - devs.min()) / (chunk.std(ddof=0) or 1e-10)
            log_vals.append(float(np.log(rs)))
            log_ns.append(float(np.log(sub)))
        if len(log_ns) < 2:
            return 0.5
        h = float(np.polyfit(log_ns, log_vals, 1)[0])
        return max(0.0, min(1.0, h))
    except Exception:
        return 0.5


def _get_htf_direction(symbol: str, timeframe: str) -> str:
    """
    上位時間足の EMA200 + RSI50 でトレンド方向を判定。
    Returns: "long" | "short" | "neutral"
    """
    htf = _HIGHER_TF.get(timeframe, "")
    if not htf or not symbol:
        return "neutral"
    try:
        from dashboard.sample_data import get_ohlcv_dataframe
        df_htf, _ = get_ohlcv_dataframe(symbol, htf, count=250)
        if len(df_htf) < 200:
            return "neutral"
        c = df_htf["close"]
        ema200 = c.ewm(span=200, adjust=False).mean()
        slope  = float(ema200.iloc[-1]) - float(ema200.iloc[-5])
        above  = float(c.iloc[-1]) > float(ema200.iloc[-1])
        # RSI50
        d    = c.diff()
        gain = d.clip(lower=0).rolling(14).mean()
        loss = (-d.clip(upper=0)).rolling(14).mean()
        rs   = gain / loss.replace(0, float("nan"))
        rsi  = float((100 - 100 / (1 + rs)).iloc[-1])
        votes_long  = sum([slope > 0, above, rsi > 50])
        votes_short = sum([slope < 0, not above, rsi < 50])
        if votes_long  >= 2:
            return "long"
        if votes_short >= 2:
            return "short"
    except Exception:
        pass
    return "neutral"


# ============================================================
# エントリースコアリング
# ============================================================

def calc_entry_score(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> dict:
    """
    複数インジケーターを組み合わせてエントリースコア(0-100)と方向性を返す。

    スコア内訳:
        HTF EMA200+RSI50 : 35点  ← 上位時間足トレンドフィルター
        EMA20方向         : 20点
        RSI(7)ゾーン      : 25点
        MACDヒストグラム  : 20点
        Stoch%K           : 20点
        ATRボラ           : 15点

    Hurst ゲート: H < 0.45（平均回帰相場）はトレンド系信号を抑制し neutral を返す。
    閾値: diff > 20 で方向性確定

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

    # ---- 0. Hurst ゲート ----
    hurst = _hurst_latest(close)
    if hurst < 0.45:
        details["Hurst"] = (f"{hurst:.2f} 平均回帰相場", 0, 0)
        return {"score": 50, "direction": "neutral", "long_score": 50,
                "short_score": 50, "details": details}

    # ---- 1. HTF EMA200 + RSI50 (重み: 35) ----
    htf_dir = _get_htf_direction(symbol, timeframe)
    if htf_dir == "long":
        l, s = 35, 0;  desc = "↑ 上位足トレンド上昇"
    elif htf_dir == "short":
        l, s = 0, 35;  desc = "↓ 上位足トレンド下降"
    else:
        l, s = 17, 17; desc = "→ 上位足中立"
    long_pts.append(l); short_pts.append(s)
    details["HTF_EMA200"] = (desc, l, s)

    # ---- 2. EMA20 方向性 (重み: 20) ----
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

    # ---- 3. RSI(7) (重み: 25) ----
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(7).mean()
    loss  = (-delta.clip(upper=0)).rolling(7).mean()
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

    # ---- 4. MACD ヒストグラム (重み: 20) ----
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

    # ---- 6. Stochastic %K (重み: 20) ----
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
