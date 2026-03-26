"""
dashboard/indicators.py

インジケーター計算モジュール。
pandas による純粋な計算のみ担当。タイムスタンプ変換は chart_utils 側で行う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# UIに表示するインジケーター選択肢
INDICATOR_OPTIONS = [
    "SMA 5", "SMA 10", "SMA 20", "SMA 50", "SMA 200",
    "EMA 5", "EMA 10", "EMA 20", "EMA 50", "EMA 200",
    "ボリンジャーバンド (20)",
    "RSI (14)",
    "MACD (12, 26, 9)",
    "ストキャスティクス (5,3,3)",
    "累積出来高デルタ (CVD)",
    "レジサポライン",
    "直近高値/安値",
    "ピボットポイント",
    "セッション区切り",
    "ダイバージェンス (RSI)",
    "ローソク足パターン",
    "ZigZag（転換点予測）",
    "エントリーシグナル",
    "VWAP",
]

# インジケーターの色
INDICATOR_COLORS: dict[str, str] = {
    "SMA 5":   "#ffffff",
    "SMA 10":  "#ffeb3b",
    "SMA 20":  "#2196f3",
    "SMA 50":  "#ff9800",
    "SMA 200": "#f44336",
    "EMA 5":   "#b0bec5",
    "EMA 10":  "#fff176",
    "EMA 20":  "#64b5f6",
    "EMA 50":  "#ffb74d",
    "EMA 200": "#ef9a9a",
}


def calc_sma(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].rolling(period).mean()


def calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def calc_bollinger(df: pd.DataFrame, period: int = 20) -> dict[str, pd.Series]:
    """1σ/2σ/3σのバンドをすべて返す。"""
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return {
        "mid":     mid,
        "upper_1": mid + 1 * std,
        "lower_1": mid - 1 * std,
        "upper_2": mid + 2 * std,
        "lower_2": mid - 2 * std,
        "upper_3": mid + 3 * std,
        "lower_3": mid - 3 * std,
    }


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = df["close"].ewm(span=fast,   adjust=False).mean()
    ema_s = df["close"].ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    hist  = macd - sig
    return macd, sig, hist


def calc_sr_lines(
    df: pd.DataFrame,
    window: int = 5,
    max_lines: int = 5,
    threshold_pct: float = 0.0015,
) -> list[dict]:
    """
    スウィングハイ/ローからレジスタンス・サポートラインを検出する。

    Parameters:
        window       : ローカル高安の判定ウィンドウ（前後 N 本）
        max_lines    : 抵抗/支持それぞれの最大本数
        threshold_pct: 同一レベルとみなす価格差率
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    res_candidates: list[float] = []
    sup_candidates: list[float] = []

    for i in range(window, n - window):
        h = float(highs[i])
        if h >= max(highs[max(0, i - window): i + window + 1]):
            res_candidates.append(h)
        lo = float(lows[i])
        if lo <= min(lows[max(0, i - window): i + window + 1]):
            sup_candidates.append(lo)

    def cluster_and_rank(levels: list[float]) -> list[float]:
        if not levels:
            return []
        sorted_lvl = sorted(levels)
        clusters: list[list[float]] = [[sorted_lvl[0]]]
        for v in sorted_lvl[1:]:
            if v <= clusters[-1][0] * (1 + threshold_pct):
                clusters[-1].append(v)
            else:
                clusters.append([v])
        clusters.sort(key=lambda c: -len(c))
        return [round(sum(c) / len(c), 5) for c in clusters[:max_lines]]

    result: list[dict] = []
    for p in cluster_and_rank(res_candidates):
        result.append({"price": p, "type": "resistance"})
    for p in cluster_and_rank(sup_candidates):
        result.append({"price": p, "type": "support"})
    return result


def calc_pivot_points(df: pd.DataFrame) -> list[dict]:
    """
    前日のOHLCから当日のピボットポイント（R3〜P〜S3）を計算する。
    1時間足以下のデータで利用する。データが日足の場合は前週データを使用。
    """
    try:
        daily = df.resample("D").agg(
            high=("high", "max"),
            low=("low",  "min"),
            close=("close", "last"),
        ).dropna()
    except Exception:
        return []

    if len(daily) < 2:
        return []

    prev = daily.iloc[-2]
    H = float(prev["high"])
    L = float(prev["low"])
    C = float(prev["close"])

    P  = (H + L + C) / 3
    R1 = 2 * P - L
    R2 = P + (H - L)
    R3 = H + 2 * (P - L)
    S1 = 2 * P - H
    S2 = P - (H - L)
    S3 = L - 2 * (H - P)

    return [
        {"price": round(P,  5), "label": "P"},
        {"price": round(R1, 5), "label": "R1"},
        {"price": round(R2, 5), "label": "R2"},
        {"price": round(R3, 5), "label": "R3"},
        {"price": round(S1, 5), "label": "S1"},
        {"price": round(S2, 5), "label": "S2"},
        {"price": round(S3, 5), "label": "S3"},
    ]


def calc_session_markers(df: pd.DataFrame, jst_offset: int) -> list[dict]:
    """
    各セッション開始時刻の最初の足にマーカーを生成する。
    東京 UTC 00:00 / London UTC 07:00 / NY UTC 12:00
    """
    sessions = [
        (0,  "東京",   "#ffeb3b"),
        (7,  "London", "#ff9800"),
        (12, "NY",     "#ce93d8"),
    ]
    markers: list[dict] = []
    seen: set = set()

    for ts in df.index:
        for utc_hour, label, color in sessions:
            if ts.hour == utc_hour:
                key = (ts.date(), utc_hour)
                if key not in seen:
                    seen.add(key)
                    markers.append({
                        "time":     int(ts.timestamp()) + jst_offset,
                        "position": "belowBar",
                        "color":    color,
                        "shape":    "circle",
                        "text":     label,
                    })
    return markers


def calc_stochastic(
    df: pd.DataFrame, k_period: int = 5, d_period: int = 3, smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    """ストキャスティクス %K（平滑化）と %D を返す。"""
    low_k  = df["low"].rolling(k_period).min()
    high_k = df["high"].rolling(k_period).max()
    raw_k  = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, float("nan"))
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def calc_recent_hl(df: pd.DataFrame, periods: list[int] | None = None) -> list[dict]:
    """直近 N 本の高値・安値ラインを {price, label} のリストで返す。"""
    if periods is None:
        periods = [5, 10, 20]
    result = []
    for p in periods:
        window = df.iloc[-p:]
        result.append({"price": round(float(window["high"].max()), 5), "label": f"H{p}"})
        result.append({"price": round(float(window["low"].min()),  5), "label": f"L{p}"})
    return result


def calc_divergence(df: pd.DataFrame, jst_offset: int, lookback: int = 100) -> list[dict]:
    """
    RSIダイバージェンスを検出してマーカーリストを返す。
    ブリッシュ: 価格安値↓ RSI安値↑ → aboveBar arrowUp (#26a69a)
    ベアリッシュ: 価格高値↑ RSI高値↓ → belowBar arrowDown (#ef5350)
    """
    if len(df) < 30:
        return []

    window = min(lookback, len(df))
    df_w   = df.iloc[-window:].copy()
    ts     = df_w.index
    close  = df_w["close"]
    high   = df_w["high"]
    low    = df_w["low"]

    # RSI 計算
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - 100 / (1 + rs)

    swing_n = 5
    price_highs: list[tuple[int, float]] = []
    price_lows:  list[tuple[int, float]] = []
    rsi_highs:   list[tuple[int, float]] = []
    rsi_lows:    list[tuple[int, float]] = []
    ts_highs:    list = []
    ts_lows:     list = []

    for i in range(swing_n, len(df_w) - swing_n):
        h  = float(high.iloc[i])
        lo = float(low.iloc[i])
        rv = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else float("nan")

        if h >= max(high.iloc[max(0, i - swing_n): i + swing_n + 1]):
            price_highs.append((i, h))
            rsi_highs.append((i, rv))
            ts_highs.append(ts[i])

        if lo <= min(low.iloc[max(0, i - swing_n): i + swing_n + 1]):
            price_lows.append((i, lo))
            rsi_lows.append((i, rv))
            ts_lows.append(ts[i])

    markers: list[dict] = []

    # ベアリッシュダイバージェンス（価格↑ RSI↓）
    for k in range(1, min(5, len(price_highs))):
        _, p1 = price_highs[-k - 1];  _, p2 = price_highs[-k]
        _, r1 = rsi_highs[-k - 1];    _, r2 = rsi_highs[-k]
        if p2 > p1 and r2 < r1 and not (np.isnan(r1) or np.isnan(r2)):
            markers.append({
                "time":     int(ts_highs[-k].timestamp()) + jst_offset,
                "position": "aboveBar",
                "color":    "#ef5350",
                "shape":    "arrowDown",
                "text":     "D↓",
            })
            break

    # ブリッシュダイバージェンス（価格↓ RSI↑）
    for k in range(1, min(5, len(price_lows))):
        _, p1 = price_lows[-k - 1];  _, p2 = price_lows[-k]
        _, r1 = rsi_lows[-k - 1];    _, r2 = rsi_lows[-k]
        if p2 < p1 and r2 > r1 and not (np.isnan(r1) or np.isnan(r2)):
            markers.append({
                "time":     int(ts_lows[-k].timestamp()) + jst_offset,
                "position": "belowBar",
                "color":    "#26a69a",
                "shape":    "arrowUp",
                "text":     "D↑",
            })
            break

    return markers


def calc_candlestick_patterns(df: pd.DataFrame, jst_offset: int, lookback: int = 100) -> list[dict]:
    """
    ローソク足パターンを認識してマーカーリストを返す。
    対象: ドージ, ハンマー, 流れ星(シューティングスター), 上昇包み足, 下降包み足
    """
    if len(df) < 3:
        return []

    window = min(lookback, len(df))
    df_w   = df.iloc[-window:].copy()
    ts     = df_w.index
    opens  = df_w["open"].values
    highs  = df_w["high"].values
    lows   = df_w["low"].values
    closes = df_w["close"].values

    markers: list[dict] = []

    for i in range(1, len(df_w)):
        o = opens[i];  h = highs[i];  lo = lows[i];  c = closes[i]
        body  = abs(c - o)
        total = h - lo
        if total < 1e-10:
            continue

        upper_wick  = h - max(c, o)
        lower_wick  = min(c, o) - lo
        body_ratio  = body / total
        pattern     = None
        color       = "#ffffff"
        position    = "belowBar"
        shape       = "circle"

        # ドージ: ボディが全体の10%以下
        if body_ratio < 0.1:
            pattern  = "Doji"
            color    = "#ffeb3b"
            position = "aboveBar"
            shape    = "circle"

        # ハンマー / ピンバー上昇: 下ヒゲ≥ボディ×2 かつ 上ヒゲ小
        elif lower_wick >= body * 2 and upper_wick < body * 0.5 and body_ratio < 0.45:
            pattern  = "Hammer"
            color    = "#26a69a"
            position = "belowBar"
            shape    = "arrowUp"

        # 流れ星: 上ヒゲ≥ボディ×2 かつ 下ヒゲ小
        elif upper_wick >= body * 2 and lower_wick < body * 0.5 and body_ratio < 0.45:
            pattern  = "Star"
            color    = "#ef5350"
            position = "aboveBar"
            shape    = "arrowDown"

        # 上昇包み足: 前が陰線 → 現在が陽線かつ前足を完全包含
        elif (closes[i - 1] < opens[i - 1]
              and c > o
              and c >= opens[i - 1] and o <= closes[i - 1]):
            pattern  = "Bull"
            color    = "#26a69a"
            position = "belowBar"
            shape    = "arrowUp"

        # 下降包み足: 前が陽線 → 現在が陰線かつ前足を完全包含
        elif (closes[i - 1] > opens[i - 1]
              and c < o
              and o >= closes[i - 1] and c <= opens[i - 1]):
            pattern  = "Bear"
            color    = "#ef5350"
            position = "aboveBar"
            shape    = "arrowDown"

        if pattern:
            markers.append({
                "time":     int(ts[i].timestamp()) + jst_offset,
                "position": position,
                "color":    color,
                "shape":    shape,
                "text":     pattern,
            })

    return markers


def calc_zigzag(df: pd.DataFrame, jst_offset: int) -> dict:
    """
    ZigZag インジケーター + 次の山/谷を予測する。

    ATR14 × 1.5 を最小スウィング幅として使用。
    確定済みの山/谷を検出し、直近スウィング幅の平均から次の転換点を予測する。

    Returns:
        {
            "line":       [{time, value}, ...],   # ZigZag折れ線
            "markers":    [{marker}, ...],          # 山▼ 谷▲ マーカー
            "prediction": {                         # 次の転換点予測
                "price":     float,
                "direction": "high" | "low",
                "label":     str,
                "pips":      float,
            } | None
        }
    """
    if len(df) < 20:
        return {"line": [], "markers": [], "prediction": None}

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    ts     = df.index
    n      = len(df)

    # ATR14 でスウィング最小幅を決定
    h_a = highs[1:]; l_a = lows[1:]; c_p = closes[:-1]
    tr  = np.maximum.reduce([h_a - l_a, np.abs(h_a - c_p), np.abs(l_a - c_p)])
    atr      = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    min_swing = atr * 1.5

    # pip サイズ自動判定（JPY ペアは price > 10）
    mid_price = float(closes[-1])
    pip_sz    = 0.01 if mid_price > 10 else 0.0001

    # ---- ZigZag ピボット検出 ----
    pivots: list[tuple[int, float, str]] = []  # (index, price, "high"|"low")
    direction: str | None = None
    ext_idx   = 0
    ext_price = closes[0]

    for i in range(1, n):
        if direction is None:
            if highs[i] >= ext_price + min_swing:
                direction = "up"; ext_idx = i; ext_price = highs[i]
            elif lows[i] <= ext_price - min_swing:
                direction = "down"; ext_idx = i; ext_price = lows[i]
        elif direction == "up":
            if highs[i] >= ext_price:
                ext_idx = i; ext_price = highs[i]
            elif ext_price - lows[i] >= min_swing:
                pivots.append((ext_idx, ext_price, "high"))
                direction = "down"; ext_idx = i; ext_price = lows[i]
        else:  # down
            if lows[i] <= ext_price:
                ext_idx = i; ext_price = lows[i]
            elif highs[i] - ext_price >= min_swing:
                pivots.append((ext_idx, ext_price, "low"))
                direction = "up"; ext_idx = i; ext_price = highs[i]

    # 現在進行中の最新ピボット（未確定）を追加
    if direction == "up":
        pivots.append((ext_idx, ext_price, "high"))
    elif direction == "down":
        pivots.append((ext_idx, ext_price, "low"))

    if len(pivots) < 2:
        return {"line": [], "markers": [], "prediction": None}

    # ---- ZigZag 折れ線データ ----
    line_data = [
        {"time": int(ts[idx].timestamp()) + jst_offset, "value": round(price, 5)}
        for idx, price, _ in pivots
    ]

    # ---- マーカー（確定済みのみ） ----
    confirmed = pivots[:-1]
    markers: list[dict] = []
    for idx, price, ptype in confirmed:
        markers.append({
            "time":     int(ts[idx].timestamp()) + jst_offset,
            "position": "aboveBar" if ptype == "high" else "belowBar",
            "color":    "#ef5350"  if ptype == "high" else "#26a69a",
            "shape":    "arrowDown" if ptype == "high" else "arrowUp",
            "text":     "",
        })

    return {"line": line_data, "markers": markers}


def calc_entry_signals(df: pd.DataFrame, jst_offset: int, params: dict | None = None) -> list[dict]:
    """
    短期トレード（スキャルピング）特化エントリーシグナル。

    ── 使用インジ（すべて短周期）────────────────────────────────
      EMA5 / EMA20    : 短期トレンド方向
      ADX(10) > 15    : レンジ相場フィルター
      Stoch(5,3,3)    : 過買い/過売りタイミング（最重要）
      RSI(9)          : モメンタム確認（短周期）
      MACD(5,13,5)    : 短期ヒストグラム転換
      BB(14)          : 価格の伸び過ぎ/収縮判定
      ATR(7)          : ボラティリティ判定・実体確認

    ── 必須フィルター ───────────────────────────────────────────
      F1. EMA5 vs EMA20 の方向一致（1段階整列）
      F2. ADX(10) ≥ 15（極端なレンジは除外、短期なので低め）
      F3. シグナルバーの実体 ≥ ATR7 × 0.2（方向性のある足のみ）

    ── タイミングスコア（最大 9 点、閾値 ≥ 4）──────────────────
      A. Stoch(5,3,3) クロス: ≤25 or ≥75 でクロス = 3pt / ≤35 or ≥65 = 1pt
      B. RSI(9): 方向ゾーン内（Long:35〜55 / Short:45〜65）で上昇/下降 = 1pt
                 Over売り/買い反転（<30 or >70）= 2pt
      C. MACD(5,13,5) ヒストグラム: ゼロクロス = 2pt / 同方向2本連続 = 1pt
      D. BB(14): 2σタッチ後帰還 = 2pt / 1σタッチ後帰還 = 1pt
      E. 勢いある実体（body > ATR7 × 0.5）= 1pt

    ── クールダウン ─────────────────────────────────────────────
      4本（1M足なら4分、5M足なら20分）
    """
    p = params or {}
    _ADX_MIN     = p.get("adx_min",      15)
    _SCORE_THRESH= p.get("score_thresh",  4)
    _STOCH_STRONG= p.get("stoch_strong", 25)   # この値以下/以上でGC/DC → 3pt
    _STOCH_NORMAL= p.get("stoch_normal", 35)   # この値以下/以上でGC/DC → 1pt
    _BODY_MULT   = p.get("body_mult",   0.2)   # body >= ATR7 × この値
    _COOLDOWN    = p.get("cooldown",      4)

    if len(df) < 35:
        return []

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    opens  = df["open"]
    volume = df["volume"]
    ts     = df.index
    n      = len(df)

    # ---- EMA5 / EMA20 ----
    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    # ---- RSI(9) ----
    _d  = close.diff()
    rsi = 100 - 100 / (
        1 + _d.clip(lower=0).rolling(9).mean()
          / (-_d.clip(upper=0)).rolling(9).mean().replace(0, float("nan"))
    )

    # ---- MACD(5,13,5) ----
    _macd = close.ewm(span=5,  adjust=False).mean() - close.ewm(span=13, adjust=False).mean()
    hist  = _macd - _macd.ewm(span=5, adjust=False).mean()

    # ---- Stochastic(5,3,3) ----
    _rk     = 100 * (close - low.rolling(5).min()) / (
                  high.rolling(5).max() - low.rolling(5).min()
              ).replace(0, float("nan"))
    stoch_k = _rk.rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()

    # ---- Bollinger Bands(14) ----
    bb_mid = close.rolling(14).mean()
    bb_std = close.rolling(14).std()
    bb_u2  = bb_mid + 2 * bb_std
    bb_l2  = bb_mid - 2 * bb_std
    bb_u1  = bb_mid + 1 * bb_std
    bb_l1  = bb_mid - 1 * bb_std

    # ---- ATR(7) ----
    _h  = high.values; _l = low.values; _c = close.values
    _tr = np.maximum.reduce([
        _h[1:] - _l[1:],
        np.abs(_h[1:] - _c[:-1]),
        np.abs(_l[1:] - _c[:-1]),
    ])
    atr7 = pd.Series(np.concatenate([[float("nan")], _tr]),
                     index=close.index).rolling(7).mean()

    # ---- ADX(10) ---- Wilder平滑化
    _pdm = np.zeros(n); _mdm = np.zeros(n); _tr2 = np.zeros(n)
    for j in range(1, n):
        hd = float(high.iloc[j]) - float(high.iloc[j-1])
        ld = float(low.iloc[j-1]) - float(low.iloc[j])
        _pdm[j] = hd if hd > ld and hd > 0 else 0.0
        _mdm[j] = ld if ld > hd and ld > 0 else 0.0
        _tr2[j]  = max(
            float(high.iloc[j]) - float(low.iloc[j]),
            abs(float(high.iloc[j]) - float(close.iloc[j-1])),
            abs(float(low.iloc[j])  - float(close.iloc[j-1])),
        )
    _a   = 1.0 / 10
    _idx = close.index
    _pdi = 100 * pd.Series(_pdm, index=_idx).ewm(alpha=_a, adjust=False).mean() / \
                 pd.Series(_tr2, index=_idx).ewm(alpha=_a, adjust=False).mean().replace(0, float("nan"))
    _mdi = 100 * pd.Series(_mdm, index=_idx).ewm(alpha=_a, adjust=False).mean() / \
                 pd.Series(_tr2, index=_idx).ewm(alpha=_a, adjust=False).mean().replace(0, float("nan"))
    _dx  = 100 * (_pdi - _mdi).abs() / (_pdi + _mdi).replace(0, float("nan"))
    adx  = _dx.ewm(alpha=_a, adjust=False).mean()

    # ---- ボリューム移動平均(10) ----
    vol_ma = volume.rolling(10).mean()

    markers: list[dict] = []
    last_long_idx  = -8
    last_short_idx = -8

    for i in range(30, n - 1):
        if any(pd.isna(x) for x in [
            ema20.iloc[i], rsi.iloc[i], hist.iloc[i],
            stoch_k.iloc[i], stoch_d.iloc[i],
            bb_mid.iloc[i], atr7.iloc[i], adx.iloc[i],
        ]):
            continue

        c   = float(close.iloc[i]);  o_i = float(opens.iloc[i])
        hi  = float(high.iloc[i]);   lo  = float(low.iloc[i])

        e5  = float(ema5.iloc[i]);   e20 = float(ema20.iloc[i])
        e5p = float(ema5.iloc[i-2])  # EMA5 傾き（2本前）

        r   = float(rsi.iloc[i]);    r1  = float(rsi.iloc[i-1])

        h_i = float(hist.iloc[i]);   h1  = float(hist.iloc[i-1]); h2 = float(hist.iloc[i-2])

        sk  = float(stoch_k.iloc[i]); sd  = float(stoch_d.iloc[i])
        sk1 = float(stoch_k.iloc[i-1]); sd1 = float(stoch_d.iloc[i-1])

        bu2 = float(bb_u2.iloc[i]);  bl2 = float(bb_l2.iloc[i])
        bu1 = float(bb_u1.iloc[i]);  bl1 = float(bb_l1.iloc[i])
        bm  = float(bb_mid.iloc[i])

        h_p1 = float(high.iloc[i-1]);  l_p1 = float(low.iloc[i-1])
        c_p  = float(close.iloc[i-1]); o_p  = float(opens.iloc[i-1])

        atr  = float(atr7.iloc[i])
        adx_v = float(adx.iloc[i])

        vol_now = float(volume.iloc[i])
        vol_avg = float(vol_ma.iloc[i]) if not pd.isna(vol_ma.iloc[i]) else 0.0

        body        = abs(c - o_i)
        total_range = hi - lo
        lower_wick  = (min(c, o_i) - lo)  if total_range > 0 else 0.0
        upper_wick  = (hi - max(c, o_i))  if total_range > 0 else 0.0

        # F2: ADX フィルター
        if adx_v < _ADX_MIN:
            continue

        # F1: トレンド方向
        trend_long  = e5 > e20 and e5 > e5p
        trend_short = e5 < e20 and e5 < e5p

        # F3: 実体チェック
        min_body = atr * _BODY_MULT if atr > 0 else 0.0

        # ===== LONG =====
        if trend_long and i - last_long_idx >= _COOLDOWN:
            if not (c > o_i and body >= min_body):
                pass
            else:
                ls = 0

                # A. Stochastic クロス
                gc = sk > sd and sk1 <= sd1
                if gc and sk <= _STOCH_STRONG:
                    ls += 3
                elif gc and sk <= _STOCH_NORMAL:
                    ls += 1

                # B. RSI
                if r < 30 and r > r1:
                    ls += 2
                elif 35 <= r <= 55 and r > r1:
                    ls += 1

                # C. MACD ヒストグラム
                if h_i > 0 and h1 <= 0:
                    ls += 2
                elif h_i > h1 > h2:
                    ls += 1

                # D. BB
                if l_p1 <= bl2 and c > bl1:
                    ls += 2
                elif l_p1 <= bl1 and c > bm:
                    ls += 1

                # E. 勢いある実体
                if body >= atr * 0.5:
                    ls += 1

                if ls >= _SCORE_THRESH:
                    markers.append({
                        "time":     int(ts[i].timestamp()) + jst_offset,
                        "position": "belowBar",
                        "color":    "#00e676",
                        "shape":    "arrowUp",
                        "text":     f"▲{ls}",
                    })
                    last_long_idx = i

        # ===== SHORT =====
        if trend_short and i - last_short_idx >= _COOLDOWN:
            if not (c < o_i and body >= min_body):
                pass
            else:
                ss = 0

                # A. Stochastic クロス
                dc = sk < sd and sk1 >= sd1
                if dc and sk >= (100 - _STOCH_STRONG):
                    ss += 3
                elif dc and sk >= (100 - _STOCH_NORMAL):
                    ss += 1

                # B. RSI
                if r > 70 and r < r1:           # 買われすぎ反転
                    ss += 2
                elif 45 <= r <= 65 and r < r1:  # 戻り売りゾーン下降
                    ss += 1

                # C. MACD(5,13,5) ヒストグラム
                if h_i < 0 and h1 >= 0:         # ゼロクロス
                    ss += 2
                elif h_i < h1 < h2:             # 2本連続下向き
                    ss += 1

                # D. BB
                if h_p1 >= bu2 and c < bu1:
                    ss += 2
                elif h_p1 >= bu1 and c < bm:
                    ss += 1

                # E. 勢いある実体
                if body >= atr * 0.5:
                    ss += 1

                if ss >= _SCORE_THRESH:
                    markers.append({
                        "time":     int(ts[i].timestamp()) + jst_offset,
                        "position": "aboveBar",
                        "color":    "#ff1744",
                        "shape":    "arrowDown",
                        "text":     f"▼{ss}",
                    })
                    last_short_idx = i

    return markers


def calc_ai_signal(
    df: pd.DataFrame,
    jst_offset: int,
    params: dict | None = None,
    extra_labels: list[dict] | None = None,
) -> list[dict]:
    """
    k-NN パターンマッチング AIシグナル（numpy のみ・追加インストール不要）。

    ── アルゴリズム ────────────────────────────────────────────
    1. 各バーから7次元の正規化特徴量を抽出
    2. 各バーを「5本後の値動き vs ATR」でラベリング（LONG/SHORT/中立）
    3. 予測したいバーに対し、過去の類似バーk=15本をユークリッド距離で検索
       ※ルックアヘッドバイアス回避：参照バー候補は予測バーより6本以上古い
    4. 距離加重投票で LONG/SHORT 確率を計算
       ・直近重み: 最近の参照バーを重視（指数減衰）
       ・相場環境マッチング: ADX・ボラティリティが類似した局面の参照を優先
       ・extra_labels: 過去の正解シグナルを高ウェイトで追加参照（自己学習）
    5. 確率 ≥ 70% かつ ADX ≥ 15（トレンド相場）でシグナル発火

    ── 特徴量（7次元）────────────────────────────────────────
      [0] RSI(9) / 100
      [1] Stochastic %K(5,3,3) / 100
      [2] MACD(5,13,5)ヒスト / ATR7（-3〜3 にクリップ）
      [3] ボリンジャー %B(14)（0=下限, 1=上限）
      [4] (close - EMA20) / ATR7（価格のEMA20からの乖離）
      [5] ボリューム比（直近 / 10本平均、0〜3 にクリップ）
      [6] ATR変化率（ATR7 / ATR14 - 1、ボラティリティ加速度）

    ── 出力 ──────────────────────────────────────────────────
      ラベル: "AI↑73%" / "AI↓81%" など
      色: Long=#00bcd4（シアン）/ Short=#ce93d8（パープル）
    """
    if len(df) < 40:
        return []

    p = params or {}
    FORWARD        = p.get("forward",        5)
    K              = p.get("k",             15)
    MIN_PROB       = p.get("min_prob",     0.70)
    COOLDOWN       = p.get("cooldown",       4)
    ADX_MIN        = p.get("adx_min",       15)
    ATR_MULT       = p.get("atr_mult",     1.5)
    RECENCY_DECAY  = p.get("recency_decay", 2.0)  # 直近重み指数減衰の強さ

    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    ts     = df.index
    n      = len(df)

    # ---- 各種インジ計算 ----
    cl = pd.Series(close)

    # RSI(9)
    _d   = cl.diff()
    rsi9 = (100 - 100 / (1 + _d.clip(lower=0).rolling(9).mean()
             / (-_d.clip(upper=0)).rolling(9).mean().replace(0, float("nan")))).values

    # Stochastic %K(5,3,3)
    _rk     = 100 * (cl - cl.rolling(5).min()) / (
                  pd.Series(high).rolling(5).max() - pd.Series(low).rolling(5).min()
              ).replace(0, float("nan"))
    stoch_k = _rk.rolling(3).mean().values

    # MACD(5,13,5) ヒストグラム
    _macd = cl.ewm(span=5, adjust=False).mean() - cl.ewm(span=13, adjust=False).mean()
    hist5 = (_macd - _macd.ewm(span=5, adjust=False).mean()).values

    # ATR7 / ATR14
    _tr    = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:]  - close[:-1]),
    ])
    _tr_s  = np.concatenate([[float("nan")], _tr])
    atr7   = pd.Series(_tr_s).rolling(7).mean().values
    atr14  = pd.Series(_tr_s).rolling(14).mean().values

    # Bollinger %B(14)
    bb_mid = cl.rolling(14).mean()
    bb_std = cl.rolling(14).std()
    bb_pct = ((cl - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, float("nan"))).values

    # EMA20
    ema20  = cl.ewm(span=20, adjust=False).mean().values

    # Volume ratio
    vol_ma10 = pd.Series(volume).rolling(10).mean().values

    # ADX(10) ── 簡易版
    _pdm = np.zeros(n); _mdm = np.zeros(n); _tr2 = np.zeros(n)
    for j in range(1, n):
        hd = high[j] - high[j-1]; ld = low[j-1] - low[j]
        _pdm[j] = hd if hd > ld and hd > 0 else 0.0
        _mdm[j] = ld if ld > hd and ld > 0 else 0.0
        _tr2[j]  = max(high[j]-low[j], abs(high[j]-close[j-1]), abs(low[j]-close[j-1]))
    _a   = 1.0 / 10
    _idx = cl.index
    _str  = pd.Series(_tr2, index=_idx).ewm(alpha=_a, adjust=False).mean().values
    _pdi  = 100 * pd.Series(_pdm, index=_idx).ewm(alpha=_a, adjust=False).mean().values / np.where(_str > 0, _str, float("nan"))
    _mdi  = 100 * pd.Series(_mdm, index=_idx).ewm(alpha=_a, adjust=False).mean().values / np.where(_str > 0, _str, float("nan"))
    _dx   = 100 * np.abs(_pdi - _mdi) / np.where((_pdi + _mdi) > 0, _pdi + _mdi, float("nan"))
    adx   = pd.Series(_dx, index=_idx).ewm(alpha=_a, adjust=False).mean().values

    # ---- 特徴量行列 ----
    feat = np.full((n, 7), float("nan"))
    for i in range(n):
        a7 = atr7[i] if (not np.isnan(atr7[i]) and atr7[i] > 0) else float("nan")
        a14 = atr14[i] if (not np.isnan(atr14[i]) and atr14[i] > 0) else float("nan")
        if any(np.isnan(x) for x in [rsi9[i], stoch_k[i], hist5[i],
                                      bb_pct[i], ema20[i], vol_ma10[i]]) or np.isnan(a7):
            continue
        feat[i, 0] = np.clip(rsi9[i] / 100.0, 0, 1)
        feat[i, 1] = np.clip(stoch_k[i] / 100.0, 0, 1)
        feat[i, 2] = np.clip(hist5[i] / a7, -3, 3) / 3.0   # -1〜1 に正規化
        feat[i, 3] = np.clip(bb_pct[i], 0, 1)
        feat[i, 4] = np.clip((close[i] - ema20[i]) / a7, -3, 3) / 3.0
        feat[i, 5] = np.clip(volume[i] / vol_ma10[i] if vol_ma10[i] > 0 else 1.0, 0, 3) / 3.0
        feat[i, 6] = np.clip((a7 / a14) - 1.0, -1, 1) if not np.isnan(a14) else 0.0

    # ---- ラベリング（ルックアヘッドなし: バーiのラベルは i+1〜i+FORWARD の未来） ----
    # label[i]: 1=LONG, -1=SHORT, 0=中立
    labels = np.zeros(n, dtype=int)
    for i in range(n - FORWARD):
        a7 = atr7[i]
        if np.isnan(a7) or a7 <= 0:
            continue
        fwd_high = np.max(high[i+1 : i+FORWARD+1])
        fwd_low  = np.min(low[i+1  : i+FORWARD+1])
        gain = fwd_high - close[i]
        loss = close[i] - fwd_low
        threshold = ATR_MULT * a7
        if gain >= threshold and gain >= 1.5 * loss:
            labels[i] = 1
        elif loss >= threshold and loss >= 1.5 * gain:
            labels[i] = -1
        # else 0 (中立) → k-NN の参照には使わない

    # ---- 相場環境バケット（ADX × ボラティリティ）----
    # 現在のバーと似た相場環境の参照バーを優先する
    atr7_valid = atr7[~np.isnan(atr7)]
    if len(atr7_valid) >= 3:
        p33 = float(np.percentile(atr7_valid, 33))
        p67 = float(np.percentile(atr7_valid, 67))
    else:
        p33 = p67 = float("nan")

    def _regime(i: int) -> int:
        """0=低ADX/低ボラ, 1〜8=ADX×ボラの組み合わせ"""
        adx_b = 0 if (np.isnan(adx[i]) or adx[i] < 20) else (1 if adx[i] < 35 else 2)
        if np.isnan(p33):
            vol_b = 1
        else:
            vol_b = 0 if atr7[i] < p33 else (1 if atr7[i] < p67 else 2)
        return adx_b * 3 + vol_b

    # ---- k-NN 予測 ----
    markers: list[dict] = []
    last_long_idx  = -COOLDOWN - 1
    last_short_idx = -COOLDOWN - 1

    # 有効な参照バーのインデックス（ラベルが0でなく、特徴量が有効なもの）
    valid_refs = np.array([
        i for i in range(n)
        if labels[i] != 0 and not np.any(np.isnan(feat[i]))
    ])
    valid_regimes = np.array([_regime(i) for i in valid_refs]) if len(valid_refs) > 0 else np.array([], dtype=int)

    for j in range(35, n - 1):
        if np.any(np.isnan(feat[j])):
            continue
        if np.isnan(adx[j]) or adx[j] < ADX_MIN:
            continue

        # 参照候補: ルックアヘッドバイアス回避
        mask  = valid_refs < j - FORWARD
        cands = valid_refs[mask]
        if len(cands) < K:
            continue

        # 相場環境マッチング: 同じ環境バケットの参照を優先
        # 2K 以上あるときのみ適用（少ないと Long/Short の偏りが大きくなるため）
        cur_regime   = _regime(j)
        regime_mask  = valid_regimes[mask] == cur_regime
        regime_cands = cands[regime_mask]
        cands_use    = regime_cands if len(regime_cands) >= K * 2 else cands

        # ユークリッド距離
        diffs = feat[cands_use] - feat[j]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))

        top_idx    = np.argsort(dists)[:K]
        top_dists  = dists[top_idx]
        top_labels = labels[cands_use[top_idx]]
        top_pos    = cands_use[top_idx]  # 参照バーの実際のインデックス（直近重みに使用）

        # 距離加重 × 直近重み（近いバーほど重視）
        dist_w    = 1.0 / (top_dists + 1e-8)
        recency_w = np.exp(-RECENCY_DECAY * (j - top_pos) / n)
        weights   = dist_w * recency_w

        long_w  = float(np.sum(weights[top_labels == 1]))
        short_w = float(np.sum(weights[top_labels == -1]))

        # extra_labels（過去の正解シグナル）を投票に追加
        # 距離 1.0 以下のパターンのみ使用し、k-NN 合計の30%以内に収める
        if extra_labels:
            base_total  = long_w + short_w
            extra_long  = 0.0
            extra_short = 0.0
            for ex in extra_labels:
                ex_feat = ex["feat"]
                if np.any(np.isnan(ex_feat)):
                    continue
                d = float(np.sqrt(np.sum((ex_feat - feat[j]) ** 2)))
                if d > 1.0:           # 類似度が低いものは無視
                    continue
                w = ex["weight"] * (1.0 - d)   # 距離が遠いほど弱く（線形減衰）
                if ex["label"] == 1:
                    extra_long  += w
                else:
                    extra_short += w
            # k-NN 票全体の30%を上限として追加（爆発防止）
            extra_sum = extra_long + extra_short
            if base_total > 0 and extra_sum > 0:
                scale    = min(1.0, 0.3 * base_total / extra_sum)
                long_w  += extra_long  * scale
                short_w += extra_short * scale

        total_w = long_w + short_w
        if total_w <= 0:
            continue

        long_prob  = long_w  / total_w
        short_prob = short_w / total_w

        if long_prob >= MIN_PROB and j - last_long_idx > COOLDOWN:
            pct = int(round(long_prob * 100))
            markers.append({
                "time":     int(ts[j].timestamp()) + jst_offset,
                "position": "belowBar",
                "color":    "#00bcd4",
                "shape":    "arrowUp",
                "text":     f"AI↑{pct}%",
            })
            last_long_idx = j

        elif short_prob >= MIN_PROB and j - last_short_idx > COOLDOWN:
            pct = int(round(short_prob * 100))
            markers.append({
                "time":     int(ts[j].timestamp()) + jst_offset,
                "position": "aboveBar",
                "color":    "#ce93d8",
                "shape":    "arrowDown",
                "text":     f"AI↓{pct}%",
            })
            last_short_idx = j

    return markers


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP（出来高加重平均価格）を計算する。

    スキャルパーが「公平価値」を判断するための基準線。
    毎日セッション開始でリセット（日付が変わるタイミング）。
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol     = df["volume"]
    # 日付ごとにリセット
    dates   = df.index.date
    vwap    = typical.copy() * float("nan")
    cum_tp_vol = 0.0
    cum_vol    = 0.0
    prev_date  = None
    for i, (tp, v, d) in enumerate(zip(typical.values, vol.values, dates)):
        if d != prev_date:
            cum_tp_vol = 0.0
            cum_vol    = 0.0
            prev_date  = d
        cum_tp_vol += tp * v
        cum_vol    += v
        vwap.iloc[i] = cum_tp_vol / cum_vol if cum_vol > 0 else tp
    return vwap


def calc_confirmation_signal(
    entry_markers: list[dict],
    ai_markers: list[dict],
    tolerance_secs: int = 300,
) -> list[dict]:
    """
    エントリーシグナルと AI シグナルが同方向に一致したバーに
    ゴールドの星マーカーを追加する。

    Parameters
    ----------
    entry_markers : エントリーシグナルのマーカーリスト（arrowUp / arrowDown）
    ai_markers    : AI シグナルのマーカーリスト（arrowUp / arrowDown）
    tolerance_secs: 同一バーとみなす時間差（秒）。デフォルト300秒＝5分足1本分

    Returns
    -------
    確認シグナルマーカーのリスト
    """
    if not entry_markers or not ai_markers:
        return []

    # エントリーマーカーを {time: direction(1=Long/-1=Short)} に変換
    entry_dir: dict[int, int] = {}
    for m in entry_markers:
        d = 1 if m.get("shape") == "arrowUp" else -1
        entry_dir[m["time"]] = d

    confirm: list[dict] = []
    for am in ai_markers:
        ai_time = am["time"]
        ai_d    = 1 if am.get("shape") == "arrowUp" else -1

        # tolerance_secs 以内にエントリーシグナルが一致するか検索
        for et, ed in entry_dir.items():
            if abs(ai_time - et) <= tolerance_secs and ai_d == ed:
                # 一致: 代表タイムスタンプとして AI 側のタイムを使用
                if ai_d == 1:
                    confirm.append({
                        "time":     ai_time,
                        "position": "belowBar",
                        "color":    "#ffd700",
                        "shape":    "circle",
                        "text":     "★",
                    })
                else:
                    confirm.append({
                        "time":     ai_time,
                        "position": "aboveBar",
                        "color":    "#ffd700",
                        "shape":    "circle",
                        "text":     "★",
                    })
                break  # 同じ AI マーカーに対して重複しない

    return confirm


def calc_cvd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    累積出来高デルタ (CVD: Cumulative Volume Delta)

    ティックデータがない OHLCV から各バーの買い/売り出来高比を推定する。

    推定式:
        delta = volume × (2×close − high − low) / (high − low)
    close が high に近いほど買いが多く、low に近いほど売りが多いと仮定。
    high == low のバー（ローソクが横一直線）はデルタを 0 とする。

    Returns:
        delta    : バーごとのデルタ（+= 買い優勢、−= 売り優勢）
        cvd_cum  : デルタの累積和（方向性トレンド把握用）
    """
    hl = (df["high"] - df["low"]).replace(0.0, float("nan"))
    delta = df["volume"] * (2 * df["close"] - df["high"] - df["low"]) / hl
    delta = delta.fillna(0.0)
    return delta, delta.cumsum()


def to_line_data(series: pd.Series, timestamps, jst_offset: int, decimals: int = 5) -> list[dict]:
    """Series → [{time, value}, ...] 変換。NaNは除外。"""
    result = []
    for ts, val in zip(timestamps, series):
        if pd.isna(val):
            continue
        result.append({
            "time":  int(ts.timestamp()) + jst_offset,
            "value": round(float(val), decimals),
        })
    return result


def calculate(df: pd.DataFrame, selected: list[str], jst_offset: int) -> dict:
    """
    選択されたインジケーターを計算して辞書で返す。

    Returns:
        dict: キー=インジケーターID, 値={"type": ..., "data": [...], ...}
    """
    result: dict = {}
    ts = df.index

    for ind in selected:
        if ind.startswith("SMA "):
            period = int(ind.split()[1])
            result[ind] = {
                "type":  "overlay",
                "color": INDICATOR_COLORS.get(ind, "#ffffff"),
                "data":  to_line_data(calc_sma(df, period), ts, jst_offset),
            }

        elif ind.startswith("EMA "):
            period = int(ind.split()[1])
            result[ind] = {
                "type":      "overlay",
                "color":     INDICATOR_COLORS.get(ind, "#aaaaaa"),
                "lineStyle": 1,  # dotted
                "data":      to_line_data(calc_ema(df, period), ts, jst_offset),
            }

        elif ind == "ボリンジャーバンド (20)":
            bands = calc_bollinger(df)
            result["BB_mid"] = {
                "type": "overlay", "color": "#9e9e9e", "lineStyle": 2,
                "data": to_line_data(bands["mid"], ts, jst_offset),
            }
            result["BB_upper_1"] = {
                "type": "overlay", "color": "#b39ddb", "lineStyle": 0,
                "data": to_line_data(bands["upper_1"], ts, jst_offset),
            }
            result["BB_lower_1"] = {
                "type": "overlay", "color": "#b39ddb", "lineStyle": 0,
                "data": to_line_data(bands["lower_1"], ts, jst_offset),
            }
            result["BB_upper_2"] = {
                "type": "overlay", "color": "#7986cb", "lineStyle": 0,
                "data": to_line_data(bands["upper_2"], ts, jst_offset),
            }
            result["BB_lower_2"] = {
                "type": "overlay", "color": "#7986cb", "lineStyle": 0,
                "data": to_line_data(bands["lower_2"], ts, jst_offset),
            }
            result["BB_upper_3"] = {
                "type": "overlay", "color": "#3f51b5", "lineStyle": 0,
                "data": to_line_data(bands["upper_3"], ts, jst_offset),
            }
            result["BB_lower_3"] = {
                "type": "overlay", "color": "#3f51b5", "lineStyle": 0,
                "data": to_line_data(bands["lower_3"], ts, jst_offset),
            }

        elif ind == "RSI (14)":
            result["RSI"] = {
                "type":  "sub_rsi",
                "color": "#7e57c2",
                "data":  to_line_data(calc_rsi(df), ts, jst_offset, decimals=2),
            }

        elif ind == "MACD (12, 26, 9)":
            macd, sig, hist = calc_macd(df)
            result["MACD"]        = {"type": "sub_macd", "color": "#2196f3",
                                      "data": to_line_data(macd, ts, jst_offset)}
            result["MACD_signal"] = {"type": "sub_macd", "color": "#ff9800",
                                      "data": to_line_data(sig,  ts, jst_offset)}
            result["MACD_hist"]   = {
                "type": "sub_macd_hist",
                "data": [
                    {
                        "time":  int(t.timestamp()) + jst_offset,
                        "value": round(float(v), 5),
                        "color": "#26a69a" if v >= 0 else "#ef5350",
                    }
                    for t, v in zip(ts, hist) if not pd.isna(v)
                ],
            }

        elif ind == "レジサポライン":
            result["SR_lines"] = {
                "type": "sr",
                "data": calc_sr_lines(df),
            }

        elif ind == "直近高値/安値":
            result["Recent_HL"] = {
                "type": "recent_hl",
                "data": calc_recent_hl(df),
            }

        elif ind == "ストキャスティクス (5,3,3)":
            k, d = calc_stochastic(df)
            result["Stoch_K"] = {
                "type": "sub_stoch", "color": "#26c6da",
                "data": to_line_data(k, ts, jst_offset, decimals=2),
            }
            result["Stoch_D"] = {
                "type": "sub_stoch", "color": "#ff7043",
                "data": to_line_data(d, ts, jst_offset, decimals=2),
            }

        elif ind == "ピボットポイント":
            result["Pivot_lines"] = {
                "type": "pivot",
                "data": calc_pivot_points(df),
            }

        elif ind == "セッション区切り":
            result["Session_markers"] = {
                "type": "session",
                "data": calc_session_markers(df, jst_offset),
            }

        elif ind == "ダイバージェンス (RSI)":
            result["Divergence_markers"] = {
                "type": "divergence",
                "data": calc_divergence(df, jst_offset),
            }

        elif ind == "ローソク足パターン":
            result["Pattern_markers"] = {
                "type": "pattern",
                "data": calc_candlestick_patterns(df, jst_offset),
            }

        elif ind == "ZigZag（転換点予測）":
            zz = calc_zigzag(df, jst_offset)
            result["ZigZag_line"] = {
                "type": "zigzag",
                "color": "#ffeb3b",
                "data": zz["line"],
            }

        elif ind == "エントリーシグナル":
            result["Entry_markers"] = {
                "type": "entry",
                "data": calc_entry_signals(df, jst_offset),
            }

        elif ind == "AI シグナル (k-NN)":
            result["AI_markers"] = {
                "type": "ai",
                "data": calc_ai_signal(df, jst_offset),
            }

        elif ind == "VWAP":
            result["VWAP"] = {
                "type":      "overlay",
                "color":     "#ff9800",
                "lineStyle": 1,   # dashed
                "lineWidth": 2,
                "data":      to_line_data(calc_vwap(df), ts, jst_offset),
            }

        elif ind == "累積出来高デルタ (CVD)":
            delta, cvd_cum = calc_cvd(df)
            # バーごとのデルタをヒストグラムで表示（0軸基準、緑=買い優勢・赤=売り優勢）
            result["CVD"] = {
                "type": "sub_cvd",
                "data": [
                    {
                        "time":  int(t.timestamp()) + jst_offset,
                        "value": round(float(v), 2),
                        "color": "#26a69a" if v >= 0 else "#ef5350",
                    }
                    for t, v in zip(ts, delta) if not pd.isna(v)
                ],
            }
            # 累積CVDをラインで重ねて方向性トレンドを表示
            result["CVD_line"] = {
                "type": "sub_cvd_line",
                "data": to_line_data(cvd_cum, ts, jst_offset, decimals=2),
            }

    return result
