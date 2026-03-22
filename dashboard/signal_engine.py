"""
dashboard/signal_engine.py

リアルタイムシグナルエンジン。
- 自動チューニング結果（最良戦略）を読み込む
- 上位タイムフレーム(EMA200)確認でフィルタリング
- エントリー価格・SL・TP を計算して返す
"""

from __future__ import annotations

from datetime import timezone as _tz

import numpy as np
import pandas as pd

# auto_tuner からセッションマッピングを共有
from dashboard.auto_tuner import SESSION_HOURS_JST as _SESSION_HOURS_JST, _ATR_SL_MULT

# 上位TF マッピング（現在足 → 確認に使う上位足）
_HIGHER_TF: dict[str, str] = {
    "1M":  "15M",
    "5M":  "1H",
    "15M": "4H",
    "30M": "4H",
    "1H":  "1D",
    "4H":  "1W",
    "1D":  "1W",
}


def _pip_size(symbol: str) -> float:
    return 0.01 if symbol.endswith("JPY") else 0.0001


def _get_htf_trend(symbol: str, base_tf: str) -> str:
    """上位TF の EMA200 方向を返す: 'up' | 'down' | 'neutral'"""
    htf = _HIGHER_TF.get(base_tf)
    if not htf:
        return "neutral"
    try:
        from dashboard.sample_data import get_ohlcv_dataframe
        df, _ = get_ohlcv_dataframe(symbol, htf, count=250)
        if len(df) < 210:
            return "neutral"
        ema200 = df["close"].ewm(span=200, adjust=False).mean()
        slope  = float(ema200.iloc[-1]) - float(ema200.iloc[-5])
        price  = float(df["close"].iloc[-1])
        ema_v  = float(ema200.iloc[-1])
        if slope > 0 and price > ema_v:
            return "up"
        elif slope < 0 and price < ema_v:
            return "down"
    except Exception:
        pass
    return "neutral"


def get_live_signal(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    rr: float = 2.0,
    atr_sl_mult: float = 1.5,
    signal_max_age_bars: int = 3,
) -> dict:
    """
    現在のライブシグナルを返す。

    Returns:
        {
            "direction":       "long" | "short" | "neutral",
            "entry":           float,
            "sl":              float,
            "tp":              float,
            "sl_pips":         float,
            "tp_pips":         float,
            "rr":              float,
            "confidence":      "high" | "medium" | "low",
            "strategy":        str,
            "strategy_params": dict,
            "htf_trend":       "up" | "down" | "neutral",
            "htf_aligned":     bool,
            "signal_age":      int,
            "oos_pf":          float,
        }
    """
    from dashboard.auto_tuner import get_tune_result
    from dashboard.backtest_engine import BacktestParams, generate_signals

    pip = _pip_size(symbol)
    neutral: dict = {
        "direction": "neutral", "entry": 0.0, "sl": 0.0, "tp": 0.0,
        "sl_pips": 0.0, "tp_pips": 0.0, "rr": rr,
        "confidence": "low", "strategy": "未チューニング", "strategy_params": {},
        "htf_trend": "neutral", "htf_aligned": False, "signal_age": 999, "oos_pf": 0.0,
        "session": "all", "recommended_lot": 0.0,
    }

    if len(df) < 60:
        return neutral

    # ---------- 自動チューニング結果 ----------
    tune = get_tune_result(symbol, timeframe)
    if tune is None:
        return neutral

    strategy = tune["strategy"]
    params   = tune["params"]

    # チューニング済みメタパラメータを取得（なければデフォルト）
    rr          = params.get("rr", rr)
    session     = params.get("session", "all")
    trade_hours = _SESSION_HOURS_JST.get(session, None)
    strategy_params = {k: v for k, v in params.items() if k not in ("rr", "session")}

    # ---------- シグナル生成 ----------
    try:
        bt_p = BacktestParams(
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[0].to_pydatetime().replace(tzinfo=_tz.utc),
            end_date=df.index[-1].to_pydatetime().replace(tzinfo=_tz.utc),
            strategy=strategy,
            strategy_params=strategy_params,
            direction="両方",
            spread_pips=1.0,
            trade_hours=trade_hours,
            adx_min=15.0,
        )
        signals = generate_signals(df, bt_p)
    except Exception:
        return {**neutral, "strategy": strategy, "strategy_params": params,
                "oos_pf": tune.get("oos_pf", 0.0)}

    # 直近 N 本以内の最後のシグナルを探す
    direction  = "neutral"
    signal_age = 999
    for i in range(1, min(signal_max_age_bars + 1, len(signals))):
        sig = int(signals.iloc[-(i + 1)])
        if sig != 0:
            direction  = "long" if sig == 1 else "short"
            signal_age = i
            break

    if direction == "neutral":
        return {**neutral, "direction": "neutral", "strategy": strategy,
                "strategy_params": params, "oos_pf": tune.get("oos_pf", 0.0)}

    # ---------- 上位TF確認 ----------
    htf_trend   = _get_htf_trend(symbol, timeframe)
    htf_aligned = (
        (htf_trend == "up"   and direction == "long")  or
        (htf_trend == "down" and direction == "short") or
        htf_trend == "neutral"
    )

    # ---------- エントリー価格 ----------
    spread     = 1.0 * pip
    last_close = float(df["close"].iloc[-1])
    entry = last_close + spread if direction == "long" else last_close - spread

    # ---------- ATR ベース SL/TP ----------
    h  = df["high"].values
    lv = df["low"].values
    c  = df["close"].values
    tr = np.maximum.reduce([
        h[1:] - lv[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(lv[1:] - c[:-1]),
    ])
    atr14      = float(np.mean(tr[-14:])) if len(tr) >= 14 else pip * 10
    sl_dist    = atr14 * _ATR_SL_MULT
    tp_dist    = sl_dist * rr

    if direction == "long":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    sl_pips = sl_dist / pip
    tp_pips = tp_dist / pip

    # ---------- 信頼度 ----------
    oos_pf = tune.get("oos_pf", 0.0)
    if htf_aligned and oos_pf >= 1.2 and signal_age == 1:
        confidence = "high"
    elif htf_aligned and oos_pf >= 1.0:
        confidence = "medium"
    else:
        confidence = "low"

    dec = 3 if pip >= 0.01 else 5

    # ---------- ATRベース推奨ロット（1%リスク・口座50万円想定） ----------
    # sl_pips × pip_value × lot = risk_jpy
    # → lot(万通貨) = risk_jpy / (sl_pips × pip × 10000)
    _risk_jpy   = 500_000 * 0.01  # 5000円
    _lot_units  = _risk_jpy / (sl_pips * pip * 10_000) if sl_pips > 0 else 0.0
    _lot_man    = round(_lot_units, 2)  # 万通貨単位

    return {
        "direction":        direction,
        "entry":            round(entry, dec),
        "sl":               round(sl,    dec),
        "tp":               round(tp,    dec),
        "sl_pips":          round(sl_pips, 1),
        "tp_pips":          round(tp_pips, 1),
        "rr":               rr,
        "confidence":       confidence,
        "strategy":         strategy,
        "strategy_params":  params,
        "htf_trend":        htf_trend,
        "htf_aligned":      htf_aligned,
        "signal_age":       signal_age,
        "oos_pf":           oos_pf,
        "session":          session,
        "recommended_lot":  _lot_man,  # 1%リスク・50万円口座想定（万通貨）
    }


# ============================================================
# チャート用エントリー・エグジットマーカー生成
# ============================================================

def get_autotune_markers(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    jst_offset: int = 0,
    atr_sl_mult: float = 1.5,
    rr: float = 2.0,
) -> list[dict]:
    """
    auto-tuned 戦略でのエントリー・エグジットタイミングをチャートマーカー形式で返す。

    エントリー:
      Long  → 緑 arrowUp  belowBar  「▲ BUY  (戦略名)」
      Short → 赤 arrowDown aboveBar 「▼ SELL (戦略名)」

    エグジット:
      SL ヒット → 赤 square   「✕SL  +/-pips」
      TP ヒット → 緑 circle   「◎TP  +pips」
      シグナル逆転 → 黄 circle 「→ EXIT +/-pips」
      期末クローズ → 表示しない（ノイズになるため）

    Returns:
        LightweightCharts setMarkers 互換のリスト（時刻昇順）
    """
    from dashboard.auto_tuner import get_tune_result
    from dashboard.backtest_engine import (
        BacktestParams, generate_signals, execute_trades,
    )

    tune = get_tune_result(symbol, timeframe)
    if tune is None or len(df) < 60:
        return []

    strategy = tune["strategy"]
    params   = tune["params"]
    short_name = strategy[:8]

    # チューニング済みメタパラメータを適用
    rr          = params.get("rr", rr)
    session     = params.get("session", "all")
    trade_hours = _SESSION_HOURS_JST.get(session, None)
    strategy_params = {k: v for k, v in params.items() if k not in ("rr", "session")}

    try:
        bt_p = BacktestParams(
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[0].to_pydatetime().replace(tzinfo=_tz.utc),
            end_date=df.index[-1].to_pydatetime().replace(tzinfo=_tz.utc),
            strategy=strategy,
            strategy_params=strategy_params,
            direction="両方",
            spread_pips=1.0,
            trade_hours=trade_hours,
            sl_tp_type="atr",
            atr_sl_mult=_ATR_SL_MULT,
            atr_tp_mult=_ATR_SL_MULT * rr,
            adx_min=15.0,
        )
        signals = generate_signals(df, bt_p)
        trades  = execute_trades(df, signals, bt_p)
    except Exception:
        return []

    markers: list[dict] = []

    for t in trades:
        entry_ts = int(t.entry_time.timestamp()) + jst_offset
        exit_ts  = int(t.exit_time.timestamp())  + jst_offset

        # ---- エントリーマーカー ----
        if t.direction == "long":
            markers.append({
                "time":     entry_ts,
                "position": "belowBar",
                "color":    "#26a69a",
                "shape":    "arrowUp",
                "text":     f"▲BUY [{short_name}]",
            })
        else:
            markers.append({
                "time":     entry_ts,
                "position": "aboveBar",
                "color":    "#ef5350",
                "shape":    "arrowDown",
                "text":     f"▼SELL [{short_name}]",
            })

        # ---- エグジットマーカー（期末クローズは除外）----
        if t.exit_reason == "end":
            continue

        pnl_str = f"{t.pnl_pips:+.1f}p"

        if t.exit_reason == "sl":
            markers.append({
                "time":     exit_ts,
                "position": "aboveBar" if t.direction == "long" else "belowBar",
                "color":    "#ef5350",
                "shape":    "square",
                "text":     f"✕SL {pnl_str}",
            })
        elif t.exit_reason == "tp":
            markers.append({
                "time":     exit_ts,
                "position": "aboveBar" if t.direction == "long" else "belowBar",
                "color":    "#26a69a",
                "shape":    "circle",
                "text":     f"◎TP {pnl_str}",
            })
        else:  # signal（逆シグナルによる決済）
            markers.append({
                "time":     exit_ts,
                "position": "aboveBar" if t.direction == "long" else "belowBar",
                "color":    "#ffeb3b",
                "shape":    "circle",
                "text":     f"→EXIT {pnl_str}",
            })

    markers.sort(key=lambda x: x["time"])
    return markers


def get_autotune_summary(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    atr_sl_mult: float = 1.5,
    rr: float = 2.0,
) -> dict:
    """
    auto-tuned 戦略のバックテスト集計を返す。

    Returns:
        {
            "total_pips":   float,   # 合計損益（pips）
            "trade_count":  int,     # 総トレード数
            "win_count":    int,     # 勝ちトレード数
            "win_rate":     float,   # 勝率 0〜100
            "profit_factor": float,  # PF
            "strategy":     str,
        }
    """
    from dashboard.auto_tuner import get_tune_result
    from dashboard.backtest_engine import BacktestParams, generate_signals, execute_trades

    empty = {
        "total_pips": 0.0, "trade_count": 0, "win_count": 0,
        "win_rate": 0.0, "profit_factor": 0.0, "strategy": "―",
    }

    tune = get_tune_result(symbol, timeframe)
    if tune is None or len(df) < 60:
        return empty

    strategy = tune["strategy"]
    params   = tune["params"]

    rr          = params.get("rr", rr)
    session     = params.get("session", "all")
    trade_hours = _SESSION_HOURS_JST.get(session, None)
    strategy_params = {k: v for k, v in params.items() if k not in ("rr", "session")}

    try:
        bt_p = BacktestParams(
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[0].to_pydatetime().replace(tzinfo=_tz.utc),
            end_date=df.index[-1].to_pydatetime().replace(tzinfo=_tz.utc),
            strategy=strategy,
            strategy_params=strategy_params,
            direction="両方",
            spread_pips=1.0,
            trade_hours=trade_hours,
            sl_tp_type="atr",
            atr_sl_mult=_ATR_SL_MULT,
            atr_tp_mult=_ATR_SL_MULT * rr,
            adx_min=15.0,
        )
        signals = generate_signals(df, bt_p)
        trades  = execute_trades(df, signals, bt_p)
    except Exception:
        return {**empty, "strategy": strategy}

    if not trades:
        return {**empty, "strategy": strategy}

    pips_list  = [t.pnl_pips for t in trades]
    total_pips = sum(pips_list)
    win_count  = sum(1 for p in pips_list if p > 0)
    gross_win  = sum(p for p in pips_list if p > 0)
    gross_loss = sum(-p for p in pips_list if p < 0)
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    return {
        "total_pips":    round(total_pips, 1),
        "trade_count":   len(trades),
        "win_count":     win_count,
        "win_rate":      round(win_count / len(trades) * 100, 1),
        "profit_factor": pf,
        "strategy":      strategy,
    }
