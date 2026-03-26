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

# meta パラメータキー（strategy_params から除外するキー）
_META_KEYS = ("rr", "session", "max_bars_in_trade", "chandelier_mult", "pullback_atr_mult")

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


# ============================================================
# 3段階ドローダウン回路遮断器（DD Circuit Breaker）
# Bailey & Lopez de Prado / Elder 6%月次ルール に基づく
# ============================================================

def get_drawdown_gate(
    current_equity: float,
    peak_equity: float,
    base_risk_pct: float = 1.0,
) -> dict:
    """
    ピーク比ドローダウンに応じたリスク乗数と取引可否を返す。

    3段階プロトコル（v2研究 Topic B より）:
      DD < 5%  : リスク通常（乗数 1.0）
      DD 5〜10%: リスク 25% 削減（乗数 0.75）
      DD 10〜15%: リスク 50% 削減（乗数 0.50）
      DD > 15% : 取引停止（乗数 0.0）

    Args:
        current_equity: 現在の口座残高（円）
        peak_equity:    ピーク口座残高（円）
        base_risk_pct:  通常時のリスク率（%）

    Returns:
        {
            "risk_multiplier":  float,   # 0.0〜1.0
            "effective_risk":   float,   # base_risk_pct × multiplier（%）
            "dd_pct":           float,   # 現在のDD率（%）
            "gate":             str,     # "open" | "reduced_25" | "reduced_50" | "closed"
            "message":          str,
        }
    """
    if peak_equity <= 0:
        return {
            "risk_multiplier": 1.0,
            "effective_risk":  base_risk_pct,
            "dd_pct":          0.0,
            "gate":            "open",
            "message":         "通常",
        }

    dd_pct = max(0.0, (peak_equity - current_equity) / peak_equity * 100.0)

    if dd_pct >= 15.0:
        return {
            "risk_multiplier": 0.0,
            "effective_risk":  0.0,
            "dd_pct":          round(dd_pct, 2),
            "gate":            "closed",
            "message":         f"取引停止: DD {dd_pct:.1f}% > 15%。全ポジションを点検してください。",
        }
    elif dd_pct >= 10.0:
        mult = 0.50
        return {
            "risk_multiplier": mult,
            "effective_risk":  round(base_risk_pct * mult, 3),
            "dd_pct":          round(dd_pct, 2),
            "gate":            "reduced_50",
            "message":         f"リスク50%削減中: DD {dd_pct:.1f}%（閾値10%）",
        }
    elif dd_pct >= 5.0:
        mult = 0.75
        return {
            "risk_multiplier": mult,
            "effective_risk":  round(base_risk_pct * mult, 3),
            "dd_pct":          round(dd_pct, 2),
            "gate":            "reduced_25",
            "message":         f"リスク25%削減中: DD {dd_pct:.1f}%（閾値5%）",
        }
    else:
        return {
            "risk_multiplier": 1.0,
            "effective_risk":  base_risk_pct,
            "dd_pct":          round(dd_pct, 2),
            "gate":            "open",
            "message":         "通常",
        }


def _get_htf_trend(symbol: str, base_tf: str) -> str:
    """上位TF の EMA200 方向 + RSI50 確認を返す: 'up' | 'down' | 'neutral'"""
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
        # RSI(14) 50ライン確認（MTF研究: RSI50フィルターで追加勝率向上）
        delta   = df["close"].diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rsi14   = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])
        if slope > 0 and price > ema_v and rsi14 > 50:
            return "up"
        elif slope < 0 and price < ema_v and rsi14 < 50:
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
    rr                = params.get("rr", rr)
    session           = params.get("session", "all")
    max_bars_in_trade = params.get("max_bars_in_trade", 0)
    chandelier_mult   = params.get("chandelier_mult", 0.0)
    pullback_atr_mult = params.get("pullback_atr_mult", 0.0)
    trade_hours       = _SESSION_HOURS_JST.get(session, None)
    strategy_params   = {k: v for k, v in params.items() if k not in _META_KEYS}

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
            hurst_filter=True,
            max_bars_in_trade=max_bars_in_trade,
            chandelier_mult=chandelier_mult,
            pullback_atr_mult=pullback_atr_mult,
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

    # ---- ハーストレジームフィルター（ライブ） ----
    try:
        from dashboard.backtest_engine import _hurst as _bt_hurst
        _TREND_ONLY = {
            "EMAクロス", "ドンチャンブレイクアウト", "トリプル確認(EMA+RSI+MACD)",
            "ロンドンブレイクアウト", "ICT_FVGスキャルパー",
        }
        _MR_ONLY    = {"RSI×BB"}
        _hurst_val  = float(_bt_hurst(df["close"]).iloc[-1])
        if strategy in _TREND_ONLY and _hurst_val < 0.45:
            return {**neutral, "direction": "neutral", "strategy": strategy,
                    "strategy_params": params, "oos_pf": tune.get("oos_pf", 0.0)}
        if strategy in _MR_ONLY and _hurst_val > 0.55:
            return {**neutral, "direction": "neutral", "strategy": strategy,
                    "strategy_params": params, "oos_pf": tune.get("oos_pf", 0.0)}
    except Exception:
        pass

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

    # ---------- ボラティリティターゲティング推奨ロット ----------
    # 現在ATR / 通常ATR(50本平均) の比でロットをスケール調整
    # 高ボラ時はリスクを自動削減、低ボラ時は上限1.0に制限
    _atr50_v   = float(np.mean(tr[-50:])) if len(tr) >= 50 else atr14
    _vol_ratio  = atr14 / _atr50_v if _atr50_v > 0 else 1.0
    _vol_scalar = min(1.0, 1.0 / _vol_ratio)   # 高ボラ → ロット削減
    _risk_jpy   = 500_000 * 0.01                # 基準リスク 5000円（1%）
    _lot_units  = (_risk_jpy * _vol_scalar) / (sl_pips * pip * 10_000) if sl_pips > 0 else 0.0
    _lot_man    = round(_lot_units, 2)

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
    rr                = params.get("rr", rr)
    session           = params.get("session", "all")
    max_bars_in_trade = params.get("max_bars_in_trade", 0)
    chandelier_mult   = params.get("chandelier_mult", 0.0)
    pullback_atr_mult = params.get("pullback_atr_mult", 0.0)
    trade_hours       = _SESSION_HOURS_JST.get(session, None)
    strategy_params   = {k: v for k, v in params.items() if k not in _META_KEYS}

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
            hurst_filter=True,
            max_bars_in_trade=max_bars_in_trade,
            chandelier_mult=chandelier_mult,
            pullback_atr_mult=pullback_atr_mult,
        )
        signals = generate_signals(df, bt_p)
        trades  = execute_trades(df, signals, bt_p)
    except Exception:
        return []

    markers: list[dict] = []

    for t in trades:
        entry_ts = int(t.entry_time.timestamp()) + jst_offset
        exit_ts  = int(t.exit_time.timestamp())  + jst_offset

        # ---- エントリーマーカー（矢印のみ・テキストなしで視認性向上）----
        if t.direction == "long":
            markers.append({
                "time":     entry_ts,
                "position": "belowBar",
                "color":    "#26a69a",
                "shape":    "arrowUp",
                "text":     "",
                "size":     1,
            })
        else:
            markers.append({
                "time":     entry_ts,
                "position": "aboveBar",
                "color":    "#ef5350",
                "shape":    "arrowDown",
                "text":     "",
                "size":     1,
            })

        # ---- エグジットマーカー（期末クローズは除外）----
        if t.exit_reason == "end":
            continue

        # pips を整数で表示（小数は不要なノイズ）
        pnl_str = f"{t.pnl_pips:+.0f}p"

        if t.exit_reason == "sl":
            markers.append({
                "time":     exit_ts,
                "position": "aboveBar" if t.direction == "long" else "belowBar",
                "color":    "#ef5350",
                "shape":    "square",
                "text":     pnl_str,
                "size":     1,
            })
        elif t.exit_reason == "tp":
            markers.append({
                "time":     exit_ts,
                "position": "aboveBar" if t.direction == "long" else "belowBar",
                "color":    "#26a69a",
                "shape":    "circle",
                "text":     pnl_str,
                "size":     1,
            })
        elif t.exit_reason == "chandelier":
            markers.append({
                "time":     exit_ts,
                "position": "aboveBar" if t.direction == "long" else "belowBar",
                "color":    "#26a69a",
                "shape":    "circle",
                "text":     pnl_str,
                "size":     1,
            })
        else:  # signal（逆シグナルによる決済）
            markers.append({
                "time":     exit_ts,
                "position": "aboveBar" if t.direction == "long" else "belowBar",
                "color":    "#ffeb3b",
                "shape":    "circle",
                "text":     pnl_str,
                "size":     1,
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

    rr                = params.get("rr", rr)
    session           = params.get("session", "all")
    max_bars_in_trade = params.get("max_bars_in_trade", 0)
    chandelier_mult   = params.get("chandelier_mult", 0.0)
    pullback_atr_mult = params.get("pullback_atr_mult", 0.0)
    trade_hours       = _SESSION_HOURS_JST.get(session, None)
    strategy_params   = {k: v for k, v in params.items() if k not in _META_KEYS}

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
            hurst_filter=True,
            max_bars_in_trade=max_bars_in_trade,
            chandelier_mult=chandelier_mult,
            pullback_atr_mult=pullback_atr_mult,
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
