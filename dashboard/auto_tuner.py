"""
dashboard/auto_tuner.py

バックグラウンド自動チューニング。
- 5戦略の縮小グリッドで IS/OOS 最適化を実行
- ローリング3窓OOS検証（汎化性能を強化）
- ADXフィルター固定適用（ADX<15のレンジ相場排除）
- 戦略別RR・セッションを自動最適化
- 結果を output/auto_tune_{symbol}_{timeframe}.json にキャッシュ（24時間有効）
- get_tune_result() で最良戦略＋パラメータを取得
"""

from __future__ import annotations

import itertools
import json
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT_DIR  = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "output"
CACHE_TTL = 24 * 3600  # 24h

# ============================================================
# セッション → JST時間帯マッピング（UTC+9）
#   ロンドン      : UTC 08-17 = JST 17-02
#   ロンドン-NY重複: UTC 13-17 = JST 22-02
# ============================================================
SESSION_HOURS_JST: dict[str, list[int] | None] = {
    "all":     None,
    "london":  [17, 18, 19, 20, 21, 22, 23, 0, 1],
    "overlap": [22, 23, 0, 1],
}

# ============================================================
# 自動チューニング用戦略・パラメータグリッド
# メタパラメータ（rr / session）を各戦略に追加
# ADX フィルター（adx_min=15）は _run_bt_on_df 内で固定適用
# ============================================================
AUTO_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "EMAクロス": {
        "short_period":      [8, 13, 21],
        "long_period":       [34, 55, 89],
        "rr":                [2.0, 2.5],
        "session":           ["all", "overlap"],
        "chandelier_mult":   [0.0, 3.0],
        "pullback_atr_mult": [0.0, 1.5],
    },
    "ドンチャンブレイクアウト": {
        "period":          [20, 40, 55],
        "rr":              [2.0, 2.5],
        "session":         ["all", "overlap"],
        "chandelier_mult": [0.0, 3.0],
    },
    "トリプル確認(EMA+RSI+MACD)": {
        "ema_period":        [50, 100],
        "rsi_period":        [9, 14],
        "rr":                [2.0, 2.5],
        "session":           ["all", "overlap"],
        "chandelier_mult":   [0.0, 3.0],
        "pullback_atr_mult": [0.0, 1.5],
    },
    "RSI×BB": {
        "rsi_period":        [9, 14],
        "oversold":          [25, 30],
        "overbought":        [70, 75],
        "bb_std":            [2.0],
        "rr":                [1.5, 2.0],   # 平均回帰は低RRで勝率優先
        "max_bars_in_trade": [0, 12, 24],  # タイムベースエグジット（平均回帰に有効）
    },
    "夜間スカルパー(4重確認)": {
        "fast_ema":          [8, 13],
        "slow_ema":          [21, 34],
        "rsi_period":        [7],
        "k_period":          [5],
        "atr_multiplier":    [0.8, 1.0],
        "rr":                [1.5, 2.0],
        "max_bars_in_trade": [0, 8, 16],   # スキャルプは短い保有が原則
    },
}

# ATR SL 倍率（固定）
_ATR_SL_MULT = 1.5


def _cache_path(symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"auto_tune_{symbol}_{timeframe}.json"


def is_cache_fresh(symbol: str, timeframe: str) -> bool:
    """キャッシュが 24 時間以内であれば True を返す。"""
    p = _cache_path(symbol, timeframe)
    if not p.exists():
        return False
    return (_time.time() - p.stat().st_mtime) < CACHE_TTL


def get_tune_result(symbol: str, timeframe: str) -> dict | None:
    """キャッシュから最良戦略を読む。なければ None。"""
    p = _cache_path(symbol, timeframe)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_bt_on_df(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    spread_pips: float = 1.0,
    lot_size: int = 10_000,
) -> dict | None:
    """
    既存の DataFrame スライスで直接バックテストを実行するヘルパー。
    load_ohlcv を経由しないので MT5 接続が不要。

    params にはメタパラメータ（rr / session）が含まれる場合があり、
    ここで抽出して BacktestParams に反映する。
    """
    from dashboard.backtest_engine import (
        BacktestParams, generate_signals, execute_trades, calc_metrics,
    )

    if len(df) < 30:
        return None

    # メタパラメータを抽出（戦略固有パラメータとは分離）
    rr                 = params.get("rr", 2.0)
    session            = params.get("session", "all")
    max_bars_in_trade  = params.get("max_bars_in_trade", 0)
    chandelier_mult    = params.get("chandelier_mult", 0.0)
    pullback_atr_mult  = params.get("pullback_atr_mult", 0.0)
    trade_hours        = SESSION_HOURS_JST.get(session, None)
    _ALL_META = ("rr", "session", "max_bars_in_trade", "chandelier_mult", "pullback_atr_mult")
    strategy_params    = {k: v for k, v in params.items() if k not in _ALL_META}

    bt_p = BacktestParams(
        symbol="DUMMY",
        timeframe="1H",
        start_date=df.index[0].to_pydatetime().replace(tzinfo=timezone.utc),
        end_date=df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
        strategy=strategy,
        strategy_params=strategy_params,
        direction="両方",
        spread_pips=spread_pips,
        lot_size=lot_size,
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

    try:
        sigs   = generate_signals(df, bt_p)
        trades = execute_trades(df, sigs, bt_p)
        if not trades:
            return None
        return calc_metrics(trades)
    except Exception:
        return None


def run_auto_tune(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict | None:
    """
    ローリング3窓 IS/OOS 最適化を実行して最良戦略を返す。

    改善点:
    - IS = 先頭60%（短縮してOOS窓を3つ確保）
    - OOS窓1 (13%): 汎化性能の第1確認
    - OOS窓2 (14%): 汎化性能の第2確認
    - OOS窓3 (13%): 最直近（最重要・重み40%）
    - 加重平均OOS PFで最良候補を選択（直近ほど重み大）
    - ADXフィルター（adx_min=15）固定適用
    - 戦略別RR・セッションを自動最適化

    Returns:
        {
            "strategy":    str,
            "params":      dict,   # rr / session を含む
            "oos_pf":      float,  # 加重平均OOS PF
            "oos_winrate": float,
            "oos_trades":  int,
            "is_score":    float,
            "tuned_at":    str,
        }
    """
    # ---------- データ取得 ----------
    if df is None or len(df) < 100:
        try:
            from dashboard.sample_data import get_ohlcv_dataframe
            df, _ = get_ohlcv_dataframe(symbol, timeframe, count=2000)
        except Exception:
            return None

    if len(df) < 100:
        return None

    # ---------- ローリング3窓 IS/OOS 分割 ----------
    # IS: 先頭60%  OOS窓1: 60〜73%  OOS窓2: 73〜87%  OOS窓3: 87〜100%
    n        = len(df)
    split_is = int(n * 0.60)
    split_o1 = int(n * 0.73)
    split_o2 = int(n * 0.87)

    is_df    = df.iloc[:split_is].copy()
    oos_dfs  = [
        df.iloc[split_is:split_o1].copy(),  # OOS窓1（重み25%）
        df.iloc[split_o1:split_o2].copy(),  # OOS窓2（重み35%）
        df.iloc[split_o2:].copy(),          # OOS窓3・最直近（重み40%）
    ]
    OOS_WEIGHTS = [0.25, 0.35, 0.40]

    # ---------- IS グリッドサーチ ----------
    total = sum(
        1
        for grid in AUTO_PARAM_GRIDS.values()
        for _ in itertools.product(*grid.values())
    )
    done    = 0
    results = []

    for strategy, grid in AUTO_PARAM_GRIDS.items():
        keys   = list(grid.keys())
        values = list(grid.values())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            done  += 1
            if progress_cb:
                progress_cb(done, total, strategy)

            m = _run_bt_on_df(is_df, strategy, params)
            if m is None or m["n_trades"] < 5 or m["total_pnl_pips"] <= 0:
                continue

            pf       = min(m["profit_factor"], 10.0) if m["profit_factor"] != float("inf") else 10.0
            wr       = m["win_rate"] / 100.0
            dd_ratio = abs(m["max_drawdown_jpy"]) / max(m["total_pnl_jpy"], 1.0)
            score    = pf * wr / (1.0 + dd_ratio)
            results.append({
                "strategy": strategy,
                "params":   params,
                "is_pf":    pf,
                "is_wr":    m["win_rate"],
                "score":    score,
            })

    if not results:
        return None

    # IS スコア上位 5 候補を 3窓OOS で検証
    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]

    best    = None
    best_pf = -1.0

    for cand in top5:
        pfs    = []
        m_last = None

        for oos_df in oos_dfs:
            m = _run_bt_on_df(oos_df, cand["strategy"], cand["params"])
            if m is not None and m["n_trades"] >= 2:
                pf = min(m["profit_factor"], 10.0) if m["profit_factor"] != float("inf") else 10.0
                pfs.append(pf)
                m_last = m
            else:
                pfs.append(None)

        # 有効窓が1つ以上あれば採用
        valid = [(w, p) for w, p in zip(OOS_WEIGHTS, pfs) if p is not None]
        if not valid:
            continue

        total_w    = sum(w for w, _ in valid)
        avg_oos_pf = sum(w * p for w, p in valid) / total_w

        if avg_oos_pf > best_pf:
            best_pf = avg_oos_pf
            best = {
                "strategy":    cand["strategy"],
                "params":      cand["params"],
                "oos_pf":      round(avg_oos_pf, 2),
                "oos_winrate": round(m_last["win_rate"], 1) if m_last else 0.0,
                "oos_trades":  m_last["n_trades"] if m_last else 0,
                "is_score":    round(cand["score"], 4),
                "tuned_at":    datetime.now(timezone.utc).isoformat(),
            }

    if best is None:
        # OOS 検証を通った候補なし → IS スコア最高を採用
        best = {
            "strategy":    top5[0]["strategy"],
            "params":      top5[0]["params"],
            "oos_pf":      0.0,
            "oos_winrate": 0.0,
            "oos_trades":  0,
            "is_score":    round(top5[0]["score"], 4),
            "tuned_at":    datetime.now(timezone.utc).isoformat(),
        }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(symbol, timeframe).write_text(
        json.dumps(best, ensure_ascii=False), encoding="utf-8"
    )
    return best
