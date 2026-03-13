"""
dashboard/optimizer.py

戦略パラメータのグリッドサーチによる最適化エンジン。
全戦略×全パラメータ組み合わせをバックテストし、成績順に返す。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Generator

import pandas as pd

from dashboard.backtest_engine import BacktestParams, BacktestResult, run_backtest

# ============================================================
# 時間帯プリセット（JST時）
# ============================================================

HOUR_PRESETS: dict[str, list[int] | None] = {
    "全時間":                    None,
    "東京時間  (JST  9:00-18:00)": list(range(9, 18)),
    "ロンドン時間 (JST 16:00-翌1:00)": list(range(16, 24)) + [0],
    "NY時間    (JST 21:00-翌6:00)": list(range(21, 24)) + list(range(0, 6)),
    "東京+ロンドン重複 (JST 16:00-18:00)": [16, 17],
    "ロンドン+NY重複  (JST 21:00-翌1:00)": [21, 22, 23, 0],
}

# ============================================================
# パラメータグリッド定義
# ============================================================

PARAM_GRIDS: dict[str, dict[str, list]] = {
    # ---- 既存5戦略 ----
    "SMAクロス": {
        "short_period": [5, 10, 15, 20],
        "long_period":  [25, 50, 75, 100],
    },
    "EMAクロス": {
        "short_period": [5, 10, 15, 20],
        "long_period":  [25, 50, 75, 100],
    },
    "RSI": {
        "period":     [7, 14, 21],
        "oversold":   [20, 25, 30],
        "overbought": [70, 75, 80],
    },
    "MACD": {
        "fast":   [8, 12, 16],
        "slow":   [21, 26, 34],
        "signal": [7, 9, 11],
    },
    "ボリンジャーバンド": {
        "period":  [10, 20, 30],
        "std_dev": [1.5, 2.0, 2.5],
    },
    # ---- 追加10戦略 ----
    "ストキャスティクス": {
        "k_period":   [9, 14, 21],
        "d_period":   [3, 5],
        "oversold":   [20, 25],
        "overbought": [75, 80],
    },
    "CCI": {
        "period":         [14, 20, 30],
        "buy_threshold":  [-100, -150],
        "sell_threshold": [100,  150],
    },
    "ウィリアムズ%R": {
        "period":     [10, 14, 21],
        "oversold":   [-80, -75],
        "overbought": [-25, -20],
    },
    "ドンチャンブレイクアウト": {
        "period": [10, 20, 40, 55],
    },
    "ATRブレイクアウト": {
        "atr_period": [10, 14, 21],
        "multiplier": [1.0, 1.5, 2.0],
    },
    "移動平均乖離率": {
        "period":    [20, 30, 50],
        "threshold": [0.5, 1.0, 1.5],
    },
    "MACDヒストグラム": {
        "fast":   [8, 12],
        "slow":   [21, 26],
        "signal": [7, 9],
    },
    "トリプルEMAクロス": {
        "fast": [5, 10],
        "mid":  [20, 30],
        "slow": [60, 100],
    },
    "ROC": {
        "period":    [9, 14, 21],
        "threshold": [0.3, 0.5, 0.8],
    },
    "RSIトレンド": {
        "period": [7, 9, 14, 21],
    },
    # ---- 複合オリジナルインジケーター ----
    "RSI×MACDクロス": {
        "rsi_period":  [7, 14],
        "oversold":    [25, 30],
        "overbought":  [70, 75],
        "macd_fast":   [8, 12],
        "macd_slow":   [21, 26],
        "macd_signal": [9],
    },
    "EMAトレンド×RSI": {
        "ema_period": [50, 100, 200],
        "rsi_period": [7, 14],
    },
    "BB×ストキャスティクス": {
        "bb_std":    [1.5, 2.0],
        "k_period":  [9, 14],
        "oversold":  [20],
        "overbought":[80],
    },
    "SMAクロス×ATRフィルター": {
        "short_period":   [10, 20],
        "long_period":    [50, 100],
        "atr_multiplier": [1.0, 1.5],
    },
    "RSI×BB": {
        "rsi_period": [7, 14],
        "oversold":   [25, 30],
        "bb_std":     [1.5, 2.0],
    },
    "MACD×ドンチャン": {
        "dc_period": [20, 40],
        "macd_fast": [8, 12],
        "macd_slow": [21, 26],
    },
    "トリプル確認(EMA+RSI+MACD)": {
        "ema_period": [50, 100, 200],
        "rsi_period": [7, 14],
    },
    "ストキャスティクス×EMAトレンド": {
        "ema_period": [50, 100, 200],
        "k_period":   [9, 14],
        "oversold":   [20, 25],
    },
    # ---- 夜間スキャルピング専用 ----
    "夜間スカルパー(4重確認)": {
        "fast_ema":      [5, 8, 13],
        "slow_ema":      [13, 21, 34],
        "rsi_period":    [5, 7],
        "k_period":      [3, 5],
        "atr_multiplier":[0.6, 0.8, 1.0],
    },
    "夜間ブレイクアウト(BB拡張)": {
        "dc_period":     [5, 10, 15],
        "bb_expand_bars":[2, 3, 5],
        "atr_multiplier":[0.8, 1.0, 1.2],
    },
    "夜間押し目買い(EMA+RSI+ATR)": {
        "trend_ema":     [13, 21, 34],
        "entry_ema":     [5, 8],
        "rsi_period":    [5, 7],
        "atr_multiplier":[0.6, 0.8, 1.0],
    },
}


def count_total_combinations(strategies: list[str] | None = None) -> int:
    """テストする組み合わせの総数を返す。"""
    total = 0
    for name, grid in PARAM_GRIDS.items():
        if strategies and name not in strategies:
            continue
        n = 1
        for v in grid.values():
            n *= len(v)
        total += n
    return total


# ============================================================
# グリッドサーチ（ジェネレータ）
# ============================================================

@dataclass
class OptimizeResult:
    strategy: str
    params: dict
    n_trades: int
    win_rate: float
    total_pnl_pips: float
    total_pnl_jpy: float
    profit_factor: float
    max_drawdown_jpy: float
    score: float         # 総合スコア（ランキング用）


def _iter_param_combinations(strategy: str) -> Generator[dict, None, None]:
    """指定戦略のパラメータ組み合わせをすべて生成する。"""
    grid = PARAM_GRIDS[strategy]
    keys   = list(grid.keys())
    values = list(grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _calc_score(r: BacktestResult, min_trades: int) -> float:
    """
    総合スコアを計算する。
    - トレード数が少なすぎる場合は -inf
    - profit_factor × win_rate で評価し、最大DDでペナルティ
    """
    if r.n_trades < min_trades:
        return float("-inf")
    if r.total_pnl_pips <= 0:
        return float("-inf")

    pf = min(r.profit_factor, 10.0)  # 上限キャップ（外れ値防止）
    wr = r.win_rate / 100.0

    # DDペナルティ: |最大DD| / 総利益（比率が低いほど良い）
    dd_ratio = abs(r.max_drawdown_jpy) / max(r.total_pnl_jpy, 1.0)
    dd_penalty = 1.0 / (1.0 + dd_ratio)

    return pf * wr * dd_penalty


def run_optimization(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    direction: str,
    spread_pips: float,
    lot_size: int,
    strategies: list[str] | None = None,
    min_trades: int = 10,
    trade_hours: list[int] | None = None,
    progress_cb=None,        # callable(done: int, total: int, label: str) | None
) -> pd.DataFrame:
    """
    グリッドサーチを実行して結果をDataFrameで返す。

    Args:
        progress_cb: 進捗コールバック。(done, total, label) を受け取る。

    Returns:
        スコア降順にソートされた結果DataFrame。
    """
    target_strategies = strategies or list(PARAM_GRIDS.keys())
    total  = count_total_combinations(target_strategies)
    done   = 0
    rows: list[OptimizeResult] = []

    for strategy in target_strategies:
        for params in _iter_param_combinations(strategy):
            label = f"{strategy}  {params}"
            if progress_cb:
                progress_cb(done, total, label)

            bt_params = BacktestParams(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                strategy_params=params,
                direction=direction,
                spread_pips=spread_pips,
                lot_size=lot_size,
                trade_hours=trade_hours,
            )
            result = run_backtest(bt_params)
            score  = _calc_score(result, min_trades)

            rows.append(OptimizeResult(
                strategy=strategy,
                params=params,
                n_trades=result.n_trades,
                win_rate=result.win_rate,
                total_pnl_pips=result.total_pnl_pips,
                total_pnl_jpy=result.total_pnl_jpy,
                profit_factor=result.profit_factor if result.profit_factor != float("inf") else 999.0,
                max_drawdown_jpy=result.max_drawdown_jpy,
                score=score,
            ))
            done += 1

    if progress_cb:
        progress_cb(total, total, "完了")

    df = pd.DataFrame([
        {
            "戦略":           r.strategy,
            "パラメータ":     _fmt_params(r.params),
            "スコア":         round(r.score, 4) if r.score != float("-inf") else None,
            "PF":             round(r.profit_factor, 2),
            "勝率(%)":        round(r.win_rate, 1),
            "総損益(pips)":   round(r.total_pnl_pips, 1),
            "総損益(円)":     int(r.total_pnl_jpy),
            "最大DD(円)":     int(r.max_drawdown_jpy),
            "トレード数":     r.n_trades,
            "_params":        r.params,   # 詳細再現用（非表示）
            "_score":         r.score,
        }
        for r in rows
    ])

    # スコア降順ソート（-inf は末尾）
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)
    df.index += 1  # 1始まりのランク
    return df


def _fmt_params(params: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in params.items())
