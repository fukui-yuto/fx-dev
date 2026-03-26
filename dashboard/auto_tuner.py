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
import math
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Optuna (ベイズ最適化 TPE) — インストール済みなら使用、なければグリッドサーチにフォールバック
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

ROOT_DIR  = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "output"
CACHE_TTL = 24 * 3600  # 24h

# ============================================================
# セッション → JST時間帯マッピング（UTC+9）
#   ロンドン      : UTC 08-17 = JST 17-02
#   ロンドン-NY重複: UTC 13-17 = JST 22-02
# ============================================================
SESSION_HOURS_JST: dict[str, list[int] | None] = {
    "all":          None,
    "london":       [17, 18, 19, 20, 21, 22, 23, 0, 1],
    "overlap":      [22, 23, 0, 1],
    "london_kill":  [16, 17, 18],   # ロンドンキルゾーン 16:00〜19:00 JST
    "ny_kill":      [21, 22, 23],   # NYオープンキルゾーン 21:00〜00:00 JST
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
        "session":           ["all", "overlap", "london_kill", "ny_kill"],
        "chandelier_mult":   [0.0, 3.0],
        "pullback_atr_mult": [0.0, 1.5],
    },
    "ドンチャンブレイクアウト": {
        "period":          [20, 40, 55],
        "rr":              [2.0, 2.5],
        "session":         ["all", "overlap", "london_kill", "ny_kill"],
        "chandelier_mult": [0.0, 3.0],
    },
    "トリプル確認(EMA+RSI+MACD)": {
        "ema_period":        [50, 100],
        "rsi_period":        [9, 14],
        "rr":                [2.0, 2.5],
        "session":           ["all", "overlap", "london_kill", "ny_kill"],
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
    # ---- 新規追加戦略 ----
    "ロンドンブレイクアウト": {
        "breakout_buffer": [0, 3],          # アジアレンジ突破バッファ（pips）
        "rr":              [1.5, 2.0, 2.5],
        "chandelier_mult": [0.0, 3.0],      # シャンデリアトレイリング
    },
    "ICT_FVGスキャルパー": {
        "swing_period":      [5, 10],       # 流動性プール確認期間
        "fvg_min_pips":      [1, 2],        # FVG 最小サイズ（pips）
        "sweep_window":      [3, 5],        # スイープ有効期間（本数）
        "rr":                [1.5, 2.0],
        "session":           ["london_kill", "ny_kill"],  # キルゾーン限定
        "max_bars_in_trade": [0, 12],       # タイムベースエグジット
    },
}

# ATR SL 倍率（固定）
_ATR_SL_MULT = 1.5

# Optuna 最適化: 戦略ごとの試行数
# 3パラメータ以下: 50試行、4パラメータ以上: 80試行（文献: MDPI 2025）
_OPTUNA_TRIALS_BASE = 50
_OPTUNA_TRIALS_LARGE = 80


def _dsr_hurdle(n_trades: int, n_trials: int) -> float:
    """
    DSR（Deflated Sharpe Ratio）ハードル計算。
    Bailey & Lopez de Prado (2014) の簡略実装。

    IS Sharpe がこの値を超えないと過剰最適化の可能性が高い。
    hurdle ≈ sqrt(2 * ln(N) / T)

    Args:
        n_trades: IS 期間のトレード数
        n_trials: 探索したパラメータ組み合わせ総数 N

    Returns:
        超えるべき最低 Sharpe ハードル（per-trade ベース）
    """
    if n_trades < 5 or n_trials < 1:
        return 0.0
    return math.sqrt(2.0 * math.log(max(n_trials, 1)) / max(n_trades, 1))


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


def _optuna_optimize_strategy(
    strategy: str,
    grid: dict,
    is_df: pd.DataFrame,
    n_trials: int,
) -> list[dict]:
    """
    Optuna TPE で IS 期間の戦略パラメータを最適化する。

    Returns:
        結果リスト [{strategy, params, is_pf, is_wr, score, is_sharpe}, ...]
    """
    results: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        params = {
            key: trial.suggest_categorical(key, values)
            for key, values in grid.items()
        }
        m = _run_bt_on_df(is_df, strategy, params)
        if m is None or m["n_trades"] < 5 or m["total_pnl_pips"] <= 0:
            return -1.0
        pf       = min(m["profit_factor"], 10.0) if m["profit_factor"] != float("inf") else 10.0
        wr       = m["win_rate"] / 100.0
        dd_ratio = abs(m["max_drawdown_jpy"]) / max(m["total_pnl_jpy"], 1.0)
        score    = pf * wr / (1.0 + dd_ratio)

        # スタディにカスタム属性として保存（コールバックでなくattr経由）
        trial.set_user_attr("is_pf",     pf)
        trial.set_user_attr("is_wr",     m["win_rate"])
        trial.set_user_attr("is_sharpe", m["sharpe_ratio"])
        trial.set_user_attr("params",    params)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for trial in study.trials:
        if trial.value is None or trial.value < 0:
            continue
        results.append({
            "strategy":   strategy,
            "params":     trial.user_attrs.get("params", {}),
            "is_pf":      trial.user_attrs.get("is_pf", 0.0),
            "is_wr":      trial.user_attrs.get("is_wr", 0.0),
            "is_sharpe":  trial.user_attrs.get("is_sharpe", 0.0),
            "score":      trial.value,
        })

    return results


def _grid_optimize_strategy(
    strategy: str,
    grid: dict,
    is_df: pd.DataFrame,
    progress_cb: Callable[[int, int, str], None] | None,
    done_ref: list[int],
    total: int,
) -> list[dict]:
    """グリッドサーチ（Optuna 未インストール時フォールバック）。"""
    results: list[dict] = []
    keys   = list(grid.keys())
    values = list(grid.values())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        done_ref[0] += 1
        if progress_cb:
            progress_cb(done_ref[0], total, strategy)

        m = _run_bt_on_df(is_df, strategy, params)
        if m is None or m["n_trades"] < 5 or m["total_pnl_pips"] <= 0:
            continue

        pf       = min(m["profit_factor"], 10.0) if m["profit_factor"] != float("inf") else 10.0
        wr       = m["win_rate"] / 100.0
        dd_ratio = abs(m["max_drawdown_jpy"]) / max(m["total_pnl_jpy"], 1.0)
        score    = pf * wr / (1.0 + dd_ratio)
        results.append({
            "strategy":  strategy,
            "params":    params,
            "is_pf":     pf,
            "is_wr":     m["win_rate"],
            "is_sharpe": m.get("sharpe_ratio", 0.0),
            "score":     score,
        })
    return results


def run_auto_tune(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict | None:
    """
    ローリング3窓 IS/OOS 最適化を実行して最良戦略を返す。

    最適化手法:
    - Optuna インストール済み: TPE ベイズ最適化（50〜80試行/戦略）
    - Optuna 未インストール: グリッドサーチ（フォールバック）

    追加機能:
    - DSR ハードル（Bailey & Lopez de Prado 2014）: 過剰最適化パラメータを自動排除
    - IS Sharpe が DSR ハードルを下回る候補は除外

    Returns:
        {
            "strategy":    str,
            "params":      dict,
            "oos_pf":      float,
            "oos_winrate": float,
            "oos_trades":  int,
            "is_score":    float,
            "is_sharpe":   float,
            "dsr_hurdle":  float,
            "n_trials":    int,
            "optimizer":   "optuna" | "grid",
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
    n        = len(df)
    split_is = int(n * 0.60)
    split_o1 = int(n * 0.73)
    split_o2 = int(n * 0.87)

    is_df   = df.iloc[:split_is].copy()
    oos_dfs = [
        df.iloc[split_is:split_o1].copy(),  # OOS窓1（重み25%）
        df.iloc[split_o1:split_o2].copy(),  # OOS窓2（重み35%）
        df.iloc[split_o2:].copy(),          # OOS窓3・最直近（重み40%）
    ]
    OOS_WEIGHTS = [0.25, 0.35, 0.40]

    # ---------- IS 最適化 ----------
    results: list[dict] = []
    total_grid = sum(
        1
        for grid in AUTO_PARAM_GRIDS.values()
        for _ in itertools.product(*grid.values())
    )
    done_ref = [0]  # グリッドサーチ進捗用（ミュータブルにするためリスト）
    optimizer_used = "grid"

    if _OPTUNA_AVAILABLE:
        optimizer_used = "optuna"
        strat_count = len(AUTO_PARAM_GRIDS)
        for idx, (strategy, grid) in enumerate(AUTO_PARAM_GRIDS.items()):
            n_params = len(grid)
            n_trials = _OPTUNA_TRIALS_LARGE if n_params >= 4 else _OPTUNA_TRIALS_BASE
            if progress_cb:
                progress_cb(idx + 1, strat_count, strategy)
            results.extend(
                _optuna_optimize_strategy(strategy, grid, is_df, n_trials)
            )
        # DSR の N = 全戦略の総試行数
        n_trials_total = sum(
            _OPTUNA_TRIALS_LARGE if len(g) >= 4 else _OPTUNA_TRIALS_BASE
            for g in AUTO_PARAM_GRIDS.values()
        )
    else:
        # グリッドサーチフォールバック
        for strategy, grid in AUTO_PARAM_GRIDS.items():
            results.extend(
                _grid_optimize_strategy(
                    strategy, grid, is_df,
                    progress_cb, done_ref, total_grid,
                )
            )
        n_trials_total = total_grid

    if not results:
        return None

    # ---------- DSR ハードルフィルタリング ----------
    # IS Sharpe が DSR ハードルを超えない候補は過剰最適化の可能性が高いので除外
    for r in results:
        is_m = _run_bt_on_df(is_df, r["strategy"], r["params"])
        if is_m is not None:
            r["is_n_trades"] = is_m["n_trades"]
        else:
            r["is_n_trades"] = 0

    dsr_filtered = [
        r for r in results
        if r["is_sharpe"] > _dsr_hurdle(r["is_n_trades"], n_trials_total)
    ]
    # DSR フィルター後に候補がなければ全候補を使う（緩和）
    candidates = dsr_filtered if dsr_filtered else results

    # IS スコア上位 5 候補を OOS 検証
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top5 = candidates[:5]

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

        valid = [(w, p) for w, p in zip(OOS_WEIGHTS, pfs) if p is not None]
        if not valid:
            continue

        total_w    = sum(w for w, _ in valid)
        avg_oos_pf = sum(w * p for w, p in valid) / total_w

        if avg_oos_pf > best_pf:
            best_pf    = avg_oos_pf
            is_n       = cand.get("is_n_trades", 0)
            hurdle     = _dsr_hurdle(is_n, n_trials_total)
            best = {
                "strategy":    cand["strategy"],
                "params":      cand["params"],
                "oos_pf":      round(avg_oos_pf, 2),
                "oos_winrate": round(m_last["win_rate"], 1) if m_last else 0.0,
                "oos_trades":  m_last["n_trades"] if m_last else 0,
                "is_score":    round(cand["score"], 4),
                "is_sharpe":   round(cand.get("is_sharpe", 0.0), 3),
                "dsr_hurdle":  round(hurdle, 3),
                "n_trials":    n_trials_total,
                "optimizer":   optimizer_used,
                "tuned_at":    datetime.now(timezone.utc).isoformat(),
            }

    if best is None:
        is_n   = top5[0].get("is_n_trades", 0)
        hurdle = _dsr_hurdle(is_n, n_trials_total)
        best = {
            "strategy":    top5[0]["strategy"],
            "params":      top5[0]["params"],
            "oos_pf":      0.0,
            "oos_winrate": 0.0,
            "oos_trades":  0,
            "is_score":    round(top5[0]["score"], 4),
            "is_sharpe":   round(top5[0].get("is_sharpe", 0.0), 3),
            "dsr_hurdle":  round(hurdle, 3),
            "n_trials":    n_trials_total,
            "optimizer":   optimizer_used,
            "tuned_at":    datetime.now(timezone.utc).isoformat(),
        }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(symbol, timeframe).write_text(
        json.dumps(best, ensure_ascii=False), encoding="utf-8"
    )
    return best
