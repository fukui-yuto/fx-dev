"""
dashboard/walkforward.py

ウォークフォワード分析エンジン。
- データ期間を複数ウィンドウに分割
- 各ウィンドウの In-Sample でグリッドサーチ最適化
- Out-of-Sample で最良パラメータを検証
- IS/OOS の成績比較でカーブフィッティングを定量評価
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dashboard.backtest_engine import (
    BacktestParams, Trade,
    run_backtest, build_equity_series,
)
from dashboard.optimizer import PARAM_GRIDS, run_optimization


# ============================================================
# データクラス
# ============================================================

@dataclass
class WalkForwardParams:
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    strategies: list[str]
    direction: str = "両方"
    spread_pips: float = 0.3
    lot_size: int = 10_000
    trade_hours: list[int] | None = None
    n_windows: int = 5
    is_ratio: float = 0.7          # ISが各ウィンドウに占める割合
    wf_method: str = "rolling"     # "rolling" | "anchored"
    min_trades: int = 5


@dataclass
class WalkForwardWindow:
    window_idx: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    # IS 最適化結果
    best_strategy: str
    best_params: dict
    is_score: float
    is_pf: float
    is_win_rate: float
    is_n_trades: int
    is_total_pnl_pips: float
    # OOS 検証結果
    oos_pf: float
    oos_win_rate: float
    oos_n_trades: int
    oos_total_pnl_pips: float
    oos_total_pnl_jpy: float
    oos_max_dd: float
    oos_trades: list[Trade] = field(default_factory=list)
    oos_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    combined_oos_equity: pd.Series   # 全OOSトレードを時系列に連結した損益曲線
    total_oos_pnl_pips: float
    total_oos_pnl_jpy: float
    oos_pf_avg: float                # 全ウィンドウの平均OOS PF
    oos_win_rate_avg: float
    profitable_windows: int          # OOSで黒字だったウィンドウ数
    oos_efficiency: float            # OOS PF avg / IS PF avg（1.0に近いほど良い）
    error_msg: str = ""


# ============================================================
# ウィンドウ分割
# ============================================================

def _split_windows(
    start: datetime,
    end: datetime,
    n_windows: int,
    is_ratio: float,
    method: str,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """
    (is_start, is_end, oos_start, oos_end) のリストを返す。

    rolling : ウィンドウを等幅でスライド。IS/OOSとも固定長。
    anchored: ISは常に開始日から伸びる拡張窓。OOSは固定長でスライド。
    """
    total_days = (end - start).days
    if total_days < 30:
        raise ValueError("分析期間が短すぎます（30日以上必要）。")

    windows: list[tuple[datetime, datetime, datetime, datetime]] = []

    if method == "rolling":
        window_days = total_days // n_windows
        is_days     = max(7, int(window_days * is_ratio))
        oos_days    = window_days - is_days
        if oos_days < 1:
            raise ValueError("OOS期間が0日になります。IS比率を下げるかウィンドウ数を減らしてください。")

        for k in range(n_windows):
            is_start  = start + timedelta(days=k * window_days)
            is_end    = is_start  + timedelta(days=is_days)
            oos_start = is_end
            oos_end   = is_start  + timedelta(days=window_days)
            if k == n_windows - 1:
                oos_end = end   # 最終ウィンドウは実際の終了日まで
            if oos_start >= end:
                break
            windows.append((is_start, is_end, oos_start, min(oos_end, end)))

    else:  # anchored
        # 全体を (n_windows + 1) 等分し、第1チャンクを初期IS、以降をOOSチャンクとして使う
        chunk_days = max(7, total_days // (n_windows + 1))
        for k in range(n_windows):
            oos_start = start + timedelta(days=(k + 1) * chunk_days)
            oos_end   = start + timedelta(days=(k + 2) * chunk_days)
            is_start  = start
            is_end    = oos_start
            if oos_start >= end:
                break
            if k == n_windows - 1:
                oos_end = end
            windows.append((is_start, is_end, oos_start, min(oos_end, end)))

    if not windows:
        raise ValueError("有効なウィンドウが生成できませんでした。設定を確認してください。")

    return windows


# ============================================================
# メイン実行
# ============================================================

def run_walk_forward(
    wf_params: WalkForwardParams,
    progress_cb=None,
) -> WalkForwardResult:
    """
    ウォークフォワード分析を実行する。

    Args:
        wf_params:   分析パラメータ
        progress_cb: callable(window_idx, n_windows, inner_done, inner_total, label)
                     UI進捗表示用コールバック。None可。
    """
    try:
        windows_def = _split_windows(
            wf_params.start_date,
            wf_params.end_date,
            wf_params.n_windows,
            wf_params.is_ratio,
            wf_params.wf_method,
        )
    except ValueError as e:
        return WalkForwardResult(
            windows=[], combined_oos_equity=pd.Series(dtype=float),
            total_oos_pnl_pips=0.0, total_oos_pnl_jpy=0.0,
            oos_pf_avg=0.0, oos_win_rate_avg=0.0,
            profitable_windows=0, oos_efficiency=0.0,
            error_msg=str(e),
        )

    n_actual = len(windows_def)
    result_windows: list[WalkForwardWindow] = []

    for win_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows_def):

        # ---- IS グリッドサーチ ----
        def _inner_progress(done: int, total: int, label: str) -> None:
            if progress_cb:
                progress_cb(win_idx, n_actual, done, total, label)

        is_df = run_optimization(
            symbol=wf_params.symbol,
            timeframe=wf_params.timeframe,
            start_date=is_start,
            end_date=is_end,
            direction=wf_params.direction,
            spread_pips=wf_params.spread_pips,
            lot_size=wf_params.lot_size,
            strategies=wf_params.strategies,
            min_trades=wf_params.min_trades,
            trade_hours=wf_params.trade_hours,
            progress_cb=_inner_progress,
        )

        valid_is = is_df[is_df["_score"] != float("-inf")]
        if valid_is.empty:
            # このウィンドウの IS で有効な結果なし → OOSはスキップ
            result_windows.append(WalkForwardWindow(
                window_idx=win_idx,
                is_start=is_start, is_end=is_end,
                oos_start=oos_start, oos_end=oos_end,
                best_strategy="(なし)", best_params={},
                is_score=float("-inf"),
                is_pf=0.0, is_win_rate=0.0, is_n_trades=0, is_total_pnl_pips=0.0,
                oos_pf=0.0, oos_win_rate=0.0, oos_n_trades=0,
                oos_total_pnl_pips=0.0, oos_total_pnl_jpy=0.0, oos_max_dd=0.0,
            ))
            continue

        best_is = valid_is.iloc[0]

        # ---- OOS バックテスト ----
        oos_bt_params = BacktestParams(
            symbol=wf_params.symbol,
            timeframe=wf_params.timeframe,
            start_date=oos_start,
            end_date=oos_end,
            strategy=best_is["戦略"],
            strategy_params=best_is["_params"],
            direction=wf_params.direction,
            spread_pips=wf_params.spread_pips,
            lot_size=wf_params.lot_size,
            trade_hours=wf_params.trade_hours,
        )
        oos_result = run_backtest(oos_bt_params)
        oos_equity = build_equity_series(oos_result.trades)

        result_windows.append(WalkForwardWindow(
            window_idx=win_idx,
            is_start=is_start, is_end=is_end,
            oos_start=oos_start, oos_end=oos_end,
            best_strategy=best_is["戦略"],
            best_params=best_is["_params"],
            is_score=float(best_is["_score"]),
            is_pf=float(best_is["PF"]),
            is_win_rate=float(best_is["勝率(%)"]),
            is_n_trades=int(best_is["トレード数"]),
            is_total_pnl_pips=float(best_is["総損益(pips)"]),
            oos_pf=oos_result.profit_factor if oos_result.profit_factor != float("inf") else 99.0,
            oos_win_rate=oos_result.win_rate,
            oos_n_trades=oos_result.n_trades,
            oos_total_pnl_pips=oos_result.total_pnl_pips,
            oos_total_pnl_jpy=oos_result.total_pnl_jpy,
            oos_max_dd=oos_result.max_drawdown_jpy,
            oos_trades=oos_result.trades,
            oos_equity=oos_equity,
        ))

    if progress_cb:
        progress_cb(n_actual, n_actual, 0, 1, "完了")

    # ---- 集計 ----
    return _aggregate(result_windows)


def _aggregate(windows: list[WalkForwardWindow]) -> WalkForwardResult:
    """ウィンドウ結果を集計してWalkForwardResultを返す。"""
    valid = [w for w in windows if w.oos_n_trades > 0]

    if not valid:
        return WalkForwardResult(
            windows=windows,
            combined_oos_equity=pd.Series(dtype=float),
            total_oos_pnl_pips=0.0, total_oos_pnl_jpy=0.0,
            oos_pf_avg=0.0, oos_win_rate_avg=0.0,
            profitable_windows=0, oos_efficiency=0.0,
            error_msg="全ウィンドウでOOSトレードが0件でした。期間・パラメータを確認してください。",
        )

    # 全OOSトレードを時系列順に連結して累積損益を計算
    all_trades: list[Trade] = []
    for w in windows:
        all_trades.extend(w.oos_trades)
    all_trades.sort(key=lambda t: t.exit_time)

    if all_trades:
        cum_pnl = list(np.cumsum([t.pnl_jpy for t in all_trades]))
        combined_equity = pd.Series(
            cum_pnl,
            index=[t.exit_time for t in all_trades],
            name="equity",
        )
    else:
        combined_equity = pd.Series(dtype=float, name="equity")

    total_pips = sum(w.oos_total_pnl_pips for w in valid)
    total_jpy  = sum(w.oos_total_pnl_jpy  for w in valid)

    pf_values  = [w.oos_pf       for w in valid]
    wr_values  = [w.oos_win_rate for w in valid]
    is_pf_values = [w.is_pf     for w in valid if w.is_pf > 0]

    oos_pf_avg  = round(float(np.mean(pf_values)), 2)  if pf_values  else 0.0
    oos_wr_avg  = round(float(np.mean(wr_values)), 1)  if wr_values  else 0.0
    is_pf_avg   = float(np.mean(is_pf_values))         if is_pf_values else 0.0

    oos_eff = round(oos_pf_avg / is_pf_avg, 2) if is_pf_avg > 0 else 0.0

    profitable = sum(1 for w in valid if w.oos_total_pnl_pips > 0)

    return WalkForwardResult(
        windows=windows,
        combined_oos_equity=combined_equity,
        total_oos_pnl_pips=round(total_pips, 2),
        total_oos_pnl_jpy=round(total_jpy, 0),
        oos_pf_avg=oos_pf_avg,
        oos_win_rate_avg=oos_wr_avg,
        profitable_windows=profitable,
        oos_efficiency=oos_eff,
    )
