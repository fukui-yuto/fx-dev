"""
dashboard/sample_data.py

データ供給モジュール。
MT5が接続済みなら実データを取得し、未接続ならサンプルデータを返す。
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import TIMEFRAME_MINUTES


# ============================================================
# MT5リアルデータ取得（キャッシュ付き）
# ============================================================

@st.cache_data(ttl=1)  # 1秒キャッシュ
def fetch_mt5_dataframe(
    symbol: str,
    timeframe: str,
    count: int = 500,
) -> pd.DataFrame:
    """
    MT5からローソク足を取得する。
    失敗した場合は例外を送出する。
    """
    from data.mt5_client import get_client
    client = get_client()
    return client.fetch_candles(symbol, timeframe, count=count)


# ============================================================
# サンプルデータ生成（フォールバック用）
# ============================================================

@st.cache_data
def generate_ohlcv_dataframe(
    symbol: str,
    timeframe: str,
    n_bars: int = 500,
    base_price: float = 148.0,
    seed: int = 42,
) -> pd.DataFrame:
    """ランダムウォークでOHLCVのサンプルDataFrameを生成する（フォールバック用）。"""
    rng = np.random.default_rng(seed)
    minutes = TIMEFRAME_MINUTES[timeframe]

    end = pd.Timestamp("2025-01-01", tz="UTC")
    timestamps = pd.date_range(
        end=end,
        periods=n_bars,
        freq=f"{minutes}min",
        tz="UTC",
    )

    returns = rng.normal(0, 0.0003, n_bars)
    close = base_price * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = base_price
    upper = np.abs(rng.normal(0, 0.0002, n_bars))
    lower = np.abs(rng.normal(0, 0.0002, n_bars))
    high = np.maximum(open_, close) * (1 + upper)
    low  = np.minimum(open_, close) * (1 - lower)
    volume = rng.poisson(lam=1000, size=n_bars).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


# ============================================================
# サンプルデータのライブ変換（タイムスタンプ現在化 + 疑似ティック）
# ============================================================

def _make_live_sample(
    symbol: str,
    timeframe: str,
    count: int,
) -> pd.DataFrame:
    """
    サンプルデータのタイムスタンプを現在時刻までシフトし、
    最終足に疑似ティックを付加してライブ感を演出する。
    """
    df = generate_ohlcv_dataframe(symbol, timeframe, n_bars=count).copy()

    # タイムスタンプを現在時刻まで移動
    minutes = TIMEFRAME_MINUTES[timeframe]
    new_end = pd.Timestamp.now("UTC").floor(f"{minutes}min")
    new_index = pd.date_range(
        end=new_end, periods=len(df), freq=f"{minutes}min", tz="UTC"
    )
    df.index = new_index
    df.index.name = "timestamp"

    # 最終足の close をランダムウォークで微小変動させる（疑似ティック）
    rng = np.random.default_rng(int(_time.time() * 10) % (2 ** 32))
    close = float(df.iloc[-1]["close"])
    delta = close * float(rng.normal(0, 0.0001))
    new_close = round(close + delta, 5)
    df.iloc[-1, df.columns.get_loc("close")] = new_close
    df.iloc[-1, df.columns.get_loc("high")] = round(
        max(float(df.iloc[-1]["high"]), new_close), 5
    )
    df.iloc[-1, df.columns.get_loc("low")] = round(
        min(float(df.iloc[-1]["low"]), new_close), 5
    )
    return df


# ============================================================
# 統合取得関数（MT5優先、フォールバックあり）
# ============================================================

def get_ohlcv_dataframe(
    symbol: str,
    timeframe: str,
    count: int = 500,
) -> tuple[pd.DataFrame, str]:
    """
    MT5が接続済みなら実データを、未接続・エラー時はサンプルデータを返す。

    Returns:
        (DataFrame, source): source は "mt5" または "sample" または "sample (error: ...)"
    """
    from data.mt5_client import is_connected
    if is_connected():
        try:
            df = fetch_mt5_dataframe(symbol, timeframe, count=count)
            return df, "mt5"
        except Exception as e:
            return _make_live_sample(symbol, timeframe, count), f"sample (error: {e})"
    return _make_live_sample(symbol, timeframe, count), "sample"


# ============================================================
# バックテスト結果サンプル生成
# ============================================================

@st.cache_data
def generate_backtest_result(
    n_trades: int = 80,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """バックテスト結果のサンプルデータを生成する。"""
    rng = np.random.default_rng(seed)

    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    entry_times = [base_time + pd.Timedelta(hours=int(h)) for h in np.cumsum(rng.integers(1, 24, n_trades))]
    exit_times  = [t + pd.Timedelta(hours=int(h)) for t, h in zip(entry_times, rng.integers(1, 12, n_trades))]

    directions    = rng.choice(["BUY", "SELL"], n_trades)
    entry_prices  = rng.uniform(145.0, 152.0, n_trades)
    price_changes = rng.normal(0, 0.5, n_trades)
    exit_prices   = entry_prices + price_changes

    pip_values = np.where(directions == "BUY", exit_prices - entry_prices, entry_prices - exit_prices)
    pnl        = np.round(pip_values * 1000, 0)
    pnl_cumsum = np.cumsum(pnl)

    trades_df = pd.DataFrame({
        "entry_time":  entry_times,
        "exit_time":   exit_times,
        "direction":   directions,
        "entry_price": np.round(entry_prices, 3),
        "exit_price":  np.round(exit_prices, 3),
        "pnl":         pnl,
        "pnl_cumsum":  pnl_cumsum,
    })

    all_times = pd.date_range(entry_times[0], exit_times[-1], freq="1h", tz="UTC")
    equity_series = pd.Series(dtype=float, index=all_times)
    for _, row in trades_df.iterrows():
        mask = (all_times >= row["entry_time"]) & (all_times <= row["exit_time"])
        equity_series.loc[mask] = row["pnl_cumsum"]
    equity_series = equity_series.ffill().fillna(0)

    equity_df = pd.DataFrame({"equity": equity_series})
    equity_df.index.name = "timestamp"

    return trades_df, equity_df


# ============================================================
# データセット一覧情報
# ============================================================

@st.cache_data(ttl=60)
def get_dataset_info(symbols: list[str], timeframes: list[str]) -> pd.DataFrame:
    """全通貨ペア×時間足の組み合わせでデータセット情報を返す。"""
    from config.settings import get_data_path

    rows = []
    for symbol in symbols:
        for tf in timeframes:
            data_dir = get_data_path(symbol, tf)
            parquet_files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []

            if parquet_files:
                dfs = [pd.read_parquet(f) for f in parquet_files]
                df = pd.concat(dfs).sort_index()
                rows.append({
                    "symbol":    symbol,
                    "timeframe": tf,
                    "件数":      len(df),
                    "開始":      str(df.index.min())[:19],
                    "終了":      str(df.index.max())[:19],
                    "データ源":  "実データ(Parquet)",
                })
            else:
                df, source = get_ohlcv_dataframe(symbol, tf, count=500)
                src_label = "MT5" if source == "mt5" else "サンプル"
                rows.append({
                    "symbol":    symbol,
                    "timeframe": tf,
                    "件数":      len(df),
                    "開始":      str(df.index.min())[:19],
                    "終了":      str(df.index.max())[:19],
                    "データ源":  src_label,
                })

    return pd.DataFrame(rows)
