"""
dashboard/backtest_engine.py

バックテストエンジン。
- OHLCVデータをローカルDBまたはMT5から取得
- 各戦略のシグナルを生成
- 翌足openで執行（先読みバイアスなし）
- スプレッド込みの損益を計算
- SL/TP（固定pips / ATR倍率）対応
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# USDJPY: 1 pip = 0.01円
PIP_VALUE = 0.01


# ============================================================
# データクラス
# ============================================================

@dataclass
class BacktestParams:
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    strategy: str            # "SMAクロス" | "EMAクロス" | "RSI" | ...
    strategy_params: dict
    direction: str           # "両方" | "ロングのみ" | "ショートのみ"
    spread_pips: float = 1.0
    lot_size: int = 10_000             # ミニロット（通貨単位）
    trade_hours: list[int] | None = None  # 取引許可時間帯（JST時）。None=全時間
    # SL/TP 設定
    sl_tp_type: str = "none"           # "none" | "fixed" | "atr"
    sl_pips: float = 20.0              # 固定SL幅（pips）
    tp_pips: float = 40.0              # 固定TP幅（pips）
    atr_sl_period: int = 14            # ATR計算期間
    atr_sl_mult: float = 1.5           # ATR × 倍率 = SL幅
    atr_tp_mult: float = 2.5           # ATR × 倍率 = TP幅
    adx_min: float = 0.0               # ADXフィルター（0=無効、15推奨）
    hurst_filter: bool = False          # ハーストレジームフィルター（True推奨）
    max_bars_in_trade: int = 0          # タイムベースエグジット（0=無効）
    chandelier_mult: float = 0.0        # シャンデリアエグジット ATR倍率（0=無効, 3.0推奨）
    pullback_atr_mult: float = 0.0      # プルバックエントリーフィルター（0=無効, 1.5推奨）


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str           # "long" | "short"
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_jpy: float
    exit_reason: str = "signal"   # "signal" | "sl" | "tp" | "end"
    hold_bars: int = 0


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_series: pd.Series  # 累積損益（円）の時系列
    total_pnl_pips: float
    total_pnl_jpy: float
    n_trades: int
    win_rate: float           # 0〜100
    profit_factor: float
    max_drawdown_jpy: float
    data_source: str          # "db" | "mt5" | "none"
    symbol: str
    timeframe: str
    strategy: str
    error_msg: str = ""       # データ取得失敗時の詳細メッセージ
    # 追加KPI
    sharpe_ratio: float = 0.0
    recovery_factor: float = 0.0
    max_consec_wins: int = 0
    max_consec_losses: int = 0
    avg_hold_bars: float = 0.0
    expected_value_pips: float = 0.0
    risk_reward_ratio: float = 0.0
    sl_hit_count: int = 0
    tp_hit_count: int = 0


# ============================================================
# データロード
# ============================================================

def load_ohlcv(params: BacktestParams) -> tuple[pd.DataFrame, str, str]:
    """
    OHLCVをローカルDB優先で取得する。なければMT5から取得。

    Returns:
        (df, source, error_msg)
        source   : "db" | "mt5" | "none"
        error_msg: 空文字 or 失敗理由の説明
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from data.local_store import query as db_query
    df = db_query(params.symbol, params.timeframe, params.start_date, params.end_date)
    if not df.empty:
        return df, "db", ""

    # ローカルDBにデータなし → MT5へフォールバック
    errors: list[str] = [
        f"ローカルDBに {params.symbol} / {params.timeframe} のデータがありません。"
    ]

    try:
        from data.mt5_client import is_available, is_connected, get_client
        if not is_available():
            errors.append("MT5ライブラリが利用できません。")
            return pd.DataFrame(), "none", "\n".join(errors)

        if not is_connected():
            errors.append("MT5に接続されていません。")
            return pd.DataFrame(), "none", "\n".join(errors)

        client = get_client()

        # まず日付範囲で試みる
        try:
            df = client.fetch_candles_range(
                params.symbol, params.timeframe,
                params.start_date, params.end_date,
            )
            if not df.empty:
                return df, "mt5", ""
        except Exception as e:
            errors.append(f"MT5（日付範囲取得）失敗: {e}")

        # 次にカウントベースで最大本数を取得してフィルタ
        try:
            df = client.fetch_candles_max(params.symbol, params.timeframe)
            if not df.empty:
                # 期間でフィルタリング
                mask = (df.index >= params.start_date) & (df.index <= params.end_date)
                df = df.loc[mask]
                if not df.empty:
                    return df, "mt5", ""
                errors.append(
                    f"MT5から取得した最大履歴データに指定期間（{params.start_date.date()} 〜 "
                    f"{params.end_date.date()}）のデータがありませんでした。"
                )
        except Exception as e:
            errors.append(f"MT5（最大履歴取得）失敗: {e}")

    except Exception as e:
        errors.append(f"MT5接続エラー: {e}")

    hint = (
        "\n\n【解決方法】「データ確認」ページで "
        f"{params.symbol} / {params.timeframe} の履歴データをダウンロードしてください。"
    )
    return pd.DataFrame(), "none", "\n".join(errors) + hint


# ============================================================
# テクニカル指標
# ============================================================

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int, slow: int, sig: int
          ) -> tuple[pd.Series, pd.Series]:
    macd_line   = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, sig)
    return macd_line, signal_line


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX（平均方向性指数）: トレンド強度 0〜100。25以上が強トレンド。"""
    up       = df["high"].diff().clip(lower=0)
    down     = (-df["low"].diff()).clip(lower=0)
    plus_dm  = pd.Series(np.where(up > down, up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where(down > up, down, 0.0), index=df.index)
    atr_s    = _atr(df, period)
    plus_di  = 100 * plus_dm.rolling(period).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_s.replace(0, np.nan)
    dx       = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


def _hurst(close: pd.Series, window: int = 100) -> pd.Series:
    """
    ローリングハースト指数（R/S分析）。
    H > 0.55: トレンド相場, H < 0.45: 平均回帰相場, 中間: ランダム
    """
    def _rs(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 10:
            return 0.5
        try:
            mean  = arr.mean()
            z     = np.cumsum(arr - mean)
            r     = z.max() - z.min()
            s     = arr.std(ddof=1)
            if s == 0 or r == 0:
                return 0.5
            return float(np.log(r / s) / np.log(n))
        except Exception:
            return 0.5
    return close.rolling(window).apply(_rs, raw=True).fillna(0.5)


def _stochastic(df: pd.DataFrame, k_period: int) -> pd.Series:
    lowest  = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    denom   = (highest - lowest).replace(0, np.nan)
    return (df["close"] - lowest) / denom * 100


# ============================================================
# シグナル生成
# ============================================================

def generate_signals(df: pd.DataFrame, params: BacktestParams) -> pd.Series:
    """
    シグナルを生成する。
      1 = ロングエントリー
     -1 = ショートエントリー
      0 = シグナルなし

    当足closeで判定 → 翌足openで執行（先読みバイアスなし）。
    """
    close = df["close"]
    p     = params.strategy_params
    sig   = pd.Series(0, index=df.index, dtype=int)

    def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a > b) & (a.shift(1) <= b.shift(1))

    def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a < b) & (a.shift(1) >= b.shift(1))

    if params.strategy == "SMAクロス":
        s = _sma(close, p.get("short_period", 10))
        l = _sma(close, p.get("long_period",  30))
        sig = np.where(_cross_up(s, l), 1, np.where(_cross_down(s, l), -1, 0))

    elif params.strategy == "EMAクロス":
        s = _ema(close, p.get("short_period", 10))
        l = _ema(close, p.get("long_period",  30))
        sig = np.where(_cross_up(s, l), 1, np.where(_cross_down(s, l), -1, 0))

    elif params.strategy == "RSI":
        rsi = _rsi(close, p.get("period", 14))
        os  = p.get("oversold",  30)
        ob  = p.get("overbought", 70)
        # RSIが売られ過ぎ閾値を上抜けでロング、買われ過ぎ閾値を下抜けでショート
        sig = np.where(_cross_up(rsi, pd.Series(os, index=rsi.index)), 1,
              np.where(_cross_down(rsi, pd.Series(ob, index=rsi.index)), -1, 0))

    elif params.strategy == "MACD":
        macd_line, signal_line = _macd(
            close,
            p.get("fast",   12),
            p.get("slow",   26),
            p.get("signal",  9),
        )
        sig = np.where(_cross_up(macd_line, signal_line), 1,
              np.where(_cross_down(macd_line, signal_line), -1, 0))

    elif params.strategy == "ボリンジャーバンド":
        period  = p.get("period",  20)
        std_dev = p.get("std_dev", 2.0)
        mid   = _sma(close, period)
        std   = close.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        sig = np.where(_cross_up(close, lower), 1,
              np.where(_cross_down(close, upper), -1, 0))

    # ---- 追加戦略 ----

    elif params.strategy == "ストキャスティクス":
        k_period   = p.get("k_period", 14)
        d_period   = p.get("d_period", 3)
        oversold   = p.get("oversold", 20)
        overbought = p.get("overbought", 80)
        k = _stochastic(df, k_period)
        d = _sma(k, d_period)
        sig = np.where(_cross_up(k, d) & (d < overbought), 1,
              np.where(_cross_down(k, d) & (d > oversold), -1, 0))

    elif params.strategy == "CCI":
        period    = p.get("period", 20)
        buy_th    = p.get("buy_threshold",  -100)
        sell_th   = p.get("sell_threshold",  100)
        typical   = (df["high"] + df["low"] + close) / 3
        ma        = _sma(typical, period)
        md        = typical.rolling(period).apply(
                        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci       = (typical - ma) / (0.015 * md.replace(0, np.nan))
        th_buy    = pd.Series(buy_th,  index=cci.index)
        th_sell   = pd.Series(sell_th, index=cci.index)
        sig = np.where(_cross_up(cci, th_buy), 1,
              np.where(_cross_down(cci, th_sell), -1, 0))

    elif params.strategy == "ウィリアムズ%R":
        period     = p.get("period", 14)
        oversold   = p.get("oversold",   -80)
        overbought = p.get("overbought", -20)
        highest    = df["high"].rolling(period).max()
        lowest     = df["low"].rolling(period).min()
        wr         = (highest - close) / (highest - lowest).replace(0, np.nan) * -100
        th_os = pd.Series(oversold,   index=wr.index)
        th_ob = pd.Series(overbought, index=wr.index)
        sig = np.where(_cross_up(wr, th_os), 1,
              np.where(_cross_down(wr, th_ob), -1, 0))

    elif params.strategy == "ドンチャンブレイクアウト":
        period  = p.get("period", 20)
        highest = df["high"].rolling(period).max().shift(1)
        lowest  = df["low"].rolling(period).min().shift(1)
        sig = np.where(close > highest, 1,
              np.where(close < lowest,  -1, 0))

    elif params.strategy == "ATRブレイクアウト":
        atr_period = p.get("atr_period", 14)
        multiplier = p.get("multiplier", 1.5)
        atr    = _atr(df, atr_period)
        upper  = close.shift(1) + atr * multiplier
        lower  = close.shift(1) - atr * multiplier
        sig = np.where(close > upper, 1,
              np.where(close < lower, -1, 0))

    elif params.strategy == "移動平均乖離率":
        period    = p.get("period", 20)
        threshold = p.get("threshold", 1.0)
        ma        = _sma(close, period)
        deviation = (close - ma) / ma * 100
        th_neg    = pd.Series(-threshold, index=deviation.index)
        th_pos    = pd.Series( threshold, index=deviation.index)
        sig = np.where(_cross_up(deviation, th_neg), 1,
              np.where(_cross_down(deviation, th_pos), -1, 0))

    elif params.strategy == "MACDヒストグラム":
        fast          = p.get("fast",   12)
        slow          = p.get("slow",   26)
        signal_period = p.get("signal",  9)
        macd_line, signal_line = _macd(close, fast, slow, signal_period)
        histogram = macd_line - signal_line
        zero      = pd.Series(0, index=histogram.index)
        sig = np.where(_cross_up(histogram, zero), 1,
              np.where(_cross_down(histogram, zero), -1, 0))

    elif params.strategy == "トリプルEMAクロス":
        fast = p.get("fast",  5)
        mid  = p.get("mid",  20)
        slow = p.get("slow", 60)
        ef = _ema(close, fast)
        em = _ema(close, mid)
        es = _ema(close, slow)
        bull      = (ef > em) & (em > es)
        bear      = (ef < em) & (em < es)
        bull_prev = bull.shift(1).infer_objects(copy=False).fillna(False)
        bear_prev = bear.shift(1).infer_objects(copy=False).fillna(False)
        sig = np.where(bull & ~bull_prev, 1,
              np.where(bear & ~bear_prev, -1, 0))

    elif params.strategy == "ROC":
        period    = p.get("period", 12)
        threshold = p.get("threshold", 0.5)
        roc       = (close - close.shift(period)) / close.shift(period).replace(0, np.nan) * 100
        th_pos = pd.Series( threshold, index=roc.index)
        th_neg = pd.Series(-threshold, index=roc.index)
        sig = np.where(_cross_up(roc, th_neg), 1,
              np.where(_cross_down(roc, th_pos), -1, 0))

    elif params.strategy == "RSIトレンド":
        period  = p.get("period", 14)
        rsi     = _rsi(close, period)
        rsi_mid = pd.Series(50, index=rsi.index)
        sig = np.where(_cross_up(rsi, rsi_mid), 1,
              np.where(_cross_down(rsi, rsi_mid), -1, 0))

    # ============================================================
    # 複合オリジナルインジケーター（2指標以上の組み合わせ）
    # ============================================================

    elif params.strategy == "RSI×MACDクロス":
        rsi_period = p.get("rsi_period",  14)
        oversold   = p.get("oversold",    30)
        overbought = p.get("overbought",  70)
        macd_fast  = p.get("macd_fast",   12)
        macd_slow  = p.get("macd_slow",   26)
        macd_sig_p = p.get("macd_signal",  9)
        rsi                    = _rsi(close, rsi_period)
        macd_line, signal_line = _macd(close, macd_fast, macd_slow, macd_sig_p)
        sig = np.where(_cross_up(macd_line, signal_line)   & (rsi > oversold),   1,
              np.where(_cross_down(macd_line, signal_line) & (rsi < overbought), -1, 0))

    elif params.strategy == "EMAトレンド×RSI":
        ema_period = p.get("ema_period", 100)
        rsi_period = p.get("rsi_period",  14)
        ema     = _ema(close, ema_period)
        rsi     = _rsi(close, rsi_period)
        rsi_mid = pd.Series(50, index=rsi.index)
        sig = np.where(_cross_up(rsi, rsi_mid)   & (close > ema), 1,
              np.where(_cross_down(rsi, rsi_mid) & (close < ema), -1, 0))

    elif params.strategy == "BB×ストキャスティクス":
        bb_period  = p.get("bb_period",  20)
        bb_std     = p.get("bb_std",    2.0)
        k_period   = p.get("k_period",   14)
        d_period   = p.get("d_period",    3)
        oversold   = p.get("oversold",   20)
        overbought = p.get("overbought", 80)
        mid   = _sma(close, bb_period)
        std   = close.rolling(bb_period).std()
        upper = mid + bb_std * std
        lower = mid - bb_std * std
        k     = _stochastic(df, k_period)
        d     = _sma(k, d_period)
        sig = np.where((close <= lower) & _cross_up(k, d)   & (d < overbought), 1,
              np.where((close >= upper) & _cross_down(k, d) & (d > oversold),  -1, 0))

    elif params.strategy == "SMAクロス×ATRフィルター":
        short_p      = p.get("short_period",   10)
        long_p       = p.get("long_period",    50)
        atr_period   = p.get("atr_period",     14)
        atr_mult     = p.get("atr_multiplier", 1.0)
        sma_s  = _sma(close, short_p)
        sma_l  = _sma(close, long_p)
        atr    = _atr(df, atr_period)
        atr_ma = _sma(atr, atr_period)
        high_vol = atr > atr_ma * atr_mult
        sig = np.where(_cross_up(sma_s, sma_l)   & high_vol, 1,
              np.where(_cross_down(sma_s, sma_l) & high_vol, -1, 0))

    elif params.strategy == "RSI×BB":
        rsi_period = p.get("rsi_period", 14)
        oversold   = p.get("oversold",   30)
        overbought = p.get("overbought", 70)
        bb_period  = p.get("bb_period",  20)
        bb_std     = p.get("bb_std",    2.0)
        rsi   = _rsi(close, rsi_period)
        mid   = _sma(close, bb_period)
        std   = close.rolling(bb_period).std()
        upper = mid + bb_std * std
        lower = mid - bb_std * std
        th_os = pd.Series(oversold,   index=rsi.index)
        th_ob = pd.Series(overbought, index=rsi.index)
        sig = np.where(_cross_up(rsi, th_os)   & (close < lower), 1,
              np.where(_cross_down(rsi, th_ob) & (close > upper), -1, 0))

    elif params.strategy == "MACD×ドンチャン":
        dc_period  = p.get("dc_period",  20)
        macd_fast  = p.get("macd_fast",  12)
        macd_slow  = p.get("macd_slow",  26)
        macd_sig_p = p.get("macd_signal", 9)
        dc_high              = df["high"].rolling(dc_period).max().shift(1)
        dc_low               = df["low"].rolling(dc_period).min().shift(1)
        macd_line, sig_line  = _macd(close, macd_fast, macd_slow, macd_sig_p)
        histogram            = macd_line - sig_line
        raw = np.where((close > dc_high) & (histogram > 0), 1,
              np.where((close < dc_low)  & (histogram < 0), -1, 0))
        raw_s   = pd.Series(raw, index=df.index, dtype=int)
        sig     = raw_s.where(raw_s != raw_s.shift(1).fillna(0), 0)

    elif params.strategy == "トリプル確認(EMA+RSI+MACD)":
        ema_period = p.get("ema_period",  100)
        rsi_period = p.get("rsi_period",   14)
        macd_fast  = p.get("macd_fast",    12)
        macd_slow  = p.get("macd_slow",    26)
        macd_sig_p = p.get("macd_signal",   9)
        ema                    = _ema(close, ema_period)
        rsi                    = _rsi(close, rsi_period)
        macd_line, signal_line = _macd(close, macd_fast, macd_slow, macd_sig_p)
        histogram              = macd_line - signal_line
        bull = (close > ema) & (rsi > 50) & (histogram > 0)
        bear = (close < ema) & (rsi < 50) & (histogram < 0)
        bull_prev = bull.shift(1).infer_objects(copy=False).fillna(False)
        bear_prev = bear.shift(1).infer_objects(copy=False).fillna(False)
        sig = np.where(bull & ~bull_prev, 1,
              np.where(bear & ~bear_prev, -1, 0))

    elif params.strategy == "ストキャスティクス×EMAトレンド":
        ema_period = p.get("ema_period", 100)
        k_period   = p.get("k_period",    14)
        d_period   = p.get("d_period",     3)
        oversold   = p.get("oversold",    20)
        overbought = p.get("overbought",  80)
        ema = _ema(close, ema_period)
        k   = _stochastic(df, k_period)
        d   = _sma(k, d_period)
        sig = np.where(_cross_up(k, d)   & (close > ema) & (d < overbought), 1,
              np.where(_cross_down(k, d) & (close < ema) & (d > oversold),  -1, 0))

    # ============================================================
    # 夜間スキャルピング専用オリジナルインジケーター
    # ============================================================

    elif params.strategy == "夜間スカルパー(4重確認)":
        fast_ema   = p.get("fast_ema",      8)
        slow_ema   = p.get("slow_ema",     21)
        rsi_period = p.get("rsi_period",    7)
        k_period   = p.get("k_period",      5)
        d_period   = p.get("d_period",      3)
        oversold   = p.get("oversold",     25)
        overbought = p.get("overbought",   75)
        atr_period = p.get("atr_period",   10)
        atr_mult   = p.get("atr_multiplier", 0.8)

        ema_f = _ema(close, fast_ema)
        ema_s = _ema(close, slow_ema)
        rsi   = _rsi(close, rsi_period)
        k     = _stochastic(df, k_period)
        d     = _sma(k, d_period)
        atr   = _atr(df, atr_period)
        atr_ma = _sma(atr, atr_period)

        uptrend   = ema_f > ema_s
        downtrend = ema_f < ema_s
        high_vol  = atr > atr_ma * atr_mult

        sig = np.where(
            uptrend   & (rsi > 50) & _cross_up(k, d)   & (d < overbought) & high_vol,  1,
            np.where(
            downtrend & (rsi < 50) & _cross_down(k, d) & (d > oversold)   & high_vol, -1, 0))

    elif params.strategy == "夜間ブレイクアウト(BB拡張)":
        dc_period  = p.get("dc_period",      10)
        bb_period  = p.get("bb_period",      20)
        bb_expand  = p.get("bb_expand_bars",  3)
        atr_period = p.get("atr_period",     10)
        atr_mult   = p.get("atr_multiplier", 1.0)

        dc_high   = df["high"].rolling(dc_period).max().shift(1)
        dc_low    = df["low"].rolling(dc_period).min().shift(1)
        bb_std    = close.rolling(bb_period).std()
        expanding = bb_std > bb_std.shift(bb_expand)
        atr       = _atr(df, atr_period)
        atr_ma    = _sma(atr, atr_period)
        high_vol  = atr > atr_ma * atr_mult

        raw = pd.Series(
            np.where((close > dc_high) & expanding & high_vol,  1,
            np.where((close < dc_low)  & expanding & high_vol, -1, 0)),
            index=df.index, dtype=int,
        )
        sig = raw.where(raw != raw.shift(1).infer_objects(copy=False).fillna(0), 0)

    elif params.strategy == "夜間押し目買い(EMA+RSI+ATR)":
        trend_ema  = p.get("trend_ema",     21)
        entry_ema  = p.get("entry_ema",      8)
        rsi_period = p.get("rsi_period",     7)
        oversold   = p.get("oversold",      35)
        overbought = p.get("overbought",    65)
        atr_period = p.get("atr_period",    10)
        atr_mult   = p.get("atr_multiplier", 0.8)

        ema_t  = _ema(close, trend_ema)
        ema_e  = _ema(close, entry_ema)
        rsi    = _rsi(close, rsi_period)
        atr    = _atr(df, atr_period)
        atr_ma = _sma(atr, atr_period)

        high_vol = atr > atr_ma * atr_mult

        pullback_long  = _cross_up(close, ema_e)   & (close > ema_t)
        pullback_short = _cross_down(close, ema_e) & (close < ema_t)

        rsi_ok_long  = (rsi > oversold)   & (rsi < 60)
        rsi_ok_short = (rsi < overbought) & (rsi > 40)

        sig = np.where(pullback_long  & rsi_ok_long  & high_vol,  1,
              np.where(pullback_short & rsi_ok_short & high_vol, -1, 0))

    # ============================================================
    # ロンドンブレイクアウト（戦略J）
    # ============================================================
    elif params.strategy == "ロンドンブレイクアウト":
        buffer_pips = p.get("breakout_buffer", 0)
        buffer      = buffer_pips * PIP_VALUE

        # JST 時刻・日付（UTC+9）
        _jst_off  = pd.Timedelta(hours=9)
        _jst_idx  = df.index + _jst_off
        _jst_hour = pd.Series(_jst_idx.hour, index=df.index, dtype=int)
        _jst_day  = pd.Series(_jst_idx.floor("D"), index=df.index)

        # アジアセッション（09〜14 JST）の当日高値・安値を計算
        _asia_m = (_jst_hour >= 9) & (_jst_hour < 15)
        _tmp = pd.DataFrame({"h": df["high"], "l": df["low"], "d": _jst_day})
        _ag  = _tmp[_asia_m].groupby("d").agg(ah=("h", "max"), al=("l", "min"))

        asia_high_s = _jst_day.map(_ag["ah"])
        asia_low_s  = _jst_day.map(_ag["al"])

        # ロンドンキルゾーン（16〜19 JST）のみシグナル生成
        _ldn = pd.Series((_jst_hour >= 16) & (_jst_hour < 20), index=df.index)

        bull = _cross_up(close, asia_high_s + buffer) & _ldn
        bear = _cross_down(close, asia_low_s  - buffer) & _ldn

        sig = np.where(bull, 1, np.where(bear, -1, 0))

    # ============================================================
    # ICT FVG スキャルパー（戦略I 簡略版）
    # ============================================================
    elif params.strategy == "ICT_FVGスキャルパー":
        swing_period = p.get("swing_period", 10)
        fvg_min_pips = p.get("fvg_min_pips",  1)
        sweep_window = p.get("sweep_window",   5)
        fvg_min      = fvg_min_pips * PIP_VALUE

        # 直近スイング高値・安値（流動性プール）
        swing_high = df["high"].rolling(swing_period).max().shift(1)
        swing_low  = df["low"].rolling(swing_period).min().shift(1)

        # 流動性スイープ検出
        # SSLスイープ: 安値が直近安値を割り込み、終値で回帰（ウィック）
        ssl_sweep = (df["low"] < swing_low) & (close > swing_low)
        # BSLスイープ: 高値が直近高値を超え、終値で回帰
        bsl_sweep = (df["high"] > swing_high) & (close < swing_high)

        # 直近 sweep_window 本以内にスイープがあったか（1本前まで）
        ssl_recent = (
            ssl_sweep.rolling(sweep_window, min_periods=1).max()
            .shift(1).fillna(0).astype(bool)
        )
        bsl_recent = (
            bsl_sweep.rolling(sweep_window, min_periods=1).max()
            .shift(1).fillna(0).astype(bool)
        )

        # フェアバリューギャップ（FVG）検出
        # 強気FVG: 2本前の高値 < 現在の安値（上昇ギャップ）
        bull_fvg = (
            (df["high"].shift(2) < df["low"]) &
            ((df["low"] - df["high"].shift(2)) >= fvg_min)
        )
        # 弱気FVG: 2本前の安値 > 現在の高値（下降ギャップ）
        bear_fvg = (
            (df["low"].shift(2) > df["high"]) &
            ((df["low"].shift(2) - df["high"]) >= fvg_min)
        )

        # シグナル: SSLスイープ後の強気FVG → ロング
        #          BSLスイープ後の弱気FVG → ショート
        sig = np.where(bull_fvg & ssl_recent, 1,
              np.where(bear_fvg & bsl_recent, -1, 0))

    sig = pd.Series(sig, index=df.index, dtype=int)

    # 方向フィルター
    if params.direction == "ロングのみ":
        sig = sig.where(sig == 1, 0)
    elif params.direction == "ショートのみ":
        sig = sig.where(sig == -1, 0)

    # 時間帯フィルター（JST = UTC+9）
    if params.trade_hours is not None:
        jst_hour  = (df.index.hour + 9) % 24
        time_mask = pd.Series(jst_hour, index=df.index).isin(params.trade_hours)
        sig       = sig.where(time_mask, 0)

    # ADX フィルター（レンジ相場排除: ADX < adx_min の足はエントリー禁止）
    if params.adx_min > 0:
        adx      = _adx(df)
        sig      = sig.where(adx >= params.adx_min, 0)

    # ハーストレジームフィルター（戦略タイプと市場レジームの整合性確認）
    if params.hurst_filter:
        _TREND_STRATS = {
            "EMAクロス", "ドンチャンブレイクアウト",
            "トリプル確認(EMA+RSI+MACD)", "夜間スカルパー(4重確認)",
            "ロンドンブレイクアウト", "ICT_FVGスキャルパー",
        }
        _MR_STRATS = {"RSI×BB", "夜間スカルパー(4重確認)"}
        hurst = _hurst(close)
        if params.strategy in _TREND_STRATS - _MR_STRATS:
            sig = sig.where(hurst >= 0.45, 0)
        elif params.strategy in _MR_STRATS - _TREND_STRATS:
            sig = sig.where(hurst <= 0.55, 0)

    # プルバックエントリーフィルター（EMA近傍のみエントリー許可、チェイス防止）
    if params.pullback_atr_mult > 0:
        _PULLBACK_STRATS = {"EMAクロス", "トリプル確認(EMA+RSI+MACD)"}
        if params.strategy in _PULLBACK_STRATS:
            atr14        = _atr(df, 14)
            anchor_period = p.get("short_period", p.get("ema_period", 21))
            ema_anchor   = _ema(close, anchor_period)
            dist         = (close - ema_anchor).abs()
            sig          = sig.where(dist <= atr14 * params.pullback_atr_mult, 0)

    return sig


# ============================================================
# SL/TP ヘルパー
# ============================================================

def _calc_sl_tp_levels(
    direction: str,
    entry_price: float,
    params: BacktestParams,
    atr_value: float,
) -> tuple[float | None, float | None]:
    """(sl_level, tp_level) を返す。sl_tp_type="none" の場合は (None, None)。"""
    if params.sl_tp_type == "none":
        return None, None
    if params.sl_tp_type == "fixed":
        sl_dist = params.sl_pips * PIP_VALUE
        tp_dist = params.tp_pips * PIP_VALUE
    else:  # "atr"
        if atr_value <= 0:
            return None, None
        sl_dist = atr_value * params.atr_sl_mult
        tp_dist = atr_value * params.atr_tp_mult

    if direction == "long":
        return entry_price - sl_dist, entry_price + tp_dist
    else:
        return entry_price + sl_dist, entry_price - tp_dist


def _check_sl_tp_bar(
    direction: str,
    bar_high: float,
    bar_low: float,
    sl_level: float | None,
    tp_level: float | None,
) -> str | None:
    """
    当該バーの高安でSL/TPがヒットしたかチェックする。
    両方ヒットした場合は SL 優先（保守的評価）。
    戻り値: "sl" | "tp" | None
    """
    if direction == "long":
        sl_hit = sl_level is not None and bar_low  <= sl_level
        tp_hit = tp_level is not None and bar_high >= tp_level
    else:
        sl_hit = sl_level is not None and bar_high >= sl_level
        tp_hit = tp_level is not None and bar_low  <= tp_level

    if sl_hit:
        return "sl"
    if tp_hit:
        return "tp"
    return None


# ============================================================
# トレード執行
# ============================================================

def execute_trades(df: pd.DataFrame, signals: pd.Series,
                   params: BacktestParams) -> list[Trade]:
    """
    シグナルに従い翌足openで執行するトレードリストを返す。
    - ポジション保有中に逆シグナルが出たらドテン（即時反転）
    - SL/TP 設定がある場合はバー内高安でヒットを判定
    """
    spread = params.spread_pips * PIP_VALUE
    trades: list[Trade] = []

    # ATRを事前計算（ATRベースSL/TPの場合）
    atr_series: pd.Series | None = None
    if params.sl_tp_type == "atr":
        atr_series = _atr(df, params.atr_sl_period)

    in_trade:    str | None   = None
    entry_time:  datetime | None = None
    entry_price: float | None = None
    sl_level:    float | None = None
    tp_level:    float | None = None
    hold_bars_count: int      = 0
    peak_price:  float        = 0.0

    n = len(df)

    def _make_trade(direction: str, ep: float, exit_raw: float,
                    et: datetime, xt: datetime,
                    reason: str, hold: int) -> Trade:
        if direction == "long":
            pnl_pips = (exit_raw - ep) / PIP_VALUE
        else:
            exit_adj = exit_raw + spread
            pnl_pips = (ep - exit_adj) / PIP_VALUE
        pnl_jpy = pnl_pips * PIP_VALUE * params.lot_size
        return Trade(
            entry_time=et,
            exit_time=xt,
            direction=direction,
            entry_price=ep,
            exit_price=exit_raw,
            pnl_pips=round(pnl_pips, 2),
            pnl_jpy=round(pnl_jpy, 0),
            exit_reason=reason,
            hold_bars=hold,
        )

    # bar i を処理するループ（i=0 は前の足のシグナルがないためスキップ）
    for i in range(1, n):
        bar      = df.iloc[i]
        bar_time = df.index[i]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low  = float(bar["low"])
        prev_sig = int(signals.iloc[i - 1])

        if in_trade is not None:
            hold_bars_count += 1

            # SL/TP チェック（バー内高安で判定）
            hit = _check_sl_tp_bar(in_trade, bar_high, bar_low, sl_level, tp_level)
            if hit is not None:
                exit_price = sl_level if hit == "sl" else tp_level
                trades.append(_make_trade(
                    in_trade, entry_price, exit_price,
                    entry_time, bar_time, hit, hold_bars_count,
                ))
                in_trade = None
                # SL/TPヒット後は同足での再エントリーなし
                continue

            # タイムベースエグジット（停滞ポジションを強制決済）
            if params.max_bars_in_trade > 0 and hold_bars_count >= params.max_bars_in_trade:
                trades.append(_make_trade(
                    in_trade, entry_price, bar_open,
                    entry_time, bar_time, "timeout", hold_bars_count,
                ))
                in_trade = None
                continue

            # シャンデリアエグジット（ピーク価格から N×ATR 引いたトレイリングストップ）
            if params.chandelier_mult > 0 and atr_series is not None:
                _atr_c = float(atr_series.iloc[i])
                if in_trade == "long":
                    peak_price = max(peak_price, bar_high)
                    if float(bar["close"]) < peak_price - params.chandelier_mult * _atr_c:
                        trades.append(_make_trade(
                            in_trade, entry_price, float(bar["close"]),
                            entry_time, bar_time, "chandelier", hold_bars_count,
                        ))
                        in_trade = None
                        continue
                else:  # short
                    peak_price = min(peak_price, bar_low)
                    if float(bar["close"]) > peak_price + params.chandelier_mult * _atr_c:
                        trades.append(_make_trade(
                            in_trade, entry_price, float(bar["close"]),
                            entry_time, bar_time, "chandelier", hold_bars_count,
                        ))
                        in_trade = None
                        continue

            # 逆シグナルでドテン（翌足openで執行 = 当足openで執行済み）
            if (in_trade == "long" and prev_sig == -1) or \
               (in_trade == "short" and prev_sig == 1):
                trades.append(_make_trade(
                    in_trade, entry_price, bar_open,
                    entry_time, bar_time, "signal", hold_bars_count,
                ))
                # ドテン
                new_dir = "long" if prev_sig == 1 else "short"
                in_trade    = new_dir
                entry_time  = bar_time
                entry_price = bar_open + spread if new_dir == "long" else bar_open - spread
                hold_bars_count = 0
                atr_val = float(atr_series.iloc[i - 1]) if atr_series is not None else 0.0
                sl_level, tp_level = _calc_sl_tp_levels(new_dir, entry_price, params, atr_val)
                peak_price = entry_price  # シャンデリア追跡リセット

        else:
            # 新規エントリー
            if prev_sig == 1:
                in_trade    = "long"
                entry_price = bar_open + spread
            elif prev_sig == -1:
                in_trade    = "short"
                entry_price = bar_open - spread

            if in_trade is not None:
                entry_time      = bar_time
                hold_bars_count = 0
                atr_val = float(atr_series.iloc[i - 1]) if atr_series is not None else 0.0
                sl_level, tp_level = _calc_sl_tp_levels(in_trade, entry_price, params, atr_val)
                peak_price = entry_price  # シャンデリア追跡リセット

    # 期末クローズ
    if in_trade is not None and entry_time is not None:
        last_bar  = df.iloc[-1]
        last_time = df.index[-1]
        hold_bars_count += 1
        trades.append(_make_trade(
            in_trade, entry_price, float(last_bar["close"]),
            entry_time, last_time, "end", hold_bars_count,
        ))

    return trades


# ============================================================
# メトリクス計算
# ============================================================

def calc_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return dict(
            total_pnl_pips=0.0, total_pnl_jpy=0.0,
            n_trades=0, win_rate=0.0,
            profit_factor=0.0, max_drawdown_jpy=0.0,
            sharpe_ratio=0.0, recovery_factor=0.0,
            max_consec_wins=0, max_consec_losses=0,
            avg_hold_bars=0.0, expected_value_pips=0.0,
            risk_reward_ratio=0.0, sl_hit_count=0, tp_hit_count=0,
        )

    pips_list = [t.pnl_pips for t in trades]
    jpy_list  = [t.pnl_jpy  for t in trades]

    wins   = [j for j in jpy_list if j > 0]
    losses = [j for j in jpy_list if j < 0]
    win_pips  = [p for p in pips_list if p > 0]
    loss_pips = [p for p in pips_list if p < 0]

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    cum    = np.cumsum(jpy_list)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak))

    total_pnl_jpy = round(sum(jpy_list), 0)

    # Sharpe ratio（トレード単位）
    arr = np.array(jpy_list)
    if len(arr) >= 2 and arr.std(ddof=1) > 0:
        sharpe = float(arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr)))
    else:
        sharpe = 0.0

    # Recovery factor
    recovery = round(total_pnl_jpy / abs(max_dd), 2) if max_dd < 0 else float("inf")

    # Max consecutive wins / losses
    max_consec_w = max_consec_l = cur_w = cur_l = 0
    for t in trades:
        if t.pnl_jpy > 0:
            cur_w += 1; cur_l = 0
            max_consec_w = max(max_consec_w, cur_w)
        elif t.pnl_jpy < 0:
            cur_l += 1; cur_w = 0
            max_consec_l = max(max_consec_l, cur_l)
        else:
            cur_w = 0; cur_l = 0

    # Average hold bars
    avg_hold = round(sum(t.hold_bars for t in trades) / len(trades), 1)

    # Expected value per trade (pips)
    expected_val = round(sum(pips_list) / len(trades), 2)

    # Risk/reward ratio (avg win pips / avg loss pips)
    avg_win_pips  = sum(win_pips)  / len(win_pips)  if win_pips  else 0.0
    avg_loss_pips = abs(sum(loss_pips) / len(loss_pips)) if loss_pips else 0.0
    rr = round(avg_win_pips / avg_loss_pips, 2) if avg_loss_pips > 0 else float("inf")

    # SL/TP hit counts
    sl_hits = sum(1 for t in trades if t.exit_reason == "sl")
    tp_hits = sum(1 for t in trades if t.exit_reason == "tp")

    return dict(
        total_pnl_pips=round(sum(pips_list), 2),
        total_pnl_jpy=total_pnl_jpy,
        n_trades=len(trades),
        win_rate=round(len(wins) / len(trades) * 100, 1),
        profit_factor=pf,
        max_drawdown_jpy=round(max_dd, 0),
        sharpe_ratio=round(sharpe, 2),
        recovery_factor=recovery,
        max_consec_wins=max_consec_w,
        max_consec_losses=max_consec_l,
        avg_hold_bars=avg_hold,
        expected_value_pips=expected_val,
        risk_reward_ratio=rr,
        sl_hit_count=sl_hits,
        tp_hit_count=tp_hits,
    )


def build_equity_series(trades: list[Trade]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float, name="equity")
    times    = [t.exit_time for t in trades]
    cum_jpy  = list(np.cumsum([t.pnl_jpy for t in trades]))
    return pd.Series(cum_jpy, index=times, name="equity")


# ============================================================
# メインエントリポイント
# ============================================================

def run_backtest(params: BacktestParams) -> BacktestResult:
    df, source, error_msg = load_ohlcv(params)
    if df.empty:
        return BacktestResult(
            trades=[], equity_series=pd.Series(dtype=float),
            total_pnl_pips=0.0, total_pnl_jpy=0.0,
            n_trades=0, win_rate=0.0, profit_factor=0.0,
            max_drawdown_jpy=0.0, data_source=source,
            symbol=params.symbol, timeframe=params.timeframe,
            strategy=params.strategy, error_msg=error_msg,
        )

    signals = generate_signals(df, params)
    trades  = execute_trades(df, signals, params)
    metrics = calc_metrics(trades)
    equity  = build_equity_series(trades)

    return BacktestResult(
        trades=trades,
        equity_series=equity,
        data_source=source,
        symbol=params.symbol,
        timeframe=params.timeframe,
        strategy=params.strategy,
        **metrics,
    )
