"""
dashboard/pages/backtest.py

バックテスト画面。
- サイドバーで戦略・パラメータ・期間・通貨ペアを設定
- ローカルDB → MT5 の順でデータ取得
- KPI（総損益・勝率・PF・最大DD・トレード数）表示
- 損益曲線（Plotly）
- トレード一覧テーブル
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from dashboard.backtest_engine import BacktestParams, run_backtest

# ============================================================
# 戦略ごとのパラメータ定義
# ============================================================

STRATEGIES = [
    # 単体インジケーター
    "SMAクロス", "EMAクロス", "RSI", "MACD", "ボリンジャーバンド",
    "ストキャスティクス", "CCI", "ウィリアムズ%R", "ドンチャンブレイクアウト",
    "ATRブレイクアウト", "移動平均乖離率", "MACDヒストグラム",
    "トリプルEMAクロス", "ROC", "RSIトレンド",
    # 複合オリジナルインジケーター
    "RSI×MACDクロス", "EMAトレンド×RSI", "BB×ストキャスティクス",
    "SMAクロス×ATRフィルター", "RSI×BB", "MACD×ドンチャン",
    "トリプル確認(EMA+RSI+MACD)", "ストキャスティクス×EMAトレンド",
    # 夜間スキャルピング専用
    "夜間スカルパー(4重確認)", "夜間ブレイクアウト(BB拡張)", "夜間押し目買い(EMA+RSI+ATR)",
]

STRATEGY_PARAMS: dict[str, list[dict]] = {
    "SMAクロス": [
        {"key": "short_period", "label": "短期SMA期間", "min": 2,  "max": 100, "default": 10, "step": 1},
        {"key": "long_period",  "label": "長期SMA期間", "min": 5,  "max": 300, "default": 30, "step": 1},
    ],
    "EMAクロス": [
        {"key": "short_period", "label": "短期EMA期間", "min": 2,  "max": 100, "default": 10, "step": 1},
        {"key": "long_period",  "label": "長期EMA期間", "min": 5,  "max": 300, "default": 30, "step": 1},
    ],
    "RSI": [
        {"key": "period",     "label": "RSI期間",    "min": 2,  "max": 50, "default": 14, "step": 1},
        {"key": "oversold",   "label": "売られ過ぎ", "min": 5,  "max": 45, "default": 30, "step": 1},
        {"key": "overbought", "label": "買われ過ぎ", "min": 55, "max": 95, "default": 70, "step": 1},
    ],
    "MACD": [
        {"key": "fast",   "label": "短期EMA",  "min": 2, "max": 50,  "default": 12, "step": 1},
        {"key": "slow",   "label": "長期EMA",  "min": 5, "max": 200, "default": 26, "step": 1},
        {"key": "signal", "label": "シグナル", "min": 2, "max": 50,  "default":  9, "step": 1},
    ],
    "ボリンジャーバンド": [
        {"key": "period",  "label": "期間",      "min": 5,  "max": 100, "default": 20,  "step": 1},
        {"key": "std_dev", "label": "標準偏差σ", "min": 0.5,"max": 4.0, "default": 2.0, "step": 0.5,
         "type": "float"},
    ],
    "ストキャスティクス": [
        {"key": "k_period",   "label": "%K期間",     "min": 3,  "max": 50,  "default": 14, "step": 1},
        {"key": "d_period",   "label": "%D期間",     "min": 1,  "max": 10,  "default":  3, "step": 1},
        {"key": "oversold",   "label": "売られ過ぎ", "min": 5,  "max": 40,  "default": 20, "step": 1},
        {"key": "overbought", "label": "買われ過ぎ", "min": 60, "max": 95,  "default": 80, "step": 1},
    ],
    "CCI": [
        {"key": "period",         "label": "期間",        "min": 5,   "max": 100,  "default": 20,   "step": 1},
        {"key": "buy_threshold",  "label": "買いライン",  "min": -300,"max": -50,  "default": -100, "step": 10},
        {"key": "sell_threshold", "label": "売りライン",  "min": 50,  "max": 300,  "default":  100, "step": 10},
    ],
    "ウィリアムズ%R": [
        {"key": "period",     "label": "期間",        "min": 3,   "max": 50,  "default": 14,  "step": 1},
        {"key": "oversold",   "label": "売られ過ぎ",  "min": -95, "max": -50, "default": -80, "step": 1},
        {"key": "overbought", "label": "買われ過ぎ",  "min": -50, "max": -5,  "default": -20, "step": 1},
    ],
    "ドンチャンブレイクアウト": [
        {"key": "period", "label": "チャネル期間", "min": 5, "max": 100, "default": 20, "step": 1},
    ],
    "ATRブレイクアウト": [
        {"key": "atr_period", "label": "ATR期間",    "min": 5,  "max": 50,  "default": 14,  "step": 1},
        {"key": "multiplier", "label": "ATR倍率",    "min": 0.5,"max": 5.0, "default": 1.5, "step": 0.5,
         "type": "float"},
    ],
    "移動平均乖離率": [
        {"key": "period",    "label": "MA期間",     "min": 5,  "max": 200, "default": 20,  "step": 1},
        {"key": "threshold", "label": "乖離率閾値(%)", "min": 0.1,"max": 5.0, "default": 1.0, "step": 0.1,
         "type": "float"},
    ],
    "MACDヒストグラム": [
        {"key": "fast",   "label": "短期EMA",  "min": 2, "max": 50,  "default": 12, "step": 1},
        {"key": "slow",   "label": "長期EMA",  "min": 5, "max": 200, "default": 26, "step": 1},
        {"key": "signal", "label": "シグナル", "min": 2, "max": 50,  "default":  9, "step": 1},
    ],
    "トリプルEMAクロス": [
        {"key": "fast", "label": "短期EMA", "min": 2,  "max": 50,  "default":  5, "step": 1},
        {"key": "mid",  "label": "中期EMA", "min": 5,  "max": 100, "default": 20, "step": 1},
        {"key": "slow", "label": "長期EMA", "min": 10, "max": 300, "default": 60, "step": 1},
    ],
    "ROC": [
        {"key": "period",    "label": "ROC期間",     "min": 2,  "max": 50,  "default": 12,  "step": 1},
        {"key": "threshold", "label": "閾値(%)",     "min": 0.1,"max": 5.0, "default": 0.5, "step": 0.1,
         "type": "float"},
    ],
    "RSIトレンド": [
        {"key": "period", "label": "RSI期間", "min": 2, "max": 50, "default": 14, "step": 1},
    ],
    # ---- 複合オリジナルインジケーター ----
    "RSI×MACDクロス": [
        {"key": "rsi_period",  "label": "RSI期間",      "min": 2,  "max": 50,  "default": 14, "step": 1},
        {"key": "oversold",    "label": "RSI 売られ過ぎ","min": 5,  "max": 45,  "default": 30, "step": 1},
        {"key": "overbought",  "label": "RSI 買われ過ぎ","min": 55, "max": 95,  "default": 70, "step": 1},
        {"key": "macd_fast",   "label": "MACD 短期EMA",  "min": 2,  "max": 50,  "default": 12, "step": 1},
        {"key": "macd_slow",   "label": "MACD 長期EMA",  "min": 5,  "max": 200, "default": 26, "step": 1},
        {"key": "macd_signal", "label": "MACD シグナル", "min": 2,  "max": 50,  "default":  9, "step": 1},
    ],
    "EMAトレンド×RSI": [
        {"key": "ema_period", "label": "EMA期間（トレンドフィルター）", "min": 10, "max": 300, "default": 100, "step": 5},
        {"key": "rsi_period", "label": "RSI期間（タイミング）",         "min": 2,  "max": 50,  "default":  14, "step": 1},
    ],
    "BB×ストキャスティクス": [
        {"key": "bb_period",  "label": "BB期間",          "min": 5,  "max": 100, "default": 20,  "step": 1},
        {"key": "bb_std",     "label": "BB標準偏差σ",     "min": 0.5,"max": 4.0, "default": 2.0, "step": 0.5, "type": "float"},
        {"key": "k_period",   "label": "ストキャス %K期間","min": 3,  "max": 50,  "default": 14,  "step": 1},
        {"key": "d_period",   "label": "ストキャス %D期間","min": 1,  "max": 10,  "default":  3,  "step": 1},
        {"key": "oversold",   "label": "ストキャス 売られ過ぎ", "min": 5,  "max": 40, "default": 20, "step": 1},
        {"key": "overbought", "label": "ストキャス 買われ過ぎ", "min": 60, "max": 95, "default": 80, "step": 1},
    ],
    "SMAクロス×ATRフィルター": [
        {"key": "short_period",   "label": "短期SMA期間",     "min": 2,  "max": 100, "default": 10,  "step": 1},
        {"key": "long_period",    "label": "長期SMA期間",     "min": 5,  "max": 300, "default": 50,  "step": 1},
        {"key": "atr_period",     "label": "ATR期間",         "min": 5,  "max": 50,  "default": 14,  "step": 1},
        {"key": "atr_multiplier", "label": "ATR倍率（ボラ閾値）", "min": 0.5,"max": 3.0, "default": 1.0, "step": 0.5, "type": "float"},
    ],
    "RSI×BB": [
        {"key": "rsi_period", "label": "RSI期間",       "min": 2,  "max": 50,  "default": 14,  "step": 1},
        {"key": "oversold",   "label": "RSI 売られ過ぎ", "min": 5,  "max": 45,  "default": 30,  "step": 1},
        {"key": "overbought", "label": "RSI 買われ過ぎ", "min": 55, "max": 95,  "default": 70,  "step": 1},
        {"key": "bb_period",  "label": "BB期間",         "min": 5,  "max": 100, "default": 20,  "step": 1},
        {"key": "bb_std",     "label": "BB標準偏差σ",    "min": 0.5,"max": 4.0, "default": 2.0, "step": 0.5, "type": "float"},
    ],
    "MACD×ドンチャン": [
        {"key": "dc_period",   "label": "ドンチャン期間",  "min": 5,  "max": 100, "default": 20, "step": 1},
        {"key": "macd_fast",   "label": "MACD 短期EMA",   "min": 2,  "max": 50,  "default": 12, "step": 1},
        {"key": "macd_slow",   "label": "MACD 長期EMA",   "min": 5,  "max": 200, "default": 26, "step": 1},
        {"key": "macd_signal", "label": "MACD シグナル",  "min": 2,  "max": 50,  "default":  9, "step": 1},
    ],
    "トリプル確認(EMA+RSI+MACD)": [
        {"key": "ema_period",  "label": "EMA期間",         "min": 10, "max": 300, "default": 100, "step": 5},
        {"key": "rsi_period",  "label": "RSI期間",         "min": 2,  "max": 50,  "default":  14, "step": 1},
        {"key": "macd_fast",   "label": "MACD 短期EMA",    "min": 2,  "max": 50,  "default":  12, "step": 1},
        {"key": "macd_slow",   "label": "MACD 長期EMA",    "min": 5,  "max": 200, "default":  26, "step": 1},
        {"key": "macd_signal", "label": "MACD シグナル",   "min": 2,  "max": 50,  "default":   9, "step": 1},
    ],
    "ストキャスティクス×EMAトレンド": [
        {"key": "ema_period", "label": "EMA期間（トレンドフィルター）", "min": 10, "max": 300, "default": 100, "step": 5},
        {"key": "k_period",   "label": "ストキャス %K期間",            "min": 3,  "max": 50,  "default":  14, "step": 1},
        {"key": "d_period",   "label": "ストキャス %D期間",            "min": 1,  "max": 10,  "default":   3, "step": 1},
        {"key": "oversold",   "label": "ストキャス 売られ過ぎ",        "min": 5,  "max": 40,  "default":  20, "step": 1},
        {"key": "overbought", "label": "ストキャス 買われ過ぎ",        "min": 60, "max": 95,  "default":  80, "step": 1},
    ],
    # ---- 夜間スキャルピング専用 ----
    "夜間スカルパー(4重確認)": [
        {"key": "fast_ema",      "label": "短期EMA（トレンド）",          "min": 2,  "max": 30,  "default":  8,   "step": 1},
        {"key": "slow_ema",      "label": "長期EMA（トレンド）",          "min": 5,  "max": 60,  "default": 21,   "step": 1},
        {"key": "rsi_period",    "label": "RSI期間",                      "min": 2,  "max": 20,  "default":  7,   "step": 1},
        {"key": "k_period",      "label": "ストキャス %K期間",            "min": 2,  "max": 20,  "default":  5,   "step": 1},
        {"key": "d_period",      "label": "ストキャス %D期間",            "min": 1,  "max": 5,   "default":  3,   "step": 1},
        {"key": "oversold",      "label": "ストキャス 売られ過ぎ",        "min": 10, "max": 40,  "default": 25,   "step": 5},
        {"key": "overbought",    "label": "ストキャス 買われ過ぎ",        "min": 60, "max": 90,  "default": 75,   "step": 5},
        {"key": "atr_period",    "label": "ATR期間",                      "min": 3,  "max": 30,  "default": 10,   "step": 1},
        {"key": "atr_multiplier","label": "ATR倍率（ボラ閾値）",          "min": 0.3,"max": 2.0, "default":  0.8, "step": 0.1, "type": "float"},
    ],
    "夜間ブレイクアウト(BB拡張)": [
        {"key": "dc_period",      "label": "ドンチャン期間",              "min": 3,  "max": 30,  "default": 10,   "step": 1},
        {"key": "bb_period",      "label": "BB期間",                      "min": 5,  "max": 50,  "default": 20,   "step": 1},
        {"key": "bb_expand_bars", "label": "BB拡大確認本数",              "min": 1,  "max": 10,  "default":  3,   "step": 1},
        {"key": "atr_period",     "label": "ATR期間",                     "min": 3,  "max": 30,  "default": 10,   "step": 1},
        {"key": "atr_multiplier", "label": "ATR倍率（ボラ閾値）",         "min": 0.3,"max": 2.0, "default":  1.0, "step": 0.1, "type": "float"},
    ],
    "夜間押し目買い(EMA+RSI+ATR)": [
        {"key": "trend_ema",     "label": "トレンドEMA期間",              "min": 5,  "max": 60,  "default": 21,   "step": 1},
        {"key": "entry_ema",     "label": "エントリーEMA期間（押し目）",  "min": 2,  "max": 20,  "default":  8,   "step": 1},
        {"key": "rsi_period",    "label": "RSI期間",                      "min": 2,  "max": 20,  "default":  7,   "step": 1},
        {"key": "oversold",      "label": "RSI 押し目ライン",             "min": 20, "max": 50,  "default": 35,   "step": 5},
        {"key": "overbought",    "label": "RSI 利食いライン",             "min": 50, "max": 80,  "default": 65,   "step": 5},
        {"key": "atr_period",    "label": "ATR期間",                      "min": 3,  "max": 30,  "default": 10,   "step": 1},
        {"key": "atr_multiplier","label": "ATR倍率（ボラ閾値）",          "min": 0.3,"max": 2.0, "default":  0.8, "step": 0.1, "type": "float"},
    ],
}


# ============================================================
# サイドバー
# ============================================================

with st.sidebar:
    st.markdown("### バックテスト設定")

    symbol    = st.selectbox("通貨ペア",   SUPPORTED_SYMBOLS,    key="bt_symbol")
    timeframe = st.selectbox("時間足",     SUPPORTED_TIMEFRAMES, key="bt_tf",
                             index=SUPPORTED_TIMEFRAMES.index("1H"))

    st.markdown("**期間**")
    today     = datetime.now(timezone.utc).date()
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("開始", value=today - timedelta(days=365), key="bt_start")
    with col_e:
        end_date   = st.date_input("終了", value=today,                       key="bt_end")

    st.divider()

    strategy = st.selectbox("戦略", STRATEGIES, key="bt_strategy")

    # 戦略パラメータを動的生成
    st.markdown("**戦略パラメータ**")
    strategy_params: dict = {}
    for p in STRATEGY_PARAMS[strategy]:
        is_float = p.get("type") == "float"
        if is_float:
            val = st.slider(
                p["label"],
                min_value=float(p["min"]), max_value=float(p["max"]),
                value=float(p["default"]), step=float(p["step"]),
                key=f"bt_p_{p['key']}",
            )
        else:
            val = st.slider(
                p["label"],
                min_value=int(p["min"]), max_value=int(p["max"]),
                value=int(p["default"]), step=int(p["step"]),
                key=f"bt_p_{p['key']}",
            )
        strategy_params[p["key"]] = val

    st.divider()

    direction  = st.radio("売買方向", ["両方", "ロングのみ", "ショートのみ"],
                          key="bt_direction", horizontal=True)
    spread     = st.number_input("スプレッド（pips）", min_value=0.0, max_value=5.0,
                                 value=0.3, step=0.1, format="%.1f", key="bt_spread")
    lot_size   = st.number_input("ロットサイズ（通貨単位）", min_value=1000, max_value=1_000_000,
                                 value=10_000, step=1000, key="bt_lot")

    st.divider()

    # 時間帯フィルター
    HOUR_PRESETS: dict = {
        "全時間":                         None,
        "東京時間  (JST  9:00-18:00)":   list(range(9, 18)),
        "ロンドン時間 (JST 16:00-翌1:00)": list(range(16, 24)) + [0],
        "NY時間    (JST 21:00-翌6:00)":  list(range(21, 24)) + list(range(0, 6)),
        "東京+ロンドン重複 (JST 16:00-18:00)": [16, 17],
        "ロンドン+NY重複  (JST 21:00-翌1:00)": [21, 22, 23, 0],
    }
    st.markdown("**取引時間帯（JST）**")
    hour_preset = st.selectbox(
        "プリセット", list(HOUR_PRESETS.keys()), key="bt_hour_preset",
        index=0,
    )
    if hour_preset == "全時間":
        trade_hours = None
        st.caption("全時間帯でトレードします。")
    else:
        trade_hours = HOUR_PRESETS[hour_preset]
        st.caption(f"対象時間: JST {sorted(trade_hours)} 時台")

    with st.expander("カスタム時間帯を指定", expanded=False):
        custom_hours = st.multiselect(
            "取引する時間帯（JST・複数選択可）",
            options=list(range(24)),
            default=trade_hours if trade_hours else list(range(24)),
            format_func=lambda h: f"{h:02d}:00-{h:02d}:59",
            key="bt_custom_hours",
        )
        if custom_hours:
            trade_hours = custom_hours
            st.caption(f"カスタム指定: JST {sorted(custom_hours)} 時台")

    st.divider()

    # SL/TP 設定
    st.markdown("**SL/TP 設定**")
    sl_tp_type = st.radio(
        "SL/TP タイプ", ["なし", "固定pips", "ATR倍率"],
        key="bt_sl_tp_type", horizontal=True,
    )
    if sl_tp_type == "固定pips":
        sl_pips = st.number_input("SL（pips）", min_value=1.0, max_value=500.0,
                                  value=20.0, step=1.0, key="bt_sl_pips")
        tp_pips = st.number_input("TP（pips）", min_value=1.0, max_value=500.0,
                                  value=40.0, step=1.0, key="bt_tp_pips")
        atr_sl_period = 14; atr_sl_mult = 1.5; atr_tp_mult = 2.5
    elif sl_tp_type == "ATR倍率":
        atr_sl_period = st.slider("ATR期間", 5, 50, 14, key="bt_atr_sl_period")
        atr_sl_mult   = st.slider("ATR SL倍率", 0.5, 5.0, 1.5, step=0.5,
                                  key="bt_atr_sl_mult")
        atr_tp_mult   = st.slider("ATR TP倍率", 0.5, 8.0, 2.5, step=0.5,
                                  key="bt_atr_tp_mult")
        sl_pips = 20.0; tp_pips = 40.0
    else:
        sl_pips = 20.0; tp_pips = 40.0
        atr_sl_period = 14; atr_sl_mult = 1.5; atr_tp_mult = 2.5

    run_btn = st.button("▶ バックテスト実行", use_container_width=True, type="primary")

# ============================================================
# メインエリア
# ============================================================

st.markdown("## バックテスト")

if not run_btn and "bt_result" not in st.session_state:
    st.info("サイドバーで設定を確認し「バックテスト実行」を押してください。")
    st.stop()

# 実行
if run_btn:
    _sl_tp_map = {"なし": "none", "固定pips": "fixed", "ATR倍率": "atr"}
    params = BacktestParams(
        symbol=symbol,
        timeframe=timeframe,
        start_date=datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc),
        end_date=datetime(end_date.year,   end_date.month,   end_date.day,   23, 59, 59, tzinfo=timezone.utc),
        strategy=strategy,
        strategy_params=strategy_params,
        direction=direction,
        spread_pips=float(spread),
        lot_size=int(lot_size),
        trade_hours=trade_hours,
        sl_tp_type=_sl_tp_map[sl_tp_type],
        sl_pips=float(sl_pips),
        tp_pips=float(tp_pips),
        atr_sl_period=int(atr_sl_period),
        atr_sl_mult=float(atr_sl_mult),
        atr_tp_mult=float(atr_tp_mult),
    )
    with st.spinner("バックテスト実行中..."):
        result = run_backtest(params)
    st.session_state["bt_result"] = result

result = st.session_state["bt_result"]

# データなし
if result.n_trades == 0:
    src_label = {"db": "ローカルDB", "mt5": "MT5", "none": "なし"}.get(result.data_source, result.data_source)
    if result.data_source == "none":
        st.error("データが取得できませんでした。")
        if result.error_msg:
            st.code(result.error_msg, language=None)
    else:
        st.warning(f"トレードが0件でした。期間・パラメータを調整してください。（データ源: {src_label}）")
    st.stop()

# ============================================================
# サブタイトル
# ============================================================

src_label   = {"db": "ローカルDB", "mt5": "MT5"}.get(result.data_source, result.data_source)
hours_label = "全時間" if trade_hours is None else f"JST {sorted(trade_hours)}時台"
st.caption(
    f"{result.symbol} / {result.timeframe}　{result.strategy}　"
    f"({start_date} 〜 {end_date})　時間帯: {hours_label}　"
    f"データ源: {src_label}　ロット: {lot_size:,}通貨　スプレッド: {spread}pips"
)

# ============================================================
# KPI
# ============================================================

pf_str  = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
rf_str  = f"{result.recovery_factor:.2f}" if result.recovery_factor != float("inf") else "∞"
rr_str  = f"{result.risk_reward_ratio:.2f}" if result.risk_reward_ratio != float("inf") else "∞"
win_cnt = sum(1 for t in result.trades if t.pnl_jpy > 0)
los_cnt = sum(1 for t in result.trades if t.pnl_jpy < 0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("総損益（円）",   f"¥{result.total_pnl_jpy:,.0f}")
col2.metric("総損益（pips）", f"{result.total_pnl_pips:+.1f}")
col3.metric("勝率",           f"{result.win_rate:.1f}%")
col4.metric("PF",             pf_str)
col5.metric("最大DD（円）",   f"¥{result.max_drawdown_jpy:,.0f}")

col6, col7, col8, col9, col10, col11 = st.columns(6)
col6.metric("トレード数", f"{result.n_trades}（勝{win_cnt}/負{los_cnt}）")
col7.metric("シャープレシオ",     f"{result.sharpe_ratio:.2f}")
col8.metric("リカバリーファクター", rf_str)
col9.metric("最大連勝/連敗",      f"{result.max_consec_wins}勝/{result.max_consec_losses}敗")
col10.metric("期待値（pips）",    f"{result.expected_value_pips:+.2f}")
col11.metric("リスクリワード",     rr_str)

# SL/TP ヒット情報
if result.sl_hit_count + result.tp_hit_count > 0:
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("SLヒット数", f"{result.sl_hit_count}件")
    sc2.metric("TPヒット数", f"{result.tp_hit_count}件")
    sc3.metric("平均保有バー数", f"{result.avg_hold_bars:.1f}本")

st.divider()

# ============================================================
# 損益曲線
# ============================================================

st.markdown("### 損益曲線")

eq = result.equity_series
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=eq.index, y=eq.values,
    mode="lines",
    name="累積損益（円）",
    line=dict(color="#26a69a", width=2),
    fill="tozeroy",
    fillcolor="rgba(38,166,154,0.15)",
))
fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
fig.update_layout(
    height=320,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=11),
               tickprefix="¥", tickformat=",.0f"),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# トレード一覧
# ============================================================

st.markdown("### トレード一覧")

_exit_reason_label = {"signal": "シグナル", "sl": "SLヒット", "tp": "TPヒット", "end": "期末"}
trades_df = pd.DataFrame([
    {
        "エントリー時刻": t.entry_time.strftime("%Y-%m-%d %H:%M"),
        "エグジット時刻": t.exit_time.strftime("%Y-%m-%d %H:%M"),
        "方向":           "買" if t.direction == "long" else "売",
        "エントリー価格": round(t.entry_price, 3),
        "エグジット価格": round(t.exit_price,  3),
        "損益(pips)":     t.pnl_pips,
        "損益(円)":       t.pnl_jpy,
        "累積損益(円)":   round(float(eq.iloc[i]) if i < len(eq) else 0.0, 0),
        "決済理由":       _exit_reason_label.get(t.exit_reason, t.exit_reason),
        "保有バー数":     t.hold_bars,
    }
    for i, t in enumerate(result.trades)
])

def _color_pnl(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return "background-color: #1a3a2a"
        if val < 0:
            return "background-color: #3a1a1a"
    return ""

styled = (
    trades_df.style
    .map(_color_pnl, subset=["損益(pips)", "損益(円)", "累積損益(円)"])
    .format({
        "エントリー価格": "{:.3f}",
        "エグジット価格": "{:.3f}",
        "損益(pips)":     "{:+.2f}",
        "損益(円)":       "¥{:,.0f}",
        "累積損益(円)":   "¥{:,.0f}",
    })
    .map(lambda v: "color:#ef5350;font-weight:bold" if v == "SLヒット"
              else ("color:#26a69a;font-weight:bold" if v == "TPヒット" else ""),
              subset=["決済理由"])
)

st.dataframe(styled, use_container_width=True, height=420, hide_index=True)
