"""
dashboard/pages/optimize.py

戦略最適化画面。
- 全戦略×全パラメータ組み合わせを1年間の1Hデータでバックテスト
- スコア（PF × 勝率 / DDペナルティ）で順位付け
- 上位結果の一覧と最良設定の詳細を表示
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
from dashboard.optimizer import HOUR_PRESETS, PARAM_GRIDS, count_total_combinations, run_optimization

# ============================================================
# サイドバー
# ============================================================

with st.sidebar:
    st.markdown("### 最適化設定")

    symbol    = st.selectbox("通貨ペア",   SUPPORTED_SYMBOLS, key="opt_symbol")
    timeframe = st.selectbox("時間足",     SUPPORTED_TIMEFRAMES, key="opt_tf",
                             index=SUPPORTED_TIMEFRAMES.index("1H"))

    today     = datetime.now(timezone.utc).date()
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("開始", value=today - timedelta(days=365), key="opt_start")
    with col_e:
        end_date   = st.date_input("終了", value=today, key="opt_end")

    st.divider()
    st.markdown("**対象戦略**")
    selected_strategies = []
    for name in PARAM_GRIDS.keys():
        if st.checkbox(name, value=True, key=f"opt_str_{name}"):
            selected_strategies.append(name)

    st.divider()
    direction  = st.radio("売買方向", ["両方", "ロングのみ", "ショートのみ"],
                          key="opt_dir", horizontal=True)
    spread     = st.number_input("スプレッド（pips）", min_value=0.0, max_value=5.0,
                                 value=0.3, step=0.1, format="%.1f", key="opt_spread")
    lot_size   = st.number_input("ロットサイズ", min_value=1000, max_value=1_000_000,
                                 value=10_000, step=1000, key="opt_lot")
    min_trades = st.number_input("最小トレード数（足切り）", min_value=1, max_value=10000,
                                 value=10, step=1, key="opt_min_trades")

    st.divider()

    # 時間帯フィルター
    st.markdown("**取引時間帯（JST）**")
    hour_preset = st.selectbox(
        "プリセット", list(HOUR_PRESETS.keys()), key="opt_hour_preset", index=0,
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
            key="opt_custom_hours",
        )
        if custom_hours:
            trade_hours = custom_hours
            st.caption(f"カスタム指定: JST {sorted(custom_hours)} 時台")

    total_combos = count_total_combinations(selected_strategies)
    st.caption(f"テスト組み合わせ数: **{total_combos}** 件")

    run_btn = st.button("▶ 最適化実行", use_container_width=True, type="primary",
                        disabled=(len(selected_strategies) == 0))

# ============================================================
# メインエリア
# ============================================================

st.markdown("## 戦略最適化")
st.caption(
    "全戦略・全パラメータ組み合わせをバックテストし、"
    "PF・勝率・最大DDを総合したスコアで順位付けします。"
)

if not run_btn and "opt_result_df" not in st.session_state:
    st.info("サイドバーで設定を確認し「最適化実行」を押してください。")
    st.stop()

# ============================================================
# 最適化実行
# ============================================================

if run_btn:
    if not selected_strategies:
        st.error("戦略を1つ以上選択してください。")
        st.stop()

    dt_start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    dt_end   = datetime(end_date.year,   end_date.month,   end_date.day, 23, 59, 59, tzinfo=timezone.utc)

    prog_bar  = st.progress(0.0, text="最適化中...")
    prog_text = st.empty()

    def _progress(done: int, total: int, label: str) -> None:
        ratio = done / total if total > 0 else 1.0
        prog_bar.progress(ratio, text=f"最適化中... {done}/{total}")
        prog_text.caption(f"実行中: {label}")

    with st.spinner(""):
        df = run_optimization(
            symbol=symbol,
            timeframe=timeframe,
            start_date=dt_start,
            end_date=dt_end,
            direction=direction,
            spread_pips=float(spread),
            lot_size=int(lot_size),
            strategies=selected_strategies,
            min_trades=int(min_trades),
            trade_hours=trade_hours,
            progress_cb=_progress,
        )

    prog_bar.empty()
    prog_text.empty()

    st.session_state["opt_result_df"]      = df
    st.session_state["opt_run_symbol"]     = symbol
    st.session_state["opt_run_timeframe"]  = timeframe
    st.session_state["opt_run_start"]      = start_date
    st.session_state["opt_run_end"]        = end_date
    st.session_state["opt_run_spread"]     = spread
    st.session_state["opt_run_lot"]        = lot_size
    st.session_state["opt_run_hours"]      = trade_hours

df           = st.session_state["opt_result_df"]
r_sym        = st.session_state.get("opt_run_symbol",    symbol)
r_tf         = st.session_state.get("opt_run_timeframe", timeframe)
r_start      = st.session_state.get("opt_run_start",     start_date)
r_end        = st.session_state.get("opt_run_end",       end_date)
r_spread     = st.session_state.get("opt_run_spread",    spread)
r_lot        = st.session_state.get("opt_run_lot",       lot_size)
r_hours      = st.session_state.get("opt_run_hours",     trade_hours)

_hours_label = "全時間" if r_hours is None else f"JST {sorted(r_hours)}時台"
st.caption(
    f"{r_sym} / {r_tf}　{r_start} 〜 {r_end}　時間帯: {_hours_label}　"
    f"スプレッド: {r_spread}pips　ロット: {r_lot:,}通貨"
)

# ============================================================
# 有効結果（スコアあり）と無効結果に分ける
# ============================================================

valid_df   = df[df["_score"] != float("-inf")].copy()
invalid_df = df[df["_score"] == float("-inf")].copy()

st.markdown(f"### 結果サマリー")
col1, col2, col3 = st.columns(3)
col1.metric("テスト数",       f"{len(df)}件")
col2.metric("有効（採用候補）", f"{len(valid_df)}件")
col3.metric("除外（最小取引数未満・損失）", f"{len(invalid_df)}件")

if valid_df.empty:
    st.warning("有効な結果がありませんでした。最小トレード数を下げるか、期間を広げてください。")
    st.stop()

st.divider()

# ============================================================
# 最良設定
# ============================================================

best = valid_df.iloc[0]
st.markdown("### 最良設定")

bc1, bc2, bc3, bc4, bc5 = st.columns(5)
bc1.metric("戦略",       best["戦略"])
bc2.metric("PF",         f"{best['PF']:.2f}")
bc3.metric("勝率",       f"{best['勝率(%)']:.1f}%")
bc4.metric("総損益(pips)", f"{best['総損益(pips)']:+.1f}")
bc5.metric("トレード数",  best["トレード数"])

st.info(f"**パラメータ:** {best['パラメータ']}")

# 最良設定の損益曲線を描画
best_params = BacktestParams(
    symbol=r_sym,
    timeframe=r_tf,
    start_date=datetime(r_start.year, r_start.month, r_start.day, tzinfo=timezone.utc),
    end_date=datetime(r_end.year, r_end.month, r_end.day, 23, 59, 59, tzinfo=timezone.utc),
    strategy=best["戦略"],
    strategy_params=best["_params"],
    direction=direction,
    spread_pips=float(r_spread),
    lot_size=int(r_lot),
    trade_hours=r_hours,
)
best_result = run_backtest(best_params)

if best_result.equity_series is not None and not best_result.equity_series.empty:
    eq = best_result.equity_series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        mode="lines",
        line=dict(color="#26a69a", width=2),
        fill="tozeroy",
        fillcolor="rgba(38,166,154,0.15)",
        name="累積損益（円）",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        height=280,
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
# 上位20件テーブル
# ============================================================

st.markdown("### 上位ランキング")

display_cols = ["戦略", "パラメータ", "スコア", "PF", "勝率(%)", "総損益(pips)", "総損益(円)", "最大DD(円)", "トレード数"]
top20 = valid_df[display_cols].head(20)

def _color_pnl(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return "background-color: #1a3a2a"
        if val < 0:
            return "background-color: #3a1a1a"
    return ""

styled = (
    top20.style
    .map(_color_pnl, subset=["総損益(pips)", "総損益(円)", "最大DD(円)"])
    .format({
        "スコア":        "{:.4f}",
        "PF":            "{:.2f}",
        "勝率(%)":       "{:.1f}",
        "総損益(pips)":  "{:+.1f}",
        "総損益(円)":    "¥{:,.0f}",
        "最大DD(円)":    "¥{:,.0f}",
    }, na_rep="-")
    .highlight_max(subset=["スコア", "PF", "勝率(%)"], color="#1a3a4a")
)

st.dataframe(styled, use_container_width=True, height=500)

st.divider()

# ============================================================
# 戦略別最良結果
# ============================================================

st.markdown("### 戦略別ベスト")

strat_best_rows = []
for strat in valid_df["戦略"].unique():
    sub = valid_df[valid_df["戦略"] == strat]
    if not sub.empty:
        row = sub.iloc[0][display_cols]
        strat_best_rows.append(row)

if strat_best_rows:
    strat_df = pd.DataFrame(strat_best_rows).reset_index(drop=True)
    strat_df.index += 1
    st.dataframe(
        strat_df.style.format({
            "スコア":       "{:.4f}",
            "PF":           "{:.2f}",
            "勝率(%)":      "{:.1f}",
            "総損益(pips)": "{:+.1f}",
            "総損益(円)":   "¥{:,.0f}",
            "最大DD(円)":   "¥{:,.0f}",
        }, na_rep="-"),
        use_container_width=True,
        hide_index=False,
    )
