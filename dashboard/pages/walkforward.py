"""
dashboard/pages/walkforward.py

ウォークフォワード分析画面。
- データを複数ウィンドウに分割
- 各ウィンドウのIS期間でグリッドサーチ最適化 → 最良パラメータ選定
- OOS期間で同パラメータを検証
- IS/OOSの乖離（過剰最適化度）を定量評価
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from dashboard.optimizer import HOUR_PRESETS, PARAM_GRIDS
from dashboard.walkforward import WalkForwardParams, run_walk_forward

# ============================================================
# サイドバー
# ============================================================

with st.sidebar:
    st.markdown("### ウォークフォワード設定")

    symbol    = st.selectbox("通貨ペア",   SUPPORTED_SYMBOLS, key="wf_symbol")
    timeframe = st.selectbox("時間足",     SUPPORTED_TIMEFRAMES, key="wf_tf",
                             index=SUPPORTED_TIMEFRAMES.index("1H"))

    today    = datetime.now(timezone.utc).date()
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("開始", value=today - timedelta(days=730), key="wf_start")
    with col_e:
        end_date   = st.date_input("終了", value=today, key="wf_end")

    st.divider()
    st.markdown("**ウォークフォワード設定**")

    wf_method = st.radio(
        "分割方式",
        ["rolling（固定幅スライド）", "anchored（拡張窓）"],
        key="wf_method", horizontal=False,
    )
    wf_method_val = "rolling" if wf_method.startswith("rolling") else "anchored"

    n_windows = st.slider("ウィンドウ数", min_value=3, max_value=10, value=5, key="wf_n_windows")
    is_ratio  = st.slider(
        "IS比率（最適化期間の割合）",
        min_value=0.50, max_value=0.85, value=0.70, step=0.05,
        key="wf_is_ratio",
        help="各ウィンドウの何割をIn-Sample（最適化）に使うか。残りがOut-of-Sample（検証）になる。",
    )

    # 期間プレビュー
    total_days = (end_date - start_date).days
    if wf_method_val == "rolling":
        w_days  = max(1, total_days // n_windows)
        is_days = max(1, int(w_days * is_ratio))
        oos_days = w_days - is_days
        st.caption(
            f"ウィンドウ幅: **{w_days}日**  "
            f"（IS: {is_days}日 / OOS: {oos_days}日）× {n_windows}窓"
        )
    else:
        chunk_days = max(1, total_days // (n_windows + 1))
        st.caption(
            f"OOSチャンク: **{chunk_days}日** × {n_windows}窓  "
            f"（ISは各窓ごとに拡張）"
        )

    st.divider()
    st.markdown("**対象戦略**")
    selected_strategies = []
    for name in PARAM_GRIDS.keys():
        if st.checkbox(name, value=True, key=f"wf_str_{name}"):
            selected_strategies.append(name)

    st.divider()
    direction  = st.radio("売買方向", ["両方", "ロングのみ", "ショートのみ"],
                          key="wf_dir", horizontal=True)
    spread     = st.number_input("スプレッド（pips）", min_value=0.0, max_value=5.0,
                                 value=0.3, step=0.1, format="%.1f", key="wf_spread")
    lot_size   = st.number_input("ロットサイズ", min_value=1000, max_value=1_000_000,
                                 value=10_000, step=1000, key="wf_lot")
    min_trades = st.number_input("最小トレード数（足切り）", min_value=1, max_value=500,
                                 value=5, step=1, key="wf_min_trades")

    st.divider()
    st.markdown("**取引時間帯（JST）**")
    hour_preset = st.selectbox(
        "プリセット", list(HOUR_PRESETS.keys()), key="wf_hour_preset", index=0,
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
            key="wf_custom_hours",
        )
        if custom_hours:
            trade_hours = custom_hours

    run_btn = st.button(
        "▶ ウォークフォワード実行",
        use_container_width=True, type="primary",
        disabled=(len(selected_strategies) == 0),
    )

# ============================================================
# メインエリア
# ============================================================

st.markdown("## ウォークフォワード分析")
st.caption(
    "データを複数ウィンドウに分割し、**IS（In-Sample）期間**でパラメータを最適化、"
    "**OOS（Out-of-Sample）期間**で同パラメータを検証します。"
    "ISとOOSの乖離が小さいほど過剰最適化が少ない堅牢な戦略と言えます。"
)

if not run_btn and "wf_result" not in st.session_state:
    st.info("サイドバーで設定を確認し「ウォークフォワード実行」を押してください。")

    with st.expander("ウォークフォワード分析とは？", expanded=True):
        st.markdown("""
**通常のバックテスト・最適化の問題点:**
- 同じデータで「最適化」と「検証」を行うとカーブフィッティング（過学習）が起きる
- バックテスト上では好成績でも、将来の相場では通用しない可能性がある

**ウォークフォワード分析の仕組み:**
```
期間全体 ─────────────────────────────────────>
         Window 1         Window 2         Window 3
         [IS ////// OOS]  [IS ////// OOS]  [IS ////// OOS]
              ↓                 ↓                 ↓
         パラメータ最適化   パラメータ最適化   パラメータ最適化
              ↓                 ↓                 ↓
           OOS検証           OOS検証           OOS検証
```

**評価指標の見方:**
- **OOS効率** = OOS平均PF ÷ IS平均PF。0.5〜1.0が理想的（1.0=ISと同等）
- **黒字窓率** = OOSで利益が出た窓の割合。高いほど安定
- **合算OOS損益曲線** = 全OOS期間を繋げた「実力値」の損益曲線
        """)
    st.stop()

# ============================================================
# ウォークフォワード実行
# ============================================================

if run_btn:
    if not selected_strategies:
        st.error("戦略を1つ以上選択してください。")
        st.stop()

    dt_start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    dt_end   = datetime(end_date.year,   end_date.month,   end_date.day, 23, 59, 59, tzinfo=timezone.utc)

    # 進捗 UI
    progress_header = st.empty()
    prog_outer      = st.progress(0.0)
    prog_inner      = st.progress(0.0)
    prog_label      = st.empty()

    def _progress(win_idx: int, n_wins: int, inner_done: int, inner_total: int, label: str) -> None:
        outer_ratio = win_idx / n_wins if n_wins > 0 else 1.0
        inner_ratio = inner_done / inner_total if inner_total > 0 else 1.0
        progress_header.markdown(
            f"**ウィンドウ {win_idx + 1}/{n_wins}** を最適化中..."
            if win_idx < n_wins else "**完了**"
        )
        prog_outer.progress(outer_ratio, text=f"全体進捗: 窓{win_idx}/{n_wins}")
        prog_inner.progress(inner_ratio, text=f"IS最適化: {inner_done}/{inner_total}")
        prog_label.caption(f"▶ {label}")

    wf_params = WalkForwardParams(
        symbol=symbol,
        timeframe=timeframe,
        start_date=dt_start,
        end_date=dt_end,
        strategies=selected_strategies,
        direction=direction,
        spread_pips=float(spread),
        lot_size=int(lot_size),
        trade_hours=trade_hours,
        n_windows=int(n_windows),
        is_ratio=float(is_ratio),
        wf_method=wf_method_val,
        min_trades=int(min_trades),
    )

    with st.spinner(""):
        wf_result = run_walk_forward(wf_params, progress_cb=_progress)

    progress_header.empty()
    prog_outer.empty()
    prog_inner.empty()
    prog_label.empty()

    st.session_state["wf_result"]    = wf_result
    st.session_state["wf_run_symbol"]    = symbol
    st.session_state["wf_run_tf"]        = timeframe
    st.session_state["wf_run_start"]     = start_date
    st.session_state["wf_run_end"]       = end_date
    st.session_state["wf_run_spread"]    = spread
    st.session_state["wf_run_lot"]       = lot_size
    st.session_state["wf_run_hours"]     = trade_hours
    st.session_state["wf_run_method"]    = wf_method_val
    st.session_state["wf_run_n_windows"] = n_windows
    st.session_state["wf_run_is_ratio"]  = is_ratio

# ============================================================
# 結果表示
# ============================================================

wf_result = st.session_state["wf_result"]
r_sym     = st.session_state.get("wf_run_symbol",    symbol)
r_tf      = st.session_state.get("wf_run_tf",        timeframe)
r_start   = st.session_state.get("wf_run_start",     start_date)
r_end     = st.session_state.get("wf_run_end",       end_date)
r_method  = st.session_state.get("wf_run_method",    wf_method_val)
r_n_win   = st.session_state.get("wf_run_n_windows", n_windows)
r_is_rat  = st.session_state.get("wf_run_is_ratio",  is_ratio)
r_hours   = st.session_state.get("wf_run_hours",     trade_hours)

if wf_result.error_msg:
    st.error(wf_result.error_msg)
    st.stop()

method_label  = "ローリング（固定幅）" if r_method == "rolling" else "アンカード（拡張窓）"
hours_label   = "全時間" if r_hours is None else f"JST {sorted(r_hours)}時台"
st.caption(
    f"{r_sym} / {r_tf}　{r_start} 〜 {r_end}　"
    f"分割方式: {method_label}　ウィンドウ数: {r_n_win}　IS比率: {r_is_rat:.0%}　"
    f"時間帯: {hours_label}"
)

# ============================================================
# サマリー KPI
# ============================================================

n_valid = sum(1 for w in wf_result.windows if w.oos_n_trades > 0)

eff_color = (
    "🟢" if wf_result.oos_efficiency >= 0.7
    else "🟡" if wf_result.oos_efficiency >= 0.4
    else "🔴"
)
win_ratio = (
    f"{wf_result.profitable_windows}/{n_valid}"
    if n_valid > 0 else "0/0"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("OOS合計損益（pips）", f"{wf_result.total_oos_pnl_pips:+.1f}")
c2.metric("OOS合計損益（円）",   f"¥{wf_result.total_oos_pnl_jpy:,.0f}")
c3.metric("OOS平均PF",           f"{wf_result.oos_pf_avg:.2f}")
c4.metric("OOS平均勝率",         f"{wf_result.oos_win_rate_avg:.1f}%")
c5.metric("黒字窓数",            win_ratio)
c6.metric(
    f"OOS効率 {eff_color}",
    f"{wf_result.oos_efficiency:.2f}",
    help="OOS平均PF ÷ IS平均PF。1.0に近いほど過剰最適化が少ない。",
)

st.divider()

# ============================================================
# 合算 OOS 損益曲線
# ============================================================

st.markdown("### 合算 OOS 損益曲線")
st.caption(
    "全ウィンドウのOOSトレードを時系列に連結した損益曲線です。"
    "これがウォークフォワード分析における「実力値」です。"
)

eq = wf_result.combined_oos_equity
if eq.empty:
    st.warning("OOSのトレードが0件でした。")
else:
    positive = eq >= 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.where(positive),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.15)",
        line=dict(color="#26a69a", width=2), name="利益",
    ))
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.where(~positive),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        line=dict(color="#ef5350", width=2), name="損失",
    ))

    # ウィンドウ境界を縦線で表示
    for w in wf_result.windows:
        fig.add_vline(
            x=w.oos_start,
            line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.3)",
            annotation_text=f"W{w.window_idx + 1}",
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color="#9e9e9e",
        )

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=11),
                   tickprefix="¥", tickformat=",.0f"),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# IS vs OOS 比較チャート
# ============================================================

st.markdown("### IS vs OOS PF 比較（過剰最適化の可視化）")
st.caption(
    "青棒がIS（最適化期間）、緑/赤棒がOOS（検証期間）のPFです。"
    "OOSがISに近いほどカーブフィッティングが少ない証拠です。"
)

valid_windows = [w for w in wf_result.windows if w.oos_n_trades > 0]
if valid_windows:
    labels   = [f"W{w.window_idx + 1}" for w in valid_windows]
    is_pfs   = [w.is_pf  for w in valid_windows]
    oos_pfs  = [w.oos_pf for w in valid_windows]
    oos_colors = ["#26a69a" if v >= 1.0 else "#ef5350" for v in oos_pfs]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="IS PF", x=labels, y=is_pfs,
        marker_color="#29b6f6",
        opacity=0.7,
    ))
    fig2.add_trace(go.Bar(
        name="OOS PF", x=labels, y=oos_pfs,
        marker_color=oos_colors,
    ))
    fig2.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1)
    fig2.update_layout(
        height=280,
        barmode="group",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=12)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=11),
                   title="PF"),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ============================================================
# ウィンドウ別詳細テーブル
# ============================================================

st.markdown("### ウィンドウ別詳細")

rows = []
for w in wf_result.windows:
    oos_pf_str = f"{w.oos_pf:.2f}" if w.oos_n_trades > 0 else "-"
    is_score_str = f"{w.is_score:.4f}" if w.is_score != float("-inf") else "-"
    rows.append({
        "窓":          f"W{w.window_idx + 1}",
        "IS期間":      f"{w.is_start.strftime('%m/%d')} 〜 {w.is_end.strftime('%m/%d')}",
        "OOS期間":     f"{w.oos_start.strftime('%m/%d')} 〜 {w.oos_end.strftime('%m/%d')}",
        "最良戦略":    w.best_strategy,
        "パラメータ":  "  ".join(f"{k}={v}" for k, v in w.best_params.items()) if w.best_params else "-",
        "IS スコア":   is_score_str,
        "IS PF":       f"{w.is_pf:.2f}" if w.is_n_trades > 0 else "-",
        "IS 勝率":     f"{w.is_win_rate:.1f}%" if w.is_n_trades > 0 else "-",
        "IS 取引数":   w.is_n_trades if w.is_n_trades > 0 else "-",
        "OOS PF":      oos_pf_str,
        "OOS 勝率":    f"{w.oos_win_rate:.1f}%" if w.oos_n_trades > 0 else "-",
        "OOS 損益(pips)": f"{w.oos_total_pnl_pips:+.1f}" if w.oos_n_trades > 0 else "-",
        "OOS 損益(円)":   int(w.oos_total_pnl_jpy) if w.oos_n_trades > 0 else "-",
        "OOS 取引数":  w.oos_n_trades if w.oos_n_trades > 0 else "-",
        "_oos_pips":   w.oos_total_pnl_pips,
    })

detail_df = pd.DataFrame(rows)

def _row_style(row):
    styles = [""] * len(row)
    try:
        v = float(row["_oos_pips"])
        color = "background-color: #1a3a2a" if v > 0 else (
                "background-color: #3a1a1a" if v < 0 else "")
        idx = list(row.index).index("OOS 損益(pips)")
        styles[idx] = color
        idx2 = list(row.index).index("OOS 損益(円)")
        styles[idx2] = color
    except Exception:
        pass
    return styles

display_cols = [c for c in detail_df.columns if not c.startswith("_")]
styled_detail = detail_df[display_cols].style.apply(_row_style, axis=1)

st.dataframe(styled_detail, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# 採用戦略の分布
# ============================================================

st.markdown("### OOSで採用された戦略の分布")
st.caption("複数ウィンドウで同じ戦略が選ばれるほど、その戦略の優位性が安定していると言えます。")

strategy_counts: dict[str, int] = {}
for w in wf_result.windows:
    if w.best_strategy and w.best_strategy != "(なし)":
        strategy_counts[w.best_strategy] = strategy_counts.get(w.best_strategy, 0) + 1

if strategy_counts:
    sorted_strats = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
    strat_labels = [s[0] for s in sorted_strats]
    strat_counts = [s[1] for s in sorted_strats]
    bar_colors   = [
        "#26a69a" if c == max(strat_counts)
        else "#29b6f6" if c >= 2
        else "#9e9e9e"
        for c in strat_counts
    ]

    fig3 = go.Figure(go.Bar(
        x=strat_counts, y=strat_labels,
        orientation="h",
        marker_color=bar_colors,
        text=strat_counts,
        textposition="outside",
    ))
    fig3.update_layout(
        height=max(200, len(strat_labels) * 35),
        margin=dict(l=0, r=50, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=11),
                   title="採用窓数"),
        yaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# 過剰最適化スコア（判定サマリー）
# ============================================================

st.divider()
st.markdown("### 過剰最適化の判定")

eff = wf_result.oos_efficiency
win_frac = wf_result.profitable_windows / n_valid if n_valid > 0 else 0.0

if eff >= 0.7 and win_frac >= 0.6:
    verdict = "🟢 過剰最適化は少なく、**堅牢な戦略**と判断できます。"
elif eff >= 0.4 and win_frac >= 0.4:
    verdict = "🟡 一定の過剰最適化が見られます。**パラメータの再確認**を推奨します。"
else:
    verdict = "🔴 **強い過剰最適化**が疑われます。実運用には注意が必要です。"

st.markdown(f"**判定:** {verdict}")

jc1, jc2, jc3 = st.columns(3)
jc1.metric(
    "OOS効率",
    f"{eff:.2f}",
    delta="良好" if eff >= 0.7 else ("要注意" if eff >= 0.4 else "過学習"),
    delta_color="normal" if eff >= 0.7 else "inverse",
)
jc2.metric(
    "黒字窓率",
    f"{win_frac:.0%}",
    delta="良好" if win_frac >= 0.6 else ("要注意" if win_frac >= 0.4 else "不安定"),
    delta_color="normal" if win_frac >= 0.6 else "inverse",
)
jc3.metric(
    "OOS総合評価",
    "合格" if eff >= 0.7 and win_frac >= 0.6 else "要改善",
)
