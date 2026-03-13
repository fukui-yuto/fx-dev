"""
dashboard/pages/learning.py

AI 学習状況モニタリングページ。

時間足ごとに独立して蓄積したフィードバックデータを可視化する。
- 時間足タブ（1M / 5M / 15M … ）
- 各タブ: 勝率・PF・累積損益・移動平均勝率・Long/Short 別精度
- 時間足間の比較テーブル
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import SUPPORTED_TIMEFRAMES
from dashboard.ai_learner import (
    load_feedback, get_stats, list_feedback_timeframes,
    _feedback_file,
)

st.markdown("## 🤖 AI 学習状況")
st.caption(
    "チャート画面で「AI シグナル (k-NN)」を有効にすると、"
    "時間足ごとに自動評価・学習が蓄積されます。"
)

# ============================================================
# データ収集（全時間足）
# ============================================================

existing_tfs = list_feedback_timeframes()

# 時間足の表示順を SUPPORTED_TIMEFRAMES に合わせる
ordered_tfs = [tf for tf in SUPPORTED_TIMEFRAMES if tf in existing_tfs]
# 万が一 settings に未定義の TF があれば末尾に追加
for tf in existing_tfs:
    if tf not in ordered_tfs:
        ordered_tfs.append(tf)

if not ordered_tfs:
    st.info(
        "まだ学習データがありません。\n\n"
        "チャート画面で「AI シグナル (k-NN)」を有効にしてしばらく使用すると、"
        "シグナルの結果が時間足ごとに自動記録されます。"
    )
    st.stop()

# ============================================================
# 時間足間比較テーブル（サマリー）
# ============================================================

summary_rows = []
all_stats: dict[str, dict] = {}

for tf in ordered_tfs:
    fb   = load_feedback(timeframe=tf)
    stat = get_stats(fb)
    all_stats[tf] = stat
    if stat["total"] == 0:
        continue
    pf = stat.get("profit_factor", 0.0)  # get_stats に profit_factor がない場合の保険
    summary_rows.append({
        "時間足":      tf,
        "学習件数":    stat["total"],
        "全体勝率%":   round(stat["win_rate"]  * 100, 1),
        "直近勝率%":   round(stat["recent_wr"] * 100, 1),
        "Long勝率%":   round(stat["long_wr"]   * 100, 1),
        "Short勝率%":  round(stat["short_wr"]  * 100, 1),
        "平均損益(p)": round(stat["avg_pips"], 2),
        "トレンド":    {"improving": "📈 改善中", "declining": "📉 低下中",
                       "stable": "➡️ 安定"}.get(stat["trend"], "➡️ 安定"),
    })

if summary_rows:
    st.subheader("時間足別サマリー")
    df_sum = pd.DataFrame(summary_rows)

    def _style_wr(val):
        try:
            v = float(val)
            if v >= 60: return "color:#26a69a;font-weight:bold"
            if v <= 45: return "color:#ef5350"
        except Exception:
            pass
        return ""

    def _style_pips(val):
        try:
            v = float(val)
            if v > 0: return "color:#26a69a"
            if v < 0: return "color:#ef5350"
        except Exception:
            pass
        return ""

    styled = (
        df_sum.style
        .map(_style_wr,   subset=["全体勝率%", "直近勝率%", "Long勝率%", "Short勝率%"])
        .map(_style_pips, subset=["平均損益(p)"])
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.divider()


# ============================================================
# 時間足タブ
# ============================================================

tabs = st.tabs([f"{tf}（{all_stats[tf]['total']}件）" for tf in ordered_tfs])

for tab, tf in zip(tabs, ordered_tfs):
    with tab:
        feedback = load_feedback(timeframe=tf)
        stats    = all_stats[tf]

        if stats["total"] == 0:
            st.info(f"{tf} 足の学習データはまだありません。")
            continue

        # ---- メトリクス ----
        trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(
            stats["trend"], "➡️")
        trend_text = {"improving": "改善中", "declining": "低下中", "stable": "安定"}.get(
            stats["trend"], "安定")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("学習件数",     f"{stats['total']} 件")
        c2.metric("全体勝率",     f"{stats['win_rate'] * 100:.1f}%",
                  delta=f"{(stats['win_rate'] - 0.5) * 100:+.1f}%")
        c3.metric("直近20件勝率", f"{stats['recent_wr'] * 100:.1f}%")
        c4.metric(f"トレンド {trend_icon}", trend_text)

        c5, c6, c7 = st.columns(3)
        c5.metric("Long 勝率",  f"{stats['long_wr']  * 100:.1f}%")
        c6.metric("Short 勝率", f"{stats['short_wr'] * 100:.1f}%")
        c7.metric("平均損益",   f"{stats['avg_pips']:+.1f} pips")

        st.divider()

        # ---- 損益推移グラフ ----
        st.subheader("累積損益推移 (pips)")

        sorted_fb = sorted(feedback, key=lambda x: x["ts"])
        cumulative = 0.0
        rows = []
        for i, r in enumerate(sorted_fb):
            cumulative += r["pips"]
            rows.append({
                "No.":        i + 1,
                "累積(pips)": round(cumulative, 2),
                "方向":       "Long" if r["direction"] == 1 else "Short",
                "結果":       "✓" if r["correct"] else "✗",
                "損益(pips)": r["pips"],
            })

        if rows:
            df_chart = pd.DataFrame(rows).set_index("No.")
            st.line_chart(df_chart["累積(pips)"], use_container_width=True)

        # ---- 勝率推移（移動平均）----
        st.subheader("勝率推移（直近20件移動平均）")
        if len(rows) >= 5:
            win_series = pd.Series(
                [1.0 if r["correct"] else 0.0 for r in sorted_fb]
            ).rolling(min(20, len(sorted_fb))).mean() * 100
            st.line_chart(win_series.rename("勝率(%)"), use_container_width=True)

        # ---- Long / Short 別損益 ----
        st.subheader("Long / Short 別累積損益")
        long_rows  = [r for r in sorted_fb if r["direction"] ==  1]
        short_rows = [r for r in sorted_fb if r["direction"] == -1]

        if long_rows or short_rows:
            _lc, _sc = st.columns(2)
            with _lc:
                if long_rows:
                    l_cum = 0.0
                    l_series = []
                    for i, r in enumerate(long_rows):
                        l_cum += r["pips"]
                        l_series.append({"No.": i + 1, "Long累積(pips)": round(l_cum, 2)})
                    st.caption(f"Long: {len(long_rows)}件 / 勝率 {stats['long_wr']*100:.1f}%")
                    st.line_chart(
                        pd.DataFrame(l_series).set_index("No.")["Long累積(pips)"],
                        use_container_width=True,
                    )
                else:
                    st.caption("Long データなし")

            with _sc:
                if short_rows:
                    s_cum = 0.0
                    s_series = []
                    for i, r in enumerate(short_rows):
                        s_cum += r["pips"]
                        s_series.append({"No.": i + 1, "Short累積(pips)": round(s_cum, 2)})
                    st.caption(f"Short: {len(short_rows)}件 / 勝率 {stats['short_wr']*100:.1f}%")
                    st.line_chart(
                        pd.DataFrame(s_series).set_index("No.")["Short累積(pips)"],
                        use_container_width=True,
                    )
                else:
                    st.caption("Short データなし")

        # ---- 詳細テーブル ----
        st.subheader("シグナル詳細")
        with st.expander("全データを表示"):
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        # ---- 学習データ管理 ----
        st.subheader("学習データ管理")
        col_info, col_reset = st.columns([3, 1])
        _fpath = _feedback_file(tf)
        col_info.caption(
            f"保存先: `{_fpath}`  \n"
            f"保存件数: {stats['total']} / 1000 件  \n"
            "古いデータは自動的に削除されます（最新1000件を保持）。"
        )
        # 誤操作防止のため確認チェックボックスを挟む
        _confirm_key = f"confirm_reset_{tf}"
        if col_reset.checkbox(f"削除確認", key=_confirm_key,
                              help="チェックを入れるとリセットボタンが有効になります"):
            if col_reset.button(f"🗑️ {tf} をリセット",
                                key=f"reset_{tf}", type="secondary"):
                if _fpath.exists():
                    _fpath.unlink()
                st.success(f"{tf} の学習データをリセットしました。")
                st.rerun()
        else:
            col_reset.caption("↑ チェックで有効化")
