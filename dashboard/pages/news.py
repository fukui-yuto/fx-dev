"""
dashboard/pages/news.py

USD/JPY ファンダメンタル分析画面。
- 5分ごとに自動更新（@st.fragment run_every=300）
- 並列取得＆分析済みデータを表示
- 方向でフィルタリング可能
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.news_utils import fetch_and_analyze_news, time_ago

_DIR_LABEL  = {"up": "▲ 上昇示唆", "down": "▼ 下落示唆", "neutral": "━ 中立"}
_DIR_COLOR  = {"up": "#26a69a",    "down": "#ef5350",    "neutral": "#9e9e9e"}
_DIR_BG     = {"up": "#26a69a22",  "down": "#ef535022",  "neutral": "#9e9e9e22"}
_IMPACT_BAR = {1: "●○○○○", 2: "●●○○○", 3: "●●●○○", 4: "●●●●○", 5: "●●●●●"}
_IMPACT_LBL = {1: "軽微", 2: "低", 3: "中", 4: "高", 5: "重大"}

st.markdown("## 📰 USD/JPY ファンダメンタル分析")


# ============================================================
# 全体を fragment に包む → 5分ごとに自動更新
# ============================================================

@st.fragment(run_every=300)
def _news_main() -> None:
    import time

    # ---- コントロールバー ----
    ctrl_cols = st.columns([1, 1, 4])
    with ctrl_cols[0]:
        if st.button("🔄 今すぐ更新", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with ctrl_cols[1]:
        filter_dir = st.selectbox(
            "方向フィルター",
            ["全て", "上昇示唆 ▲", "下落示唆 ▼", "中立 ━"],
            label_visibility="collapsed",
            key="news_filter",
        )

    remaining = 300 - (int(time.time()) % 300)
    st.caption(f"5分ごとに自動更新　｜　次の更新まで {remaining}秒")

    dir_map    = {"全て": None, "上昇示唆 ▲": "up", "下落示唆 ▼": "down", "中立 ━": "neutral"}
    target_dir = dir_map[filter_dir]

    # ---- データ取得（キャッシュから or 並列フェッチ）----
    with st.spinner("ニュースを取得・分析中..."):
        items = fetch_and_analyze_news()

    if not items:
        st.warning("ニュースを取得できませんでした。インターネット接続を確認してください。")
        return

    # ---- 統計サマリー ----
    analyses   = [it["analysis"] for it in items]
    up_count   = sum(1 for a in analyses if a["direction"] == "up")
    down_count = sum(1 for a in analyses if a["direction"] == "down")
    total      = len(analyses)
    sentiment  = "強気優勢" if up_count > down_count else "弱気優勢" if down_count > up_count else "拮抗"
    sent_color = "#26a69a" if up_count > down_count else "#ef5350" if down_count > up_count else "#9e9e9e"

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("取得記事数", total)
    s2.metric("上昇示唆", f"{up_count}件",
              delta=f"{up_count/total*100:.0f}%" if total else None)
    s3.metric("下落示唆", f"{down_count}件",
              delta=f"-{down_count/total*100:.0f}%" if total else None, delta_color="inverse")
    s4.markdown("**総合センチメント**")
    s4.markdown(
        f"<span style='color:{sent_color};font-size:1.2em;font-weight:bold'>{sentiment}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- ニュースカード ----
    shown = 0
    for it in items:
        analysis  = it["analysis"]
        direction = analysis.get("direction", "neutral")
        if target_dir and direction != target_dir:
            continue

        impact     = analysis.get("impact", 1)
        color      = _DIR_COLOR[direction]
        bg         = _DIR_BG[direction]
        label      = _DIR_LABEL[direction]
        bar        = _IMPACT_BAR.get(impact, "●○○○○")
        imp_lbl    = _IMPACT_LBL.get(impact, "")
        title_ja   = analysis.get("title_ja") or ""
        title_main = title_ja if title_ja else it["title"]
        title_sub  = (
            f"<div style='font-size:0.75em;color:#666;margin-top:2px'>{it['title']}</div>"
            if title_ja and title_ja != it["title"] else ""
        )

        st.markdown(
            f"""
            <div style="background:{bg};border-left:4px solid {color};
                        border-radius:6px;padding:12px 16px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;
                            align-items:flex-start;gap:12px">
                    <div style="flex:1">
                        <a href="{it['link']}" target="_blank"
                           style="color:#f0f2f6;font-weight:bold;font-size:0.95em;
                                  text-decoration:none">{title_main}</a>
                        {title_sub}
                        <div style="margin-top:4px;font-size:0.8em;color:#9e9e9e">
                            📡 {it['source']}　🕐 {time_ago(it['published'])}
                        </div>
                        {"<div style='margin-top:6px;font-size:0.85em;color:#ccc'>💡 " + analysis['summary'] + "</div>" if analysis.get('summary') else ""}
                        {"<div style='font-size:0.78em;color:#888;margin-top:2px'>根拠: " + analysis['reason'] + "</div>" if analysis.get('reason') else ""}
                    </div>
                    <div style="text-align:center;min-width:80px">
                        <div style="color:{color};font-size:1.4em;font-weight:bold">{label.split()[0]}</div>
                        <div style="color:{color};font-size:0.75em">{label[2:]}</div>
                        <div style="margin-top:4px;font-size:0.85em;letter-spacing:1px">{bar}</div>
                        <div style="font-size:0.72em;color:#888">影響度 {imp_lbl}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        shown += 1
        if shown >= 30:
            break

    if shown == 0:
        st.info("該当するニュースがありません。")


_news_main()
