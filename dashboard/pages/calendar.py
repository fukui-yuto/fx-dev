"""
dashboard/pages/calendar.py

経済指標カレンダー画面。
- ForexFactory APIから今週・来週のイベントを表示
- 国・インパクトでフィルタリング可能
- 5分ごとに自動更新
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.calendar_utils import fetch_both_weeks_with_status, _calendar_cache

_IMPACT_EMOJI = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
_JST = timezone(timedelta(hours=9))

_DIRECTION_BADGE = {
    "bullish": ("▲", "#26a69a"),
    "bearish": ("▼", "#ef5350"),
    "neutral": ("━", "#9e9e9e"),
}


@st.fragment(run_every=600)
def _calendar_view() -> None:
    events, error_kind = fetch_both_weeks_with_status()

    # ---- コントロールバー ----
    col1, col2, col3 = st.columns([1, 2, 3])
    with col1:
        if st.button("🔄 更新", use_container_width=True):
            _calendar_cache().invalidate()
            st.rerun()
    with col2:
        selected_impacts = st.multiselect(
            "インパクト",
            ["High", "Medium", "Low"],
            default=["High", "Medium"],
            label_visibility="collapsed",
        )
    with col3:
        all_currencies = sorted({ev["country"] for ev in events if ev["country"]})
        selected_currencies = st.multiselect(
            "通貨",
            all_currencies,
            default=[c for c in ["USD", "JPY", "EUR", "GBP"] if c in all_currencies],
            label_visibility="collapsed",
        )

    if not events:
        if error_kind == "ratelimit":
            st.warning(
                "⏳ ForexFactory のレートリミットに達しました。"
                "約10分後に自動で再取得します。しばらくお待ちください。",
                icon="⚠️",
            )
        else:
            st.warning(
                "データを取得できませんでした。"
                "ネットワーク接続を確認して「🔄 更新」を押してください。",
                icon="⚠️",
            )
        return

    # ---- フィルタリング ----
    filtered = [
        ev for ev in events
        if ev["impact"] in (selected_impacts or ["High", "Medium", "Low"])
        and (not selected_currencies or ev["country"] in selected_currencies)
    ]

    now_utc = datetime.now(timezone.utc)

    # ---- サマリー ----
    upcoming = [ev for ev in filtered if ev["dt_utc"] > now_utc]
    past     = [ev for ev in filtered if ev["dt_utc"] <= now_utc]

    hi_count = sum(1 for ev in upcoming if ev["impact"] == "High")
    md_count = sum(1 for ev in upcoming if ev["impact"] == "Medium")

    c1, c2, c3 = st.columns(3)
    c1.metric("今後の予定", f"{len(upcoming)}件")
    c2.metric("🔴 高インパクト", f"{hi_count}件")
    c3.metric("🟠 中インパクト", f"{md_count}件")

    st.divider()

    # ---- 今後のイベント ----
    tab1, tab2 = st.tabs(["📅 今後の予定", "📋 過去のイベント（結果あり）"])

    with tab1:
        if not upcoming:
            st.info("フィルター条件に一致する今後のイベントはありません。")
        else:
            _render_events(upcoming, now_utc, show_countdown=True)

    with tab2:
        if not past:
            st.info("フィルター条件に一致する過去のイベントはありません。")
        else:
            _render_events(list(reversed(past)), now_utc, show_countdown=False)


def _render_events(events: list[dict], now_utc: datetime, show_countdown: bool) -> None:
    for ev in events:
        dt_jst   = ev["dt_utc"].astimezone(_JST)
        impact   = ev["impact"]
        emoji    = _IMPACT_EMOJI.get(impact, "⚪")
        date_str = dt_jst.strftime("%m/%d(%a) %H:%M JST")

        if show_countdown:
            delta = ev["dt_utc"] - now_utc
            total_s = int(delta.total_seconds())
            if total_s < 3600:
                countdown = f"あと {total_s // 60}分"
            elif total_s < 86400:
                countdown = f"あと {total_s // 3600}時間{(total_s % 3600) // 60}分"
            else:
                countdown = f"あと {total_s // 86400}日"
        else:
            countdown = ""

        if not show_countdown and ev["actual"] != "-":
            direction = ev.get("direction", "neutral")
            mark, color = _DIRECTION_BADGE[direction]
            badge = (f"<span style='background:{color}33;color:{color};"
                     f"padding:1px 6px;border-radius:3px;font-size:0.8em;"
                     f"font-weight:bold'>{mark}</span> ")
        else:
            badge = ""
        title_line = (
            f"{emoji} {badge}**{ev['country']}** {ev['title']}"
            f"　<span style='color:#9e9e9e;font-size:0.85em'>{date_str}"
            f"{f'　{countdown}' if countdown else ''}</span>"
        )
        st.markdown(title_line, unsafe_allow_html=True)

        meta_cols = st.columns(3)
        meta_cols[0].caption(f"予想: {ev['forecast']}")
        meta_cols[1].caption(f"前回: {ev['previous']}")
        if ev["actual"] != "-":
            direction = ev.get("direction", "neutral")
            mark, color = _DIRECTION_BADGE[direction]
            meta_cols[2].markdown(
                f"<span style='font-size:0.8em;color:#aaa'>結果: </span>"
                f"<span style='font-weight:bold'>{ev['actual']}</span> "
                f"<span style='color:{color};font-weight:bold;font-size:1.1em'>{mark}</span>"
                f"<span style='font-size:0.75em;color:{color}'> vs 予想 {ev['forecast']}</span>",
                unsafe_allow_html=True,
            )
        else:
            if show_countdown:
                meta_cols[2].caption("結果: 未発表")
            else:
                meta_cols[2].caption("結果: -")


# ============================================================
# ページ本体
# ============================================================

st.markdown("## 📅 経済指標カレンダー")
st.caption("ForexFactory より取得。5分ごとに自動更新されます。")

_calendar_view()
