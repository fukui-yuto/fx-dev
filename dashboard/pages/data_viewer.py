"""
dashboard/pages/data_viewer.py

ローカルデータ管理画面。
- MT5から全履歴を取得してSQLiteに保存
- 差分更新（最終保存日時以降だけ追加）
- 保存済みデータのプレビューと統計
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from data.local_store import delete_data, get_latest_timestamp, get_stats, query, upsert

st.markdown("## 🗂️ ローカルデータ管理")
st.caption("MT5から取得したOHLCVデータをローカルDBに保存・管理します。バックテストはここのデータを使用します。")

# ============================================================
# 保存済みデータ一覧
# ============================================================

st.markdown("### 保存済みデータ")

stats = get_stats()
if stats.empty:
    st.info("まだデータが保存されていません。下の「データ取得」でMT5からダウンロードしてください。")
else:
    st.dataframe(stats, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# データ取得
# ============================================================

st.markdown("### データ取得（MT5）")

from data.mt5_client import is_available, is_connected

if not is_available():
    st.error("MT5ライブラリが利用できません。")
elif not is_connected():
    st.warning("MT5未接続です。MetaTrader 5を起動してログインしてください。")
else:
    col_s, col_t = st.columns(2)
    with col_s:
        sel_symbols = st.multiselect(
            "通貨ペア", SUPPORTED_SYMBOLS, default=SUPPORTED_SYMBOLS[:1]
        )
    with col_t:
        sel_timeframes = st.multiselect(
            "時間足", SUPPORTED_TIMEFRAMES, default=["1H"]
        )

    col_b1, col_b2, col_b3 = st.columns(3)

    # ---- 全履歴取得 ----
    with col_b1:
        if st.button("📥 全履歴取得（最大）", use_container_width=True,
                     help="MT5が保持している最大限の履歴を取得します（初回推奨）"):
            if not sel_symbols or not sel_timeframes:
                st.warning("通貨ペアと時間足を選択してください。")
            else:
                from data.mt5_client import get_client
                client  = get_client()
                total   = 0
                progress = st.progress(0, text="取得中...")
                n = len(sel_symbols) * len(sel_timeframes)
                done = 0
                for sym in sel_symbols:
                    for tf in sel_timeframes:
                        progress.progress(done / n, text=f"取得中: {sym} {tf}")
                        try:
                            df   = client.fetch_candles_max(sym, tf)
                            cnt  = upsert(sym, tf, df)
                            total += cnt
                        except Exception as e:
                            st.warning(f"{sym} {tf}: {e}")
                        done += 1
                progress.progress(1.0, text="完了")
                st.success(f"合計 {total:,} 件を保存しました。")
                st.rerun()

    # ---- 差分更新 ----
    with col_b2:
        if st.button("🔄 差分更新", use_container_width=True,
                     help="最後に保存した日時以降のデータだけ追加します"):
            if not sel_symbols or not sel_timeframes:
                st.warning("通貨ペアと時間足を選択してください。")
            else:
                from data.mt5_client import get_client
                client   = get_client()
                total    = 0
                progress = st.progress(0, text="更新中...")
                n = len(sel_symbols) * len(sel_timeframes)
                done = 0
                for sym in sel_symbols:
                    for tf in sel_timeframes:
                        progress.progress(done / n, text=f"更新中: {sym} {tf}")
                        try:
                            latest = get_latest_timestamp(sym, tf)
                            if latest is None:
                                df = client.fetch_candles_max(sym, tf)
                            else:
                                df = client.fetch_candles_range(
                                    sym, tf, latest, datetime.now(timezone.utc)
                                )
                            cnt   = upsert(sym, tf, df)
                            total += cnt
                        except Exception as e:
                            st.warning(f"{sym} {tf}: {e}")
                        done += 1
                progress.progress(1.0, text="完了")
                st.success(f"合計 {total:,} 件を追加・更新しました。")
                st.rerun()

    # ---- 削除 ----
    with col_b3:
        if st.button("🗑️ 選択データ削除", use_container_width=True,
                     type="secondary",
                     help="選択した通貨ペア×時間足のデータをDBから削除します"):
            if not sel_symbols or not sel_timeframes:
                st.warning("通貨ペアと時間足を選択してください。")
            else:
                total = 0
                for sym in sel_symbols:
                    for tf in sel_timeframes:
                        total += delete_data(sym, tf)
                st.success(f"{total:,} 件を削除しました。")
                st.rerun()

st.divider()

# ============================================================
# データプレビュー
# ============================================================

st.markdown("### データプレビュー")

col1, col2 = st.columns(2)
with col1:
    preview_symbol = st.selectbox("通貨ペア", SUPPORTED_SYMBOLS, key="prev_sym")
with col2:
    preview_tf = st.selectbox("時間足", SUPPORTED_TIMEFRAMES, key="prev_tf", index=3)

df_preview = query(preview_symbol, preview_tf)

if df_preview.empty:
    st.info("このシンボル/時間足のデータはまだ保存されていません。")
else:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("件数",    f"{len(df_preview):,}")
    col_b.metric("開始",    str(df_preview.index.min())[:16])
    col_c.metric("終了",    str(df_preview.index.max())[:16])
    col_d.metric("最新終値", f"{df_preview['close'].iloc[-1]:.3f}")

    tab1, tab2, tab3 = st.tabs(["先頭 20件", "末尾 20件", "統計"])
    fmt = {"open": "{:.5f}", "high": "{:.5f}", "low": "{:.5f}",
           "close": "{:.5f}", "volume": "{:,.0f}"}
    with tab1:
        st.dataframe(df_preview.head(20).style.format(fmt), use_container_width=True)
    with tab2:
        st.dataframe(df_preview.tail(20).sort_index(ascending=False).style.format(fmt),
                     use_container_width=True)
    with tab3:
        st.dataframe(df_preview.describe().style.format("{:.5f}"), use_container_width=True)
