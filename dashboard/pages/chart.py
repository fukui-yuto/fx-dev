"""
dashboard/pages/chart.py

チャート画面。
- 1 / 2 / 4 分割レイアウト（ラジオ選択）
- パネルごとに通貨ペア・時間足・インジケーターを独立設定
- 設定は session_state に保持（分割数を変えても復元される）
- 起動時に設定ファイルから復元、変更時に自動保存
- LightweightCharts HTML は設定変更時のみ再生成（ズーム保持）
- fragment が毎秒 JSON を更新 → チャートJS がポーリング
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES, SYMBOL_GROUPS
from dashboard.sample_data import get_ohlcv_dataframe
from dashboard.chart_utils import build_panel_html, write_panel_json
from dashboard.indicators import INDICATOR_OPTIONS, calculate

# 設定保存先
_SETTINGS_FILE = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "chart_settings.json"

# シグナルパラメーター保存先（チューニングページと共用）
_PARAMS_FILE = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "signal_params.json"


def _load_signal_params() -> dict:
    """チューニングページで保存したシグナルパラメーターを読み込む。"""
    if _PARAMS_FILE.exists():
        try:
            return json.loads(_PARAMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"entry": {}, "ai": {}}

# 保存対象キー（パネル0〜3の全設定 + layout）
_SAVE_KEYS = ["layout"] + [
    f"p{i}_{k}"
    for i in range(4)
    for k in ("symbol", "timeframe", "n_bars", "indicators", "show_ind", "show_markers", "notify", "cvd_scale")
]

# ============================================================
# 設定の永続化
# ============================================================

def _load_settings() -> None:
    """起動時に設定ファイルから session_state へ復元する。"""
    if not _SETTINGS_FILE.exists():
        return
    try:
        saved = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        for key, val in saved.items():
            if key not in st.session_state:
                st.session_state[key] = val
    except Exception:
        pass


def _save_settings() -> None:
    """現在の session_state を設定ファイルへ書き込む。"""
    data = {k: st.session_state[k] for k in _SAVE_KEYS if k in st.session_state}
    try:
        _SETTINGS_FILE.parent.mkdir(exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# ============================================================
# ヘルパー
# ============================================================

def _panel_html_key(panel_id: int, symbol: str, timeframe: str, n_bars: int, indicators: list, show_ind: bool = True, cvd_scale: int = 1) -> str:
    s = f"{panel_id}|{symbol}|{timeframe}|{n_bars}|{'|'.join(sorted(indicators))}|{show_ind}|cvd{cvd_scale}"
    return "ph_" + hashlib.md5(s.encode()).hexdigest()[:12]


def _default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


def _panel_chart_height(layout: int) -> int:
    return {1: 520, 2: 460, 4: 320}[layout]


def _component_height(layout: int) -> int:
    return _panel_chart_height(layout) + 18


# ============================================================
# 起動時に設定を復元・静的JSONをリセット
# ============================================================

_load_settings()

# 旧サイズ(8192B)のパネルJSONが残っていると書き込み後に旧データが混入するため削除する
def _reset_panel_json() -> None:
    from dashboard.chart_utils import _STATIC_DIR
    for _p in (_STATIC_DIR).glob("panel_*.json"):
        try:
            _p.unlink()
        except Exception:
            pass

if "panel_json_reset" not in st.session_state:
    _reset_panel_json()
    st.session_state["panel_json_reset"] = True

# ============================================================
# サイドバー
# ============================================================

with st.sidebar:
    # ---- クイックリンク ----
    st.markdown("### 🔗 クイックリンク")
    _ql1, _ql2 = st.columns(2)
    _ql1.page_link("dashboard/pages/backtest.py", label="📊 BT",      use_container_width=True)
    _ql2.page_link("dashboard/pages/news.py",     label="📰 ニュース", use_container_width=True)

    st.divider()
    st.markdown("### 📊 表示設定")

    # ---- 通貨ペア一括変更 ----
    # カウンターでウィジェットキーを変えることで選択後にリセットする
    _bc = st.session_state.get("_bulk_counter", 0)

    _group_options = ["（グループ選択）"] + list(SYMBOL_GROUPS.keys())
    _bulk_group    = st.selectbox("グループ一括変更", _group_options, key=f"bulk_group_{_bc}")

    if _bulk_group != "（グループ選択）":
        _group_symbols = SYMBOL_GROUPS[_bulk_group]
        _bulk_sym      = st.selectbox(
            "通貨ペア", ["（ペア選択）"] + _group_symbols, key=f"bulk_sym_{_bc}"
        )
        if _bulk_sym != "（ペア選択）":
            for _pi in range(4):
                st.session_state[f"p{_pi}_symbol"] = _bulk_sym
            st.session_state["_bulk_counter"] = _bc + 1  # キーを変えてウィジェットをリセット
            _save_settings()
            st.rerun()

    # 分割数（保存値があれば index を合わせる）
    _layout_options = [1, 2, 4]
    _saved_layout   = st.session_state.get("layout", 1)
    _layout_index   = _layout_options.index(_saved_layout) if _saved_layout in _layout_options else 0

    layout = st.radio("分割", _layout_options, horizontal=True,
                      key="layout", index=_layout_index)
    panel_count = layout

    st.divider()

    # パネルごとの設定
    for i in range(panel_count):
        _default(f"p{i}_symbol",     SUPPORTED_SYMBOLS[0])
        _default(f"p{i}_timeframe",  SUPPORTED_TIMEFRAMES[3])  # 1H
        _default(f"p{i}_n_bars",     200)
        _default(f"p{i}_indicators", [])
        _default(f"p{i}_show_ind",     True)
        _default(f"p{i}_show_markers", True)
        _default(f"p{i}_notify",       False)
        _default(f"p{i}_cvd_scale",    1)

        label = f"パネル {i+1}" if panel_count > 1 else "📈 パネル設定"
        with st.expander(label, expanded=(panel_count == 1)):
            st.session_state[f"p{i}_symbol"] = st.selectbox(
                "通貨ペア", SUPPORTED_SYMBOLS,
                index=SUPPORTED_SYMBOLS.index(st.session_state[f"p{i}_symbol"]),
                key=f"sb_sym_{i}")
            st.session_state[f"p{i}_timeframe"] = st.selectbox(
                "時間足", SUPPORTED_TIMEFRAMES,
                index=SUPPORTED_TIMEFRAMES.index(st.session_state[f"p{i}_timeframe"]),
                key=f"sb_tf_{i}")
            st.session_state[f"p{i}_n_bars"] = st.slider(
                "取得本数", 50, 1000,
                value=st.session_state[f"p{i}_n_bars"],
                step=50, key=f"sb_nb_{i}")
            # 保存設定に廃止済みオプションが残っている場合に除去
            _saved_ind = [x for x in st.session_state[f"p{i}_indicators"] if x in INDICATOR_OPTIONS]
            st.session_state[f"p{i}_indicators"] = st.multiselect(
                "インジケーター", INDICATOR_OPTIONS,
                default=_saved_ind,
                key=f"sb_ind_{i}")
            _tog_c1, _tog_c2, _tog_c3 = st.columns(3)
            st.session_state[f"p{i}_show_ind"] = _tog_c1.toggle(
                "インジ表示",
                value=st.session_state[f"p{i}_show_ind"],
                key=f"sb_show_ind_{i}")
            st.session_state[f"p{i}_show_markers"] = _tog_c2.toggle(
                "売買マーカー",
                value=st.session_state[f"p{i}_show_markers"],
                key=f"sb_show_markers_{i}")
            st.session_state[f"p{i}_notify"] = _tog_c3.toggle(
                "🔔 通知",
                value=st.session_state[f"p{i}_notify"],
                key=f"sb_notify_{i}",
                help="エントリー・エグジットシグナル発生時にトースト通知を表示します")
            if "累積出来高デルタ (CVD)" in st.session_state[f"p{i}_indicators"]:
                st.session_state[f"p{i}_cvd_scale"] = st.slider(
                    "CVD デルタ倍率",
                    min_value=1, max_value=200,
                    value=int(st.session_state[f"p{i}_cvd_scale"]),
                    step=1, key=f"sb_cvd_scale_{i}",
                    help="デルタバーの高さを調整します。大きくするほどバーが見やすくなります")

    st.divider()
    st.markdown("### 📐 トレードツール")

    # ---- RR 計算ツール ----
    with st.expander("📐 RR 計算ツール", expanded=False):
        _sym0 = st.session_state.get("p0_symbol", SUPPORTED_SYMBOLS[0])
        _is_jpy  = _sym0.endswith("JPY")
        _pip_sz  = 0.01 if _is_jpy else 0.0001
        _fmt     = "%.3f" if _is_jpy else "%.5f"
        rr_entry = st.number_input("エントリー", value=0.0, format=_fmt, key="rr_entry")
        rr_sl    = st.number_input("SL",         value=0.0, format=_fmt, key="rr_sl")
        rr_tp    = st.number_input("TP",         value=0.0, format=_fmt, key="rr_tp")
        rr_lot   = st.number_input("ロット（通貨）", value=10000, step=1000, key="rr_lot")
        if rr_entry > 0 and rr_sl > 0 and rr_tp > 0:
            sl_pips     = abs(rr_entry - rr_sl) / _pip_sz
            tp_pips     = abs(rr_tp    - rr_entry) / _pip_sz
            rr_ratio    = tp_pips / sl_pips if sl_pips > 0 else 0.0
            loss_jpy    = sl_pips * _pip_sz * rr_lot
            profit_jpy  = tp_pips * _pip_sz * rr_lot
            rr_c1, rr_c2 = st.columns(2)
            rr_c1.metric("SL", f"{sl_pips:.1f}pips", f"-¥{loss_jpy:,.0f}")
            rr_c2.metric("TP", f"{tp_pips:.1f}pips", f"+¥{profit_jpy:,.0f}")
            _rr_ok = rr_ratio >= 2.0
            st.metric(
                "リスクリワード比",
                f"1 : {rr_ratio:.2f}",
                delta="良好 (≥2.0)" if _rr_ok else "要検討 (<2.0)",
                delta_color="normal" if _rr_ok else "inverse",
            )

    # ---- ポジションサイズ計算ツール ----
    with st.expander("💰 ポジションサイズ計算", expanded=False):
        _sym0_ps   = st.session_state.get("p0_symbol", SUPPORTED_SYMBOLS[0])
        _is_jpy_ps = _sym0_ps.endswith("JPY")
        _pip_sz_ps = 0.01 if _is_jpy_ps else 0.0001
        ps_balance  = st.number_input("口座残高（円）", value=500_000, step=10_000, min_value=1_000, key="ps_balance")
        ps_risk_pct = st.slider("リスク率（%）", 0.5, 5.0, 1.0, 0.5, key="ps_risk_pct")
        ps_sl_pips  = st.number_input("SL（pips）", value=10.0, step=0.5, min_value=0.1, key="ps_sl_pips")
        if not _is_jpy_ps:
            ps_usdjpy = st.number_input("USDJPY レート", value=150.0, step=1.0,
                                        min_value=50.0, max_value=300.0, key="ps_usdjpy",
                                        help="ドルストレートの円換算に使用")
        else:
            ps_usdjpy = 150.0
        if ps_sl_pips > 0 and ps_balance > 0:
            risk_jpy      = ps_balance * ps_risk_pct / 100
            pip_val_1unit = _pip_sz_ps if _is_jpy_ps else _pip_sz_ps * ps_usdjpy
            lot_units     = risk_jpy / (ps_sl_pips * pip_val_1unit) if ps_sl_pips > 0 else 0
            lot_10k       = lot_units / 10_000
            tp_2r_pips    = ps_sl_pips * 2
            tp_3r_pips    = ps_sl_pips * 3
            ps_c1, ps_c2  = st.columns(2)
            ps_c1.metric("推奨ロット", f"{lot_10k:.2f}万通貨", f"{lot_units:,.0f}通貨")
            ps_c2.metric("リスク金額", f"¥{risk_jpy:,.0f}", f"SL {ps_sl_pips:.1f}pips")
            st.caption(
                f"TP目安: 1:2RR → **{tp_2r_pips:.1f}pips**　|　"
                f"1:3RR → **{tp_3r_pips:.1f}pips**"
            )

    st.divider()

    # MT5 接続ステータス
    from data.mt5_client import is_connected, is_available
    if not is_available():
        st.error("MT5 ライブラリなし", icon="❌")
        connected = False
    elif is_connected():
        st.success("MT5 接続中", icon="✅")
        connected = True
        if st.button("MT5 再接続", use_container_width=True):
            from data.mt5_client import reset_client
            reset_client()
            st.cache_data.clear()
            st.rerun()
    else:
        st.warning("MT5 未接続 — 自動接続中…", icon="⚠️")
        st.caption("MetaTrader 5 を起動してログインしてください。")
        connected = False

# サイドバー操作後に設定を保存
_save_settings()

# ============================================================
# パネル設定を読み込む
# ============================================================

port        = int(st.get_option("server.port"))
chart_h     = _panel_chart_height(panel_count)
component_h = _component_height(panel_count)

panel_cfgs = [
    {
        "symbol":       st.session_state[f"p{i}_symbol"],
        "timeframe":    st.session_state[f"p{i}_timeframe"],
        "n_bars":       st.session_state[f"p{i}_n_bars"],
        "indicators":   st.session_state[f"p{i}_indicators"],
        "show_ind":     st.session_state[f"p{i}_show_ind"],
        "show_markers": st.session_state.get(f"p{i}_show_markers", True),
        "notify":       st.session_state.get(f"p{i}_notify", False),
        "cvd_scale":    int(st.session_state.get(f"p{i}_cvd_scale", 1)),
    }
    for i in range(panel_count)
]

# ============================================================
# チャートHTML描画（設定変更時のみ再生成）
# ============================================================

def _render_panel_html(panel_id: int, cfg: dict) -> None:
    """HTMLキャッシュを確認して描画する。設定変更時のみ再生成。"""
    from dashboard.chart_utils import JST_OFFSET
    eff_ind   = cfg["indicators"] if cfg.get("show_ind", True) else []
    cvd_scale = int(cfg.get("cvd_scale", 1))
    html_key  = _panel_html_key(
        panel_id, cfg["symbol"], cfg["timeframe"], cfg["n_bars"], cfg["indicators"],
        show_ind=cfg.get("show_ind", True), cvd_scale=cvd_scale,
    )
    if html_key not in st.session_state:
        df, _ = get_ohlcv_dataframe(cfg["symbol"], cfg["timeframe"], count=cfg["n_bars"])
        ind   = calculate(df, eff_ind, JST_OFFSET)

        # マーカー系を ind から抽出して initial_events に含める（即時表示のため）
        # LightweightCharts は時刻昇順ソートを要求するのでソートしてから渡す
        _marker_keys = ("Session_markers", "Divergence_markers", "Pattern_markers",
                        "Entry_markers")
        initial_events: list[dict] = []
        for _mk in _marker_keys:
            if _mk in ind:
                _ev = ind[_mk].get("data", [])
                initial_events.extend(_ev)
        initial_events.sort(key=lambda e: e["time"])

        write_panel_json(df, panel_id, ind, events=initial_events, cvd_scale=cvd_scale)
        st.session_state[html_key] = build_panel_html(
            df, port, panel_id, cfg["symbol"], ind, height=chart_h,
            initial_events=initial_events, cvd_scale=cvd_scale,
        )
        # 同パネルIDの古いHTMLキャッシュを削除
        to_del = [
            k for k in st.session_state
            if k.startswith("ph_") and k != html_key
            and st.session_state.get(f"_ph_panel_{k}") == panel_id
        ]
        for k in to_del:
            del st.session_state[k]
        st.session_state[f"_ph_panel_{html_key}"] = panel_id

    components.html(st.session_state[html_key], height=component_h)


def _panel_title(i: int, cfg: dict) -> str:
    """パネルタイトル文字列を返す（接続ステータス付き）。"""
    from data.mt5_client import is_connected, is_available
    if not is_available():
        conn = "❌ MT5なし"
    elif is_connected():
        conn = "🟢 MT5"
    else:
        conn = "🟡 サンプル"
    return f"{cfg['symbol']} / {cfg['timeframe']}　{conn}"


# ---- 1分割 ----
if panel_count == 1:
    st.markdown(f"### {_panel_title(0, panel_cfgs[0])}")
    _render_panel_html(0, panel_cfgs[0])

# ---- 2分割 ----
elif panel_count == 2:
    cols = st.columns(2)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{_panel_title(i, panel_cfgs[i])}**")
            _render_panel_html(i, panel_cfgs[i])

# ---- 4分割 ----
else:
    for row in range(2):
        cols = st.columns(2)
        for col_idx, col in enumerate(cols):
            i = row * 2 + col_idx
            with col:
                st.markdown(f"**{_panel_title(i, panel_cfgs[i])}**")
                _render_panel_html(i, panel_cfgs[i])

# ============================================================
# メトリクス + JSON更新フラグメント（1秒ごと）
# ============================================================

@st.fragment(run_every="1s")
def panel_fragment(panel_id: int, symbol: str, timeframe: str, n_bars: int, indicators: list, show_ind: bool = True, show_markers: bool = True, notify: bool = False, cvd_scale: int = 1) -> None:
    from datetime import datetime, timezone
    from dashboard.chart_utils import JST_OFFSET
    df, source = get_ohlcv_dataframe(symbol, timeframe, count=n_bars)

    # インジケーター計算（非表示時は空リスト）
    # エントリーシグナルは新バー形成時のみ再計算（毎秒の計算コストを削減）
    eff_ind = indicators if show_ind else []
    has_entry = "エントリーシグナル" in eff_ind
    eff_ind_fast = [x for x in eff_ind if x != "エントリーシグナル"]
    ind = calculate(df, eff_ind_fast, JST_OFFSET)

    # ---- auto-tuned エントリー・エグジットマーカー＋集計（新バー形成時のみ再計算）----
    autotune_markers: list[dict] = []
    _at_stats: dict = {}
    _at_bar_key = f"_at_{panel_id}_{str(df.index[-1])}"
    if _at_bar_key not in st.session_state:
        for _k in [k for k in st.session_state if k.startswith(f"_at_{panel_id}_")]:
            del st.session_state[_k]
        try:
            from dashboard.signal_engine import get_autotune_markers, get_autotune_summary
            _markers = get_autotune_markers(symbol, timeframe, df, jst_offset=JST_OFFSET)
            _stats   = get_autotune_summary(symbol, timeframe, df)
        except Exception:
            _markers, _stats = [], {}
        st.session_state[_at_bar_key] = {"markers": _markers, "stats": _stats}
    _at_cache = st.session_state[_at_bar_key]
    if show_markers:
        autotune_markers = _at_cache.get("markers", [])
    _at_stats = _at_cache.get("stats", {})

    # ---- 旧エントリーシグナル（ユーザーが明示選択した場合のみ）----
    # auto-tuned マーカーと重複しないよう、ユーザーが「エントリーシグナル」を
    # 選択した場合のみ旧スコアリングの矢印も追加表示する
    if has_entry:
        from dashboard.indicators import calc_entry_signals
        from dashboard.chart_utils import JST_OFFSET as _JST
        from config.signal_defaults import get_entry_params
        _sig_params   = _load_signal_params()
        _entry_params = get_entry_params(timeframe, _sig_params.get("entry"))
        _bar_key = f"_ec_{panel_id}_{str(df.index[-1])}"
        if _bar_key not in st.session_state:
            for _k in [k for k in st.session_state if k.startswith(f"_ec_{panel_id}_")]:
                del st.session_state[_k]
            st.session_state[_bar_key] = calc_entry_signals(df, _JST, params=_entry_params)
        ind["Entry_markers"] = {"type": "entry", "data": st.session_state[_bar_key]}

    # セッション区切り・ダイバージェンス・パターンマーカーを ind から抽出してイベントに合流
    session_markers: list[dict] = []
    if "Session_markers" in ind:
        session_markers = ind.pop("Session_markers").get("data", [])

    divergence_markers: list[dict] = []
    if "Divergence_markers" in ind:
        divergence_markers = ind.pop("Divergence_markers").get("data", [])

    pattern_markers: list[dict] = []
    if "Pattern_markers" in ind:
        pattern_markers = ind.pop("Pattern_markers").get("data", [])

    entry_markers: list[dict] = []
    if "Entry_markers" in ind:
        entry_markers = ind.pop("Entry_markers").get("data", [])

    # 経済指標マーカー取得（5分キャッシュ済みなので100ms毎呼び出しも軽量）
    try:
        from dashboard.calendar_utils import get_high_impact_for_symbols
        econ_events = get_high_impact_for_symbols([symbol], JST_OFFSET)
    except Exception:
        econ_events = []

    # auto-tuned マーカーを常に表示（旧エントリーシグナルは選択時のみ追加）
    events = (
        session_markers + divergence_markers + pattern_markers
        + autotune_markers + entry_markers + econ_events
    )

    # ライブシグナルのSL/TPライン＆リアルタイムマーカーを計算
    _signal_lines: dict | None = None
    _live_marker: list[dict] = []
    try:
        from dashboard.signal_engine import get_live_signal
        from dashboard.chart_utils import JST_OFFSET as _JST2
        _sig = get_live_signal(symbol, timeframe, df)
        if _sig["direction"] != "neutral":
            _signal_lines = {
                "direction": _sig["direction"],
                "entry":     _sig["entry"],
                "sl":        _sig["sl"],
                "tp":        _sig["tp"],
                "sl_pips":   _sig["sl_pips"],
                "tp_pips":   _sig["tp_pips"],
            }
            # 最新足にリアルタイムエントリーマーカーを追加（歴史マーカーと色/サイズで区別）
            _is_long  = _sig["direction"] == "long"
            _last_ts  = int(df.index[-1].timestamp()) + _JST2
            _live_marker = [{
                "time":     _last_ts,
                "position": "belowBar" if _is_long else "aboveBar",
                "color":    "#00e676" if _is_long else "#ff1744",
                "shape":    "arrowUp" if _is_long else "arrowDown",
                "text":     f"{'▲BUY' if _is_long else '▼SELL'} {_sig['entry']}",
                "size":     2,
            }]
    except Exception:
        pass

    # ---- 通知（エントリー・エグジット）— トースト＋音 ----
    _notification: dict | None = None
    if notify:
        import time as _time
        _notify_dir_key = f"_notify_prev_dir_{panel_id}"
        _prev_dir = st.session_state.get(_notify_dir_key, "neutral")
        try:
            _cur_dir = _sig["direction"]
        except NameError:
            _cur_dir = "neutral"

        if _prev_dir == "neutral" and _cur_dir in ("long", "short"):
            _arrow = "▲ BUY" if _cur_dir == "long" else "▼ SELL"
            _ep  = f"{_signal_lines['entry']}"        if _signal_lines else ""
            _slp = f"{_signal_lines['sl_pips']:.1f}p" if _signal_lines else ""
            _tpp = f"{_signal_lines['tp_pips']:.1f}p" if _signal_lines else ""
            st.toast(
                f"**{symbol} {timeframe}** {_arrow} シグナル\n"
                f"Entry: {_ep}  SL: {_slp}  TP: {_tpp}",
                icon="📈" if _cur_dir == "long" else "📉",
            )
            _notification = {"type": f"entry_{_cur_dir}", "ts": int(_time.time() * 1000)}
        elif _prev_dir in ("long", "short") and _cur_dir == "neutral":
            _dir_label = "BUY" if _prev_dir == "long" else "SELL"
            st.toast(
                f"**{symbol} {timeframe}** {_dir_label} シグナル消滅",
                icon="⏸",
            )
            _notification = {"type": "exit", "ts": int(_time.time() * 1000)}

        st.session_state[_notify_dir_key] = _cur_dir

    write_panel_json(df, panel_id, ind, events=events + _live_marker, signal_lines=_signal_lines, notification=_notification, cvd_scale=cvd_scale)

    # データ源ラベル
    src = ("🟢 MT5" if source == "mt5"
           else "🟡 サンプル" if source == "sample"
           else "🔴 エラー")

    # ---- ボラティリティ状況 ----
    try:
        import numpy as np
        _h = df["high"].values; _l = df["low"].values; _c = df["close"].values
        tr = np.maximum.reduce([
            _h[1:] - _l[1:],
            np.abs(_h[1:] - _c[:-1]),
            np.abs(_l[1:] - _c[:-1]),
        ])
        atr14  = float(np.convolve(tr, np.ones(14)/14, mode="valid")[-1]) if len(tr) >= 14 else 0.0
        atr50  = float(np.convolve(tr, np.ones(50)/50, mode="valid")[-1]) if len(tr) >= 50 else atr14
        pip_sz = 0.01 if symbol.endswith("JPY") else 0.0001
        atr_p  = round(atr14 / pip_sz, 1)
        ratio  = atr14 / atr50 if atr50 > 0 else 1.0
        vol_icon  = "🔴" if ratio > 1.3 else ("🟡" if ratio < 0.7 else "🟢")
        vol_label = f"{'高ボラ' if ratio > 1.3 else '閑散' if ratio < 0.7 else '通常'} {atr_p:.0f}pips"
    except Exception:
        vol_icon, vol_label = "⚪", "―"

    # ---- ボリューム急増検知 ----
    vol_spike_msg  = ""
    vol_spike_level = 0  # 0=normal 1=増加 2=急増
    try:
        import numpy as _np2
        vols = df["volume"].values
        if len(vols) >= 20:
            avg_vol    = float(_np2.mean(vols[-21:-1]))
            last_vol   = float(vols[-1])
            spike_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
            if spike_ratio >= 3.0:
                vol_spike_msg   = f"⚡ ボリューム急増 {spike_ratio:.1f}x"
                vol_spike_level = 2
            elif spike_ratio >= 2.0:
                vol_spike_msg   = f"📈 ボリューム増加 {spike_ratio:.1f}x"
                vol_spike_level = 1
    except Exception:
        pass

    # ---- 高インパクト指標カウントダウン ----
    countdown_str  = ""
    news_warn_secs = 9999
    news_warn_text = ""
    try:
        from dashboard.calendar_utils import fetch_both_weeks
        sym_up     = symbol.upper().replace("/", "")
        currencies = {sym_up[:3], sym_up[3:6]} if len(sym_up) >= 6 else set()
        now_utc    = datetime.now(timezone.utc)
        upcoming   = sorted(
            [e for e in fetch_both_weeks()
             if e["dt_utc"] > now_utc and e["impact"] == "High"
             and e["country"] in currencies],
            key=lambda e: e["dt_utc"],
        )
        if upcoming:
            nxt  = upcoming[0]
            secs = int((nxt["dt_utc"] - now_utc).total_seconds())
            news_warn_secs = secs
            tstr = f"{secs//60}分{secs%60}秒" if secs < 3600 else f"{secs//3600}時間{(secs%3600)//60}分"
            countdown_str  = f"🔔 {nxt['country']} {nxt['title'][:22]} まで **{tstr}**"
            if secs <= 300:
                news_warn_text = f"⚠️ 重要指標まで {tstr} — {nxt['country']} {nxt['title'][:30]}"
    except Exception:
        pass

    # ---- 現在バーの RSI / ADX / Stoch（バー変化時のみ再計算）----
    _ind_cache_key = f"_ind_cache_{panel_id}_{str(df.index[-1])}"
    if _ind_cache_key not in st.session_state:
        # 古いキャッシュを削除
        for _k in [k for k in st.session_state if k.startswith(f"_ind_cache_{panel_id}_")]:
            del st.session_state[_k]
        _rsi_v = _adx_v = _stoch_v = None
        try:
            import numpy as _np3
            _c2 = df["close"].values.astype(float)
            _hv = df["high"].values.astype(float)
            _lv = df["low"].values.astype(float)
            if len(_c2) >= 10:
                _diff = _np3.diff(_c2[-10:])
                _up   = _np3.where(_diff > 0, _diff, 0.0)
                _dn   = _np3.where(_diff < 0, -_diff, 0.0)
                _au, _ad = float(_np3.mean(_up)), float(_np3.mean(_dn))
                _rsi_v = round(100 - 100 / (1 + _au / (_ad + 1e-12)), 1)
            if len(_hv) >= 15:
                _dmp = _np3.maximum(_np3.diff(_hv[-15:]), 0)
                _dmn = _np3.maximum(-_np3.diff(_lv[-15:]), 0)
                _tr2 = _np3.maximum.reduce([
                    _hv[-14:] - _lv[-14:],
                    _np3.abs(_hv[-14:] - _c2[-15:-1]),
                    _np3.abs(_lv[-14:] - _c2[-15:-1]),
                ])
                _atr2 = float(_np3.mean(_tr2))
                if _atr2 > 0:
                    _dip = float(_np3.mean(_dmp)) / _atr2 * 100
                    _din = float(_np3.mean(_dmn)) / _atr2 * 100
                    _adx_v = round(abs(_dip - _din) / ((_dip + _din) + 1e-12) * 100, 1)
            if len(_hv) >= 8:
                _hi5 = float(_np3.max(_hv[-5:]))
                _lo5 = float(_np3.min(_lv[-5:]))
                _stoch_v = round((_c2[-1] - _lo5) / ((_hi5 - _lo5) + 1e-12) * 100, 1)
        except Exception:
            pass
        st.session_state[_ind_cache_key] = (_rsi_v, _adx_v, _stoch_v)
    rsi_val, adx_val, stoch_val = st.session_state[_ind_cache_key]

    # ---- 現在バー状態バー表示（色コーディング付き）----
    def _colored(val: str, color: str) -> str:
        return f"<span style='color:{color};font-weight:bold'>{val}</span>"

    _mc1, _mc2, _mc3, _mc4 = st.columns(4)

    # ボラティリティ
    _vol_color = "#ef5350" if ratio > 1.3 else ("#ffb300" if ratio < 0.7 else "#26a69a")
    _mc1.markdown(
        f"**ボラ**  \n"
        + _colored(f"{atr_p:.0f}p" if vol_icon != "⚪" else "―", _vol_color)
        + f"  \n<small style='color:#9e9e9e'>{vol_label}</small>",
        unsafe_allow_html=True,
    )

    # RSI
    if rsi_val is not None:
        _rc = "#ef5350" if rsi_val >= 70 else ("#26a69a" if rsi_val <= 30 else "#e0e0e0")
        _rt = "買われ過ぎ ≥70" if rsi_val >= 70 else ("売られ過ぎ ≤30" if rsi_val <= 30 else "中立")
        _mc2.markdown(
            f"**RSI(9)**  \n" + _colored(f"{rsi_val:.0f}", _rc)
            + f"  \n<small style='color:#9e9e9e'>{_rt}</small>",
            unsafe_allow_html=True,
        )
    else:
        _mc2.markdown("**RSI(9)**  \n―")

    # ADX
    if adx_val is not None:
        _ac = "#26a69a" if adx_val >= 25 else ("#ef5350" if adx_val < 15 else "#ffb300")
        _at = "強トレンド ≥25" if adx_val >= 25 else ("レンジ <15" if adx_val < 15 else "中程度")
        _mc3.markdown(
            f"**ADX(14)**  \n" + _colored(f"{adx_val:.0f}", _ac)
            + f"  \n<small style='color:#9e9e9e'>{_at}</small>",
            unsafe_allow_html=True,
        )
    else:
        _mc3.markdown("**ADX(14)**  \n―")

    # Stoch%K
    if stoch_val is not None:
        _sc2 = "#ef5350" if stoch_val >= 80 else ("#26a69a" if stoch_val <= 20 else "#e0e0e0")
        _st2 = "OB（売り検討）≥80" if stoch_val >= 80 else ("OS（買い検討）≤20" if stoch_val <= 20 else "中立")
        _mc4.markdown(
            f"**Stoch%K**  \n" + _colored(f"{stoch_val:.0f}", _sc2)
            + f"  \n<small style='color:#9e9e9e'>{_st2}</small>",
            unsafe_allow_html=True,
        )
    else:
        _mc4.markdown("**Stoch%K**  \n―")

    # ---- 売買マーカー損益サマリー ----
    if _at_stats and _at_stats.get("trade_count", 0) > 0:
        _pips   = _at_stats["total_pips"]
        _cnt    = _at_stats["trade_count"]
        _wr     = _at_stats["win_rate"]
        _pf     = _at_stats["profit_factor"]
        _pf_str = f"{_pf:.2f}" if _pf != float("inf") else "∞"
        _pips_color = "#26a69a" if _pips >= 0 else "#ef5350"
        _pips_sign  = "+" if _pips >= 0 else ""
        st.markdown(
            f"<div style='background:#1e222d;border-radius:6px;padding:6px 10px;margin:4px 0;"
            f"display:flex;gap:16px;align-items:center;font-size:0.82rem'>"
            f"<span style='color:#9e9e9e'>📊 売買マーカー損益 ({symbol} {timeframe})</span>"
            f"<span style='color:{_pips_color};font-weight:bold;font-size:1rem'>"
            f"{_pips_sign}{_pips:.1f} pips</span>"
            f"<span style='color:#9e9e9e'>{_cnt}回</span>"
            f"<span style='color:#e0e0e0'>勝率 {_wr:.0f}%</span>"
            f"<span style='color:#e0e0e0'>PF {_pf_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- ボラティリティ・エントリー適性チェック ----
    if ratio > 1.8:
        st.warning(
            f"⚡ 急激なボラ拡大 ({ratio:.1f}x) — ストップ注意。ロットを絞ること",
            icon="⚠️",
        )

    # ---- アラート表示（重要警告のみ、パネル0のみ表示）----
    if panel_id == 0:
        if news_warn_secs <= 300:
            st.error(news_warn_text, icon="🚨")
        elif news_warn_secs <= 900:
            st.warning(countdown_str, icon="⏰")
        if vol_spike_level == 2:
            st.error(f"⚡ {vol_spike_msg} — ブレイクアウトに注意", icon="⚡")
        elif vol_spike_level == 1:
            st.info(f"{vol_spike_msg} — 勢いを確認", icon="📈")



# ============================================================
# MT5 自動再接続フラグメント（5秒ごと）
# ============================================================

@st.fragment(run_every=5)
def _mt5_autoconnect() -> None:
    """MT5が未接続の場合、自動で再接続を試みる。接続成功時にページを更新する。"""
    from data.mt5_client import is_available, is_connected, try_connect
    if not is_available():
        return
    if is_connected():
        return
    # 接続を試みる
    if try_connect():
        st.cache_data.clear()
        st.rerun()


_mt5_autoconnect()

# 起動時にチューニングが古い場合は自動実行
def _maybe_auto_tune(symbol: str, timeframe: str) -> None:
    from dashboard.auto_tuner import is_cache_fresh
    if not is_cache_fresh(symbol, timeframe):
        with st.spinner(f"🔬 {symbol}/{timeframe} を自動最適化中（初回のみ時間がかかります）..."):
            from dashboard.auto_tuner import run_auto_tune
            run_auto_tune(symbol, timeframe)


_maybe_auto_tune(panel_cfgs[0]["symbol"], panel_cfgs[0]["timeframe"])

for i, cfg in enumerate(panel_cfgs):
    panel_fragment(i, cfg["symbol"], cfg["timeframe"], cfg["n_bars"], cfg["indicators"], cfg.get("show_ind", True), cfg.get("show_markers", True), cfg.get("notify", False), cfg.get("cvd_scale", 1))

# ============================================================
# MTF 整合性パネル（短期トレード特化）
# ============================================================

# 短期トレード向け上位足マッピング（現在足を含む全参照TF）
_MTF_UPPER: dict[str, list[str]] = {
    "1M":  ["1M",  "5M",  "15M", "30M", "1H",  "4H"],
    "5M":  ["5M",  "15M", "30M", "1H",  "4H",  "1D"],
    "15M": ["15M", "30M", "1H",  "4H",  "1D"],
    "30M": ["30M", "1H",  "4H",  "1D"],
    "1H":  ["1H",  "4H",  "1D",  "1W"],
    "4H":  ["4H",  "1D",  "1W"],
    "1D":  ["1D",  "1W"],
    "1W":  ["1W"],
}


@st.fragment(run_every=5)
def _mtf_panel(symbol: str, current_tf: str) -> None:
    upper_tfs = _MTF_UPPER.get(current_tf, [])
    if not upper_tfs:
        return

    import numpy as np
    import pandas as pd
    pip_sz = 0.01 if symbol.endswith("JPY") else 0.0001

    rows      = []
    tf_scores = []   # (bull_count, bear_count) per TF

    for tf in upper_tfs:
        try:
            df_tf, _ = get_ohlcv_dataframe(symbol, tf, count=80)
            if len(df_tf) < 20:
                continue
            close = df_tf["close"]

            # ---- EMA5 / EMA9 クロス（短期トレンド）----
            ema5  = close.ewm(span=5,  adjust=False).mean()
            ema9  = close.ewm(span=9,  adjust=False).mean()
            e5    = float(ema5.iloc[-1])
            e9    = float(ema9.iloc[-1])
            e5_p  = float(ema5.iloc[-4])  # 4本前
            last_c = float(close.iloc[-1])

            ema_bull = last_c > e5 and e5 > e9 and e5 > e5_p
            ema_bear = last_c < e5 and e5 < e9 and e5 < e5_p
            if ema_bull:
                ema_label = "🟢 上昇"
            elif ema_bear:
                ema_label = "🔴 下降"
            else:
                ema_label = "⚪ 中立"

            # ---- MACD ヒストグラム方向 ----
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd  = ema12 - ema26
            sig   = macd.ewm(span=9, adjust=False).mean()
            hist  = macd - sig
            h_now  = float(hist.iloc[-1])
            h_prev = float(hist.iloc[-2])
            macd_bull = h_now > 0 and h_now >= h_prev
            macd_bear = h_now < 0 and h_now <= h_prev
            if macd_bull:
                macd_label = "↑"
            elif macd_bear:
                macd_label = "↓"
            else:
                macd_label = "→"

            # ---- RSI（50基準でバイアス判定）----
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, float("nan"))
            rsi_v = float((100 - 100 / (1 + rs)).iloc[-1])
            rsi_bull = rsi_v > 55
            rsi_bear = rsi_v < 45

            # ---- 直近3本の勢い ----
            last3_close = close.iloc[-3:].values
            last3_open  = df_tf["open"].iloc[-3:].values
            up_bars   = int(sum(c > o for c, o in zip(last3_close, last3_open)))
            down_bars = 3 - up_bars
            mom_bull  = up_bars >= 2
            mom_bear  = down_bars >= 2
            mom_label = f"↑{up_bars}/↓{down_bars}"

            # ---- ATR pips ----
            h = df_tf["high"].values; lv = df_tf["low"].values; cv = df_tf["close"].values
            tr    = np.maximum.reduce([
                h[1:] - lv[1:],
                np.abs(h[1:] - cv[:-1]),
                np.abs(lv[1:] - cv[:-1]),
            ])
            atr_p = round(float(np.mean(tr[-14:])) / pip_sz, 1) if len(tr) >= 14 else 0.0

            # ---- 強弱スコア（4指標で判定）----
            bull_pts = sum([ema_bull, macd_bull, rsi_bull, mom_bull])
            bear_pts = sum([ema_bear, macd_bear, rsi_bear, mom_bear])
            tf_scores.append((bull_pts, bear_pts))

            score_str = f"🟢×{bull_pts}" if bull_pts > bear_pts else (
                        f"🔴×{bear_pts}" if bear_pts > bull_pts else "⚪")

            rows.append({
                "TF":        tf,
                "EMA5/9":    ema_label,
                "MACD":      macd_label,
                f"RSI":      f"{rsi_v:.0f}",
                "勢い(3本)": mom_label,
                "ATR":       f"{atr_p:.1f}p",
                "スコア":    score_str,
            })
        except Exception:
            continue

    if not rows:
        return

    # ---- 総合判定 ----
    total_bull = sum(b for b, _ in tf_scores)
    total_bear = sum(b for _, b in tf_scores)
    n_tfs      = len(tf_scores)
    bull_tfs   = sum(1 for b, br in tf_scores if b > br)
    bear_tfs   = sum(1 for b, br in tf_scores if br > b)
    max_bull   = n_tfs * 4  # 全TF・全指標が強気の場合

    with st.expander(
        f"📊 MTF整合性 [{current_tf}] — {symbol}　"
        f"{'✅ ロング優位' if bull_tfs > bear_tfs else ('✅ ショート優位' if bear_tfs > bull_tfs else '⏸ 待機')}",
        expanded=True,
    ):
        mtf_df = pd.DataFrame(rows)

        def _style_ema(val: str) -> str:
            if "上昇" in val: return "color:#26a69a;font-weight:bold"
            if "下降" in val: return "color:#ef5350;font-weight:bold"
            return "color:#9e9e9e"

        def _style_macd(val: str) -> str:
            if val == "↑": return "color:#26a69a;font-weight:bold"
            if val == "↓": return "color:#ef5350;font-weight:bold"
            return "color:#9e9e9e"

        def _style_rsi(val: str) -> str:
            try:
                v = float(val)
                if v >= 70: return "color:#ef5350;font-weight:bold"
                if v <= 30: return "color:#26a69a;font-weight:bold"
                if v >= 55: return "color:#26a69a"
                if v <= 45: return "color:#ef5350"
            except Exception:
                pass
            return ""

        def _style_score(val: str) -> str:
            if "🟢" in val: return "color:#26a69a;font-weight:bold"
            if "🔴" in val: return "color:#ef5350;font-weight:bold"
            return "color:#9e9e9e"

        styled = (
            mtf_df.style
            .map(_style_ema,   subset=["EMA5/9"])
            .map(_style_macd,  subset=["MACD"])
            .map(_style_rsi,   subset=["RSI"])
            .map(_style_score, subset=["スコア"])
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)

        # ---- 総合判定メッセージ ----
        if bull_tfs == n_tfs:
            st.success(f"✅ 全{n_tfs}TF **ロング優位** — エントリー可（買い方向）")
        elif bear_tfs == n_tfs:
            st.error(f"✅ 全{n_tfs}TF **ショート優位** — エントリー可（売り方向）")
        elif bull_tfs > bear_tfs:
            st.info(f"🟢 ロング優勢 ({bull_tfs}/{n_tfs}TF一致) — 慎重にロング検討")
        elif bear_tfs > bull_tfs:
            st.info(f"🔴 ショート優勢 ({bear_tfs}/{n_tfs}TF一致) — 慎重にショート検討")
        else:
            st.warning(f"⏸ 上位足が不一致 ({bull_tfs}vs{bear_tfs}) — エントリー待機")

        # ---- 短期スキャルプ向けエントリー判定 ----
        # 1M/5M では「1M方向 vs 5M方向」の整合性チェックが重要
        _tf_dir = {}   # {"1M": 1, "5M": -1, ...}  1=bull, -1=bear, 0=neutral
        for tf_, (bp, brp) in zip(upper_tfs, tf_scores):
            _tf_dir[tf_] = 1 if bp > brp else (-1 if brp > bp else 0)

        if current_tf in ("1M", "5M"):
            _micro = _tf_dir.get(current_tf, 0)
            _upper = _tf_dir.get("5M", 0) if current_tf == "1M" else _tf_dir.get("15M", 0)
            _upper_tf_name = "5M" if current_tf == "1M" else "15M"
            if _micro == _upper and _micro != 0:
                _dir_str = "🟢 ロング" if _micro == 1 else "🔴 ショート"
                st.success(f"⚡ スキャルプ推奨: {current_tf}と{_upper_tf_name}が同方向 → **{_dir_str}エントリー有利**")
            elif _micro != 0 and _upper != 0 and _micro != _upper:
                _m_str = "↑" if _micro == 1 else "↓"
                _u_str = "↑" if _upper == 1 else "↓"
                st.warning(
                    f"⚠️ 方向競合: {current_tf}{_m_str} vs {_upper_tf_name}{_u_str} "
                    f"— 逆張りリスク大。エントリー慎重に"
                )


# ============================================================
# エントリースコアリング（5秒ごと自動更新）
# ============================================================

@st.fragment(run_every=5)
def _entry_score_panel(symbol: str, timeframe: str, n_bars: int) -> None:
    from dashboard.chart_utils import JST_OFFSET
    from dashboard.signal_score import calc_entry_score

    try:
        df, _ = get_ohlcv_dataframe(symbol, timeframe, count=n_bars)
        result = calc_entry_score(df)
    except Exception:
        return

    score     = result["score"]
    direction = result["direction"]
    details   = result["details"]
    l_score   = result["long_score"]
    s_score   = result["short_score"]

    dir_label = {"long": "🟢 ロング優位", "short": "🔴 ショート優位", "neutral": "⚪ 中立"}[direction]
    dir_color = {"long": "#26a69a",       "short": "#ef5350",         "neutral": "#9e9e9e"}[direction]
    bar_filled = int(score / 10)
    bar_str    = "█" * bar_filled + "░" * (10 - bar_filled)

    with st.expander(
        f"🎯 エントリースコア — {symbol}/{timeframe}　{dir_label}　{score}点",
        expanded=True,
    ):
        st.markdown(
            f"<span style='color:{dir_color};font-size:1.4em;font-weight:bold'>"
            f"{bar_str}　{score} / 100</span>",
            unsafe_allow_html=True,
        )

        import pandas as pd
        detail_rows = [
            {"項目": k, "状態": v[0], "Long": v[1], "Short": v[2]}
            for k, v in details.items()
        ]
        if detail_rows:
            df_det = pd.DataFrame(detail_rows)

            def _style_long(val):
                return "color:#26a69a;font-weight:bold" if val >= 15 else ""

            def _style_short(val):
                return "color:#ef5350;font-weight:bold" if val >= 15 else ""

            styled = (
                df_det.style
                .map(_style_long,  subset=["Long"])
                .map(_style_short, subset=["Short"])
            )
            st.dataframe(styled, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Long合計",  l_score, delta=None)
        c2.metric("Short合計", s_score, delta=None)


# ============================================================
# 通貨強弱メーター（60秒ごと自動更新）
# ============================================================

@st.fragment(run_every=60)
def _currency_strength_panel(timeframe: str) -> None:
    from dashboard.signal_score import calc_currency_strength

    try:
        strength = calc_currency_strength(timeframe=timeframe)
    except Exception:
        return

    if not strength:
        return

    import pandas as pd
    currencies = list(strength.keys())
    values     = list(strength.values())
    max_abs    = max(abs(v) for v in values) if values else 1.0

    with st.expander(f"💱 通貨強弱メーター ({timeframe}　{len(currencies)}通貨)", expanded=True):
        for cur, val in zip(currencies, values):
            ratio    = val / max_abs if max_abs > 0 else 0.0
            bar_len  = int(abs(ratio) * 20)
            if val >= 0:
                bar    = "█" * bar_len
                color  = "#26a69a"
                prefix = "+"
            else:
                bar    = "█" * bar_len
                color  = "#ef5350"
                prefix = ""
            st.markdown(
                f"<span style='font-family:monospace'>"
                f"<b>{cur}</b>　"
                f"<span style='color:{color}'>{bar:<20}</span>　"
                f"<span style='color:{color}'>{prefix}{val:+.2f}</span>"
                f"</span>",
                unsafe_allow_html=True,
            )

        # 最強 vs 最弱の推奨ペア
        if len(currencies) >= 2:
            strongest = currencies[0]
            weakest   = currencies[-1]
            st.markdown("---")
            st.markdown(
                f"**推奨ペア**: "
                f"<span style='color:#26a69a'>{strongest}</span> vs "
                f"<span style='color:#ef5350'>{weakest}</span>　→　"
                f"`{strongest}{weakest}` または `{weakest}{strongest}` でトレンドフォロー",
                unsafe_allow_html=True,
            )


# ============================================================
# 直近の重要ニュース（5分ごと自動更新）
# ============================================================

@st.fragment(run_every=300)
def _chart_news() -> None:
    from datetime import datetime, timezone
    from dashboard.news_utils import fetch_and_analyze_news, time_ago

    try:
        items = fetch_and_analyze_news()
    except Exception:
        return

    now = datetime.now(timezone.utc)
    important = [
        it for it in items
        if it["analysis"]["impact"] >= 3
        and it.get("published")
        and (now - it["published"]).total_seconds() < 21600  # 6時間以内
    ]

    if not important:
        st.caption("過去6時間に影響度3以上のニュースはありません。")
        return

    _arrow = {"up": "▲", "down": "▼", "neutral": "━"}
    _color = {"up": "#26a69a", "down": "#ef5350", "neutral": "#9e9e9e"}
    _bar   = {3: "●●●○○", 4: "●●●●○", 5: "●●●●●"}

    st.caption(f"影響度3以上 / 過去6時間 — {len(important)}件")
    for it in important[:8]:
        a         = it["analysis"]
        direction = a.get("direction", "neutral")
        arrow     = _arrow[direction]
        color     = _color[direction]
        bar       = _bar.get(a.get("impact", 3), "●●●○○")
        title_ja  = a.get("title_ja") or it["title"]
        st.markdown(
            f"<span style='color:{color};font-weight:bold'>{arrow} {title_ja}</span>"
            f"<span style='font-size:0.8em;color:#9e9e9e'>"
            f"　{bar}　{it['source']}・{time_ago(it['published'])}</span>",
            unsafe_allow_html=True,
        )


@st.fragment(run_every=5)
def _live_signal_panel(symbol: str, timeframe: str, n_bars: int) -> None:
    """ライブシグナル（エントリー価格・SL・TP）を表示する"""
    from dashboard.signal_engine import get_live_signal
    from dashboard.auto_tuner import is_cache_fresh, run_auto_tune, get_tune_result

    df, _ = get_ohlcv_dataframe(symbol, timeframe, count=n_bars)
    sig = get_live_signal(symbol, timeframe, df)

    direction  = sig["direction"]
    confidence = sig["confidence"]
    strategy   = sig["strategy"]
    oos_pf     = sig["oos_pf"]
    htf_trend  = sig["htf_trend"]
    htf_aligned = sig["htf_aligned"]
    signal_age = sig["signal_age"]

    # ----- 自動チューニングステータス -----
    tune = get_tune_result(symbol, timeframe)
    if tune:
        import time as _t
        from dashboard.auto_tuner import _cache_path
        p = _cache_path(symbol, timeframe)
        age_h = (_t.time() - p.stat().st_mtime) / 3600 if p.exists() else 0
        tune_status = f"✅ 最終チューニング: {age_h:.0f}時間前  |  戦略: {tune['strategy']}  |  OOS PF: {tune['oos_pf']:.2f}"
    else:
        tune_status = "⚠️ 未チューニング"

    # ----- 方向ラベル -----
    if direction == "long":
        dir_label = "🟢 BUY"
        dir_color = "#26a69a"
    elif direction == "short":
        dir_label = "🔴 SELL"
        dir_color = "#ef5350"
    else:
        dir_label = "⏸ 待機"
        dir_color = "#9e9e9e"

    conf_icon = {"high": "🔥", "medium": "⚡", "low": "❄️"}[confidence]
    htf_icon  = "✅" if htf_aligned else "⚠️"

    with st.expander(
        f"🎯 ライブシグナル — {symbol}/{timeframe}　{dir_label}　{conf_icon}",
        expanded=True,
    ):
        st.caption(tune_status)
        st.markdown(
            f"<span style='color:{dir_color};font-size:2em;font-weight:bold'>{dir_label}</span>",
            unsafe_allow_html=True,
        )

        if direction != "neutral":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("エントリー価格", f"{sig['entry']}")
            c2.metric("SL", f"{sig['sl']}", delta=f"-{sig['sl_pips']:.1f}pips", delta_color="inverse")
            c3.metric("TP", f"{sig['tp']}", delta=f"+{sig['tp_pips']:.1f}pips")
            c4.metric("推奨ロット", f"{sig.get('recommended_lot', 0):.2f}万通貨",
                      help="口座50万円・1%リスク基準。実際の口座に合わせてください。")

            _session_label = {"all": "全時間", "london": "ロンドン", "overlap": "ロンドン-NY重複"}.get(
                sig.get("session", "all"), sig.get("session", "all"))
            st.markdown(
                f"**RR比**: 1:{sig['rr']:.1f}　|　"
                f"**上位TF**: {htf_icon} {htf_trend}方向　|　"
                f"**シグナル経過**: {signal_age}本前　|　"
                f"**セッション**: {_session_label}　|　"
                f"**戦略**: {strategy}"
            )

            if confidence == "high":
                st.success(f"🔥 高信頼度シグナル — OOS PF {oos_pf:.2f}・上位TF一致")
            elif confidence == "medium":
                st.info(f"⚡ 中信頼度シグナル — OOS PF {oos_pf:.2f}")
            else:
                if not htf_aligned:
                    st.warning(f"⚠️ 上位TF逆行 ({htf_trend}) — エントリー慎重に")
                else:
                    st.warning(f"❄️ 低信頼度 — OOS PF {oos_pf:.2f}")
        else:
            st.info("現在シグナルなし — 次のシグナルを待機中")

        # 再チューニングボタン
        if st.button("🔄 今すぐ再チューニング", key=f"retune_{symbol}_{timeframe}"):
            with st.spinner("最適化中..."):
                from dashboard.auto_tuner import run_auto_tune
                run_auto_tune(symbol, timeframe, df=df)
            st.rerun()


# ============================================================
# 分析パネル（タブ切り替えで縦スクロール削減）
# ============================================================

_tab_signal, _tab_mtf, _tab_score, _tab_cs, _tab_news = st.tabs([
    "🎯 ライブシグナル",
    "📊 MTF整合性",
    "🎯 エントリースコア",
    "💱 通貨強弱",
    "📰 ニュース",
])

with _tab_signal:
    if panel_count == 1:
        _live_signal_panel(panel_cfgs[0]["symbol"], panel_cfgs[0]["timeframe"], panel_cfgs[0]["n_bars"])
    else:
        _lsc = st.columns(min(panel_count, 2))
        for _i in range(panel_count):
            with _lsc[_i % 2]:
                _live_signal_panel(panel_cfgs[_i]["symbol"], panel_cfgs[_i]["timeframe"], panel_cfgs[_i]["n_bars"])

with _tab_mtf:
    if panel_count == 1:
        _mtf_panel(panel_cfgs[0]["symbol"], panel_cfgs[0]["timeframe"])
    else:
        _mc = st.columns(min(panel_count, 2))
        for _i in range(panel_count):
            with _mc[_i % 2]:
                _mtf_panel(panel_cfgs[_i]["symbol"], panel_cfgs[_i]["timeframe"])

with _tab_score:
    if panel_count == 1:
        _entry_score_panel(panel_cfgs[0]["symbol"], panel_cfgs[0]["timeframe"], panel_cfgs[0]["n_bars"])
    else:
        _sc = st.columns(min(panel_count, 2))
        for _i in range(panel_count):
            with _sc[_i % 2]:
                _entry_score_panel(panel_cfgs[_i]["symbol"], panel_cfgs[_i]["timeframe"], panel_cfgs[_i]["n_bars"])

with _tab_cs:
    _currency_strength_panel(panel_cfgs[0]["timeframe"])

with _tab_news:
    _chart_news()
