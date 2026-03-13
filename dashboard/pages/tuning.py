"""
dashboard/pages/tuning.py

シグナルチューニングページ。

【自動最適化モード（推奨）】
  2段階最適化を実施する。
  Phase 1（探索）: 広いパラメーター空間をランダムサーチ
  Phase 2（絞込）: Phase 1 の上位候補近傍を集中探索（局所最適化）

  この2段階構成により、同じ試行回数でも純粋なランダムサーチより
  大幅に良いパラメーターを発見できる。

【全時間足一括最適化】
  全8時間足（1M〜1W）を順番に自動最適化し一括保存する。

【手動モード】
  スライダーでパラメーターを手動調整しバックテスト統計を確認できる。

最良パラメーターは .streamlit/signal_params.json に保存し、
チャート画面でも自動適用される。
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import SUPPORTED_TIMEFRAMES, SYMBOL_GROUPS
from config.signal_defaults import (
    get_entry_params, get_ai_params, EVAL_BARS_DEFAULT,
    ENTRY_DEFAULTS, AI_DEFAULTS,
)
from dashboard.sample_data import get_ohlcv_dataframe
from dashboard.indicators import calc_entry_signals, calc_ai_signal

# パラメーター保存先
_PARAMS_FILE = (
    Path(__file__).resolve().parent.parent.parent / ".streamlit" / "signal_params.json"
)

# ============================================================
# パラメーター永続化
# ============================================================

def _load_params() -> dict:
    if _PARAMS_FILE.exists():
        try:
            return json.loads(_PARAMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"entry": {}, "ai": {}}


def _save_params(timeframe: str, entry_params: dict, ai_params: dict) -> None:
    saved = _load_params()
    saved.setdefault("entry", {})[timeframe] = entry_params
    saved.setdefault("ai",    {})[timeframe] = ai_params
    _PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PARAMS_FILE.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# バックテスト評価
# ============================================================

def _evaluate_signals(
    df: pd.DataFrame,
    markers: list[dict],
    eval_bars: int,
    jst_offset: int,
    pip_size: float,
) -> dict:
    empty = {
        "count": 0, "win_rate": 0.0,
        "avg_gain_pips": 0.0, "avg_loss_pips": 0.0,
        "profit_factor": 0.0, "total_pips": 0.0,
        "wins": 0, "losses": 0,
    }
    if not markers or len(df) < eval_bars + 1:
        return empty

    ts_to_idx = {int(ts.timestamp()) + jst_offset: i for i, ts in enumerate(df.index)}
    close = df["close"].values
    wins, losses = 0, 0
    total_gain, total_loss = 0.0, 0.0
    results = []

    for m in markers:
        idx = ts_to_idx.get(m["time"])
        if idx is None or idx + eval_bars >= len(df):
            continue
        pnl = ((close[idx + eval_bars] - close[idx]) / pip_size
               if m["shape"] == "arrowUp"
               else (close[idx] - close[idx + eval_bars]) / pip_size)
        results.append(pnl)
        if pnl > 0:
            wins += 1;       total_gain += pnl
        else:
            losses += 1;     total_loss += abs(pnl)

    count = len(results)
    if count == 0:
        return empty

    pf = (total_gain / total_loss) if total_loss > 0 else float("inf")
    return {
        "count":         count,
        "win_rate":      wins / count,
        "avg_gain_pips": total_gain / wins   if wins   > 0 else 0.0,
        "avg_loss_pips": total_loss / losses if losses > 0 else 0.0,
        "profit_factor": pf,
        "total_pips":    sum(results),
        "wins":          wins,
        "losses":        losses,
    }


def _opt_score(stats: dict) -> float:
    """
    最適化スコア（大きいほど良い）。

    score = PF × 勝率 × log(件数+1) × 勝ち平均/負け平均比

    ・PF は最大5でキャップ
    ・5件未満は除外
    ・平均利益/平均損失比（期待値）を追加し
      「少ない勝ちで大負けするパラメーター」をさらに排除
    """
    count = stats["count"]
    if count < 5:
        return -1.0
    pf   = min(stats["profit_factor"], 5.0)
    wr   = stats["win_rate"]
    rr   = (stats["avg_gain_pips"] / stats["avg_loss_pips"]
            if stats["avg_loss_pips"] > 0 else 1.0)
    rr   = min(rr, 4.0)
    return pf * wr * math.log(count + 1) * max(rr, 0.5)


# ============================================================
# パラメーターサンプリング（探索フェーズ）
# ============================================================

def _sample_entry(rng: np.random.Generator, default_eval: int) -> dict:
    stoch_strong = int(rng.choice([8, 10, 12, 15, 18, 20, 22, 25, 28]))
    stoch_normal = int(min(stoch_strong + int(rng.choice([5, 8, 10, 12, 15, 18, 20])), 48))
    cd_max       = max(default_eval * 4, 12)
    return {
        "adx_min":      int(rng.choice([5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30])),
        "score_thresh": int(rng.choice([2, 3, 4, 5, 6, 7])),
        "stoch_strong": stoch_strong,
        "stoch_normal": stoch_normal,
        "body_mult":    float(rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35])),
        "cooldown":     int(rng.integers(1, cd_max + 1)),
    }


def _sample_ai(rng: np.random.Generator, default_eval: int) -> dict:
    fwd_max = max(default_eval * 4, 20)
    cd_max  = max(default_eval * 4, 12)
    return {
        "forward":  int(rng.integers(max(1, default_eval // 2), fwd_max + 1)),
        "k":        int(rng.choice([8, 10, 12, 15, 18, 20, 22, 25])),
        "min_prob": float(rng.choice([0.58, 0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82])),
        "cooldown": int(rng.integers(1, cd_max + 1)),
        "adx_min":  int(rng.choice([5, 8, 10, 12, 15, 18, 20, 22, 25])),
        "atr_mult": float(rng.choice([0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])),
    }


# ============================================================
# パラメーター近傍生成（絞込フェーズ）
# ============================================================

def _perturb_entry(rng: np.random.Generator, p: dict, default_eval: int) -> dict:
    """既存パラメーターを小さくランダム変動させた近傍を返す。"""
    cd_max       = max(default_eval * 4, 12)
    stoch_strong = int(np.clip(p["stoch_strong"] + rng.integers(-5, 6), 5, 35))
    stoch_normal = int(np.clip(p["stoch_normal"] + rng.integers(-5, 6), stoch_strong + 3, 48))
    return {
        "adx_min":      int(np.clip(p["adx_min"]      + rng.integers(-4, 5), 3, 35)),
        "score_thresh": int(np.clip(p["score_thresh"]  + rng.integers(-1, 2), 2, 8)),
        "stoch_strong": stoch_strong,
        "stoch_normal": stoch_normal,
        "body_mult":    float(np.clip(round(p["body_mult"] + rng.choice([-0.05, 0, 0.05]), 2), 0.03, 0.40)),
        "cooldown":     int(np.clip(p["cooldown"]      + rng.integers(-2, 3), 1, cd_max)),
    }


def _perturb_ai(rng: np.random.Generator, p: dict, default_eval: int) -> dict:
    fwd_max = max(default_eval * 4, 20)
    cd_max  = max(default_eval * 4, 12)
    return {
        "forward":  int(np.clip(p["forward"]  + rng.integers(-2, 3), 1, fwd_max)),
        "k":        int(np.clip(p["k"]        + rng.integers(-3, 4), 5, 30)),
        "min_prob": float(np.clip(round(p["min_prob"] + rng.choice([-0.05, 0, 0.05]), 2), 0.55, 0.85)),
        "cooldown": int(np.clip(p["cooldown"] + rng.integers(-2, 3), 1, cd_max)),
        "adx_min":  int(np.clip(p["adx_min"] + rng.integers(-4, 5), 3, 35)),
        "atr_mult": float(np.clip(round(p["atr_mult"] + rng.choice([-0.2, 0, 0.2]), 1), 0.3, 3.0)),
    }


# ============================================================
# 2段階最適化エンジン
# ============================================================

def _two_phase_optimize(
    calc_fn,
    sample_fn,
    perturb_fn,
    df: pd.DataFrame,
    eval_bars: int,
    jst_offset: int,
    pip_size: float,
    default_eval: int,
    n_iter: int,
    top_k: int = 5,
    seed: int = 42,
    progress_cb=None,
) -> tuple[dict, dict, list[dict]]:
    """
    2段階最適化エンジン（エントリー・AIの両方に対応）。

    Phase 1（探索）: n_iter × 40% 回 ランダムサーチで広域探索
    Phase 2（絞込）: 上位 top_k 候補の近傍を n_iter × 60% 回 集中探索

    Returns:
        (best_params, best_stats, top_k_history)
    """
    rng     = np.random.default_rng(seed)
    phase1  = max(10, int(n_iter * 0.40))
    phase2  = n_iter - phase1
    total   = phase1 + phase2

    # --- Phase 1: 広域ランダムサーチ ---
    candidates: list[tuple[float, dict, dict]] = []  # (score, params, stats)

    for i in range(phase1):
        params = sample_fn(rng, default_eval)
        try:
            markers = calc_fn(df, jst_offset, params=params)
            stats   = _evaluate_signals(df, markers, eval_bars, jst_offset, pip_size)
            score   = _opt_score(stats)
        except Exception:
            score, stats = -1.0, {}

        candidates.append((score, params, stats))
        if progress_cb:
            progress_cb(i + 1, total, candidates)

    # 上位 top_k 候補を選出
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [c for c in candidates[:top_k] if c[0] > -1.0]
    if not top_candidates:
        top_candidates = candidates[:1]

    # --- Phase 2: 上位候補の近傍を集中探索 ---
    per_cand = max(1, phase2 // max(len(top_candidates), 1))
    done     = phase1

    for rank, (_, base_params, _) in enumerate(top_candidates):
        iters_this = per_cand if rank < len(top_candidates) - 1 else (phase2 - per_cand * rank)
        for j in range(max(1, iters_this)):
            params = perturb_fn(rng, base_params, default_eval)
            try:
                markers = calc_fn(df, jst_offset, params=params)
                stats   = _evaluate_signals(df, markers, eval_bars, jst_offset, pip_size)
                score   = _opt_score(stats)
            except Exception:
                score, stats = -1.0, {}

            candidates.append((score, params, stats))
            done += 1
            if progress_cb:
                progress_cb(done, total, candidates)

    # 全試行から最良を返す
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_params, best_stats = candidates[0]
    top_history = [
        {"score": s, "params": p, "stats": st}
        for s, p, st in candidates[:top_k]
        if s > -1.0
    ]
    return best_params, best_stats or {}, top_history


def _optimize_entry(df, eval_bars, jst_offset, pip_size, default_eval,
                    n_iter=300, seed=42, progress_cb=None):
    return _two_phase_optimize(
        calc_entry_signals, _sample_entry, _perturb_entry,
        df, eval_bars, jst_offset, pip_size, default_eval,
        n_iter=n_iter, seed=seed, progress_cb=progress_cb,
    )


def _optimize_ai(df, eval_bars, jst_offset, pip_size, default_eval,
                 n_iter=80, seed=42, progress_cb=None):
    return _two_phase_optimize(
        calc_ai_signal, _sample_ai, _perturb_ai,
        df, eval_bars, jst_offset, pip_size, default_eval,
        n_iter=n_iter, seed=seed, progress_cb=progress_cb,
    )


# ============================================================
# 表示ヘルパー
# ============================================================

def _display_stats(stats: dict) -> None:
    if not stats or stats.get("count", 0) == 0:
        st.warning("シグナルが0件です。バー数を増やすかパラメーターを調整してください。")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("シグナル数", stats["count"])
    wr = stats["win_rate"] * 100
    c2.metric("勝率", f"{wr:.1f}%",
              delta=f"+{wr-50:.1f}%" if wr >= 50 else f"{wr-50:.1f}%")
    pf     = stats["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    c3.metric("プロフィットファクター", pf_str)
    c4.metric("累積損益 (pips)", f"{stats['total_pips']:+.1f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("勝ち", f"{stats['wins']}回 / 平均 +{stats['avg_gain_pips']:.1f}p")
    c6.metric("負け", f"{stats['losses']}回 / 平均 -{stats['avg_loss_pips']:.1f}p")
    if pf >= 1.5 and wr >= 55:
        c7.success("良好 ✓")
    elif pf >= 1.2 and wr >= 50:
        c7.info("合格 △")
    else:
        c7.error("要改善 ✗")


def _display_params_table(params: dict, defaults: dict) -> None:
    rows = [
        {"パラメーター": k, "最適値": v, "デフォルト": defaults.get(k, "―")}
        for k, v in params.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _show_pnl_chart(df, markers, eval_bars, jst_offset, pip_size) -> None:
    ts_to_idx  = {int(ts.timestamp()) + jst_offset: i for i, ts in enumerate(df.index)}
    close      = df["close"].values
    pnl_series = []
    cumulative = 0.0

    for m in sorted(markers, key=lambda x: x["time"]):
        idx = ts_to_idx.get(m["time"])
        if idx is None or idx + eval_bars >= len(df):
            continue
        pnl = ((close[idx + eval_bars] - close[idx]) / pip_size
               if m["shape"] == "arrowUp"
               else (close[idx] - close[idx + eval_bars]) / pip_size)
        cumulative += pnl
        pnl_series.append({"bar": idx, "pnl": round(pnl, 2),
                            "cumulative": round(cumulative, 2),
                            "direction": "Long" if m["shape"] == "arrowUp" else "Short"})

    if not pnl_series:
        return
    pnl_df = pd.DataFrame(pnl_series)
    st.line_chart(pnl_df.set_index("bar")["cumulative"], use_container_width=True)
    with st.expander("個別トレード詳細"):
        st.dataframe(
            pnl_df[["bar", "direction", "pnl", "cumulative"]].rename(columns={
                "bar": "バー", "direction": "方向",
                "pnl": "損益(pips)", "cumulative": "累積(pips)"}),
            use_container_width=True,
        )


def _make_progress_cb(prog_bar, status_ph, phase_label: str):
    """プログレスバー更新コールバックを生成する。"""
    def _cb(done: int, total: int, candidates: list):
        pct = done / total
        best = max(candidates, key=lambda x: x[0]) if candidates else (0, {}, {})
        bs, bp, bst = best
        prog_bar.progress(pct, text=f"{phase_label} {done}/{total} ({int(pct*100)}%)")
        if bst and bst.get("count", 0) > 0:
            status_ph.caption(
                f"現在の最良 — PF: {min(bst['profit_factor'], 5.0):.2f}  "
                f"勝率: {bst['win_rate']*100:.1f}%  "
                f"シグナル: {bst['count']}件  "
                f"累積: {bst['total_pips']:+.1f}pips"
            )
    return _cb


# ============================================================
# メインページ
# ============================================================

def main() -> None:
    st.title("パラメーター自動最適化")
    st.caption(
        "2段階最適化（広域探索 → 近傍集中探索）でパラメーターを自動で求めます。  \n"
        "最良のパラメーターは自動保存され、チャート画面に即時反映されます。"
    )

    saved = _load_params()

    # ---- サイドバー ----
    with st.sidebar:
        st.header("データ設定")

        all_symbols = [s for group in SYMBOL_GROUPS.values() for s in group]
        symbol    = st.selectbox("通貨ペア", all_symbols, index=0)
        tf_idx    = SUPPORTED_TIMEFRAMES.index("1M") if "1M" in SUPPORTED_TIMEFRAMES else 0
        timeframe = st.selectbox("時間足", SUPPORTED_TIMEFRAMES, index=tf_idx)

        default_bars = {"1M": 1500, "5M": 800, "15M": 500}.get(timeframe, 300)
        n_bars = st.slider("バー数", 200, 3000, default_bars, 100)

        default_eval = EVAL_BARS_DEFAULT.get(timeframe, 5)
        eval_max     = max(default_eval * 5, 30)
        eval_bars    = st.slider(
            "評価バー数（利確目標）", 1, eval_max, default_eval,
            help=f"30分利確 = {default_eval}本 ({timeframe}足)",
        )

        st.divider()
        st.info(f"**{timeframe} 足の30分利確 = {default_eval} 本**\n\n評価: エントリーから N本後の終値")

        st.divider()
        st.subheader("最適化設定")
        st.caption("試行回数が多いほど精度↑ / 時間↑")
        entry_n_iter = st.slider("エントリー：試行回数", 50, 1000, 300, 50,
                                 help="O(N) なので多くしても高速")
        ai_n_iter    = st.slider("AI：試行回数", 20, 200, 80, 10,
                                 help="k-NN はやや重いため 80〜120 推奨")
        st.caption(
            f"エントリー: Phase1={int(entry_n_iter*0.4)}回 → Phase2={entry_n_iter - int(entry_n_iter*0.4)}回  \n"
            f"AI: Phase1={int(ai_n_iter*0.4)}回 → Phase2={ai_n_iter - int(ai_n_iter*0.4)}回"
        )

    pip_size   = 0.01 if symbol.endswith("JPY") else 0.0001
    jst_offset = 9 * 3600

    # データ取得
    with st.spinner("データ取得中..."):
        try:
            df, data_source = get_ohlcv_dataframe(symbol, timeframe, n_bars)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            return

    if df is None or len(df) < 50:
        st.error("データが不足しています。")
        return

    _src_icon = "🟢 MT5" if data_source == "mt5" else "🟡 サンプル"
    st.success(
        f"{_src_icon} | {symbol} {timeframe} | {len(df)} 本 | "
        f"{df.index[0].strftime('%Y-%m-%d')} 〜 {df.index[-1].strftime('%Y-%m-%d')}"
    )

    ep_saved = saved.get("entry", {}).get(timeframe)
    ap_saved = saved.get("ai",    {}).get(timeframe)
    ep = get_entry_params(timeframe, {timeframe: ep_saved} if ep_saved else None)
    ap = get_ai_params(timeframe,    {timeframe: ap_saved} if ap_saved else None)

    tab_auto, tab_all_tf, tab_entry, tab_ai = st.tabs([
        "🤖 自動最適化",
        "🌐 全時間足一括最適化",
        "エントリーシグナル（手動）",
        "AI シグナル（手動）",
    ])

    # ============================================================
    # 自動最適化タブ（現在の時間足）
    # ============================================================
    with tab_auto:
        st.subheader(f"自動最適化 — {symbol} / {timeframe}")
        st.markdown(
            "**Phase 1（探索）**: 広いパラメーター空間をランダムサーチ  \n"
            "**Phase 2（絞込）**: 上位候補の近傍を集中的に探索  \n"
            "→ 純粋なランダムより少ない試行で高品質なパラメーターを発見"
        )

        col_e, col_a = st.columns(2)

        # ---- エントリー自動最適化 ----
        with col_e:
            st.markdown("#### エントリーシグナル")
            if st.button(f"🔍 最適化 ({entry_n_iter}回)",
                         key="btn_auto_entry", use_container_width=True, type="primary"):
                prog  = st.progress(0, text="Phase1 探索中...")
                stat  = st.empty()
                t0    = time.time()
                best_ep, best_es, top_ep = _optimize_entry(
                    df, eval_bars, jst_offset, pip_size, default_eval,
                    n_iter=entry_n_iter,
                    progress_cb=_make_progress_cb(prog, stat, "エントリー"),
                )
                elapsed = time.time() - t0
                prog.progress(1.0, text=f"完了 ✓ ({elapsed:.1f}秒)")
                if best_es and best_es.get("count", 0) > 0:
                    _save_params(timeframe, best_ep, ap)
                    ep = best_ep
                    st.success(f"✅ 最良パラメーターを保存しました（{datetime.now().strftime('%H:%M:%S')}）")
                    st.session_state[f"_opt_entry_{timeframe}"] = (best_ep, best_es, top_ep)
                else:
                    st.warning("有効なパラメーターが見つかりませんでした。バー数を増やしてください。")

            if f"_opt_entry_{timeframe}" in st.session_state:
                best_ep, best_es, top_ep = st.session_state[f"_opt_entry_{timeframe}"]
                _display_stats(best_es)
                with st.expander("最適パラメーター"):
                    _display_params_table(best_ep, ENTRY_DEFAULTS.get(timeframe, {}))
                if top_ep:
                    with st.expander(f"上位 {len(top_ep)} 件の候補"):
                        rows = []
                        for rank, c in enumerate(top_ep):
                            s = c["stats"]
                            rows.append({
                                "順位": rank + 1,
                                "スコア": round(c["score"], 3),
                                "PF": round(min(s.get("profit_factor", 0), 5), 2),
                                "勝率%": round(s.get("win_rate", 0) * 100, 1),
                                "件数": s.get("count", 0),
                                "累積pips": round(s.get("total_pips", 0), 1),
                            })
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                with st.expander("損益推移グラフ"):
                    _markers_ep = calc_entry_signals(df, jst_offset, params=best_ep)
                    _show_pnl_chart(df, _markers_ep, eval_bars, jst_offset, pip_size)

        # ---- AI自動最適化 ----
        with col_a:
            st.markdown("#### AI シグナル (k-NN)")
            st.caption(f"k-NN は計算が重いため {ai_n_iter} 回を推奨")
            if st.button(f"🔍 最適化 ({ai_n_iter}回)",
                         key="btn_auto_ai", use_container_width=True, type="primary"):
                prog2 = st.progress(0, text="Phase1 探索中...")
                stat2 = st.empty()
                t0    = time.time()
                best_ap, best_as, top_ap = _optimize_ai(
                    df, eval_bars, jst_offset, pip_size, default_eval,
                    n_iter=ai_n_iter,
                    progress_cb=_make_progress_cb(prog2, stat2, "AI"),
                )
                elapsed = time.time() - t0
                prog2.progress(1.0, text=f"完了 ✓ ({elapsed:.1f}秒)")
                if best_as and best_as.get("count", 0) > 0:
                    _save_params(timeframe, ep, best_ap)
                    ap = best_ap
                    st.success(f"✅ 最良パラメーターを保存しました（{datetime.now().strftime('%H:%M:%S')}）")
                    st.session_state[f"_opt_ai_{timeframe}"] = (best_ap, best_as, top_ap)
                else:
                    st.warning("有効なパラメーターが見つかりませんでした。バー数を増やしてください。")

            if f"_opt_ai_{timeframe}" in st.session_state:
                best_ap, best_as, top_ap = st.session_state[f"_opt_ai_{timeframe}"]
                _display_stats(best_as)
                with st.expander("最適パラメーター"):
                    _display_params_table(best_ap, AI_DEFAULTS.get(timeframe, {}))
                if top_ap:
                    with st.expander(f"上位 {len(top_ap)} 件の候補"):
                        rows = []
                        for rank, c in enumerate(top_ap):
                            s = c["stats"]
                            rows.append({
                                "順位": rank + 1,
                                "スコア": round(c["score"], 3),
                                "PF": round(min(s.get("profit_factor", 0), 5), 2),
                                "勝率%": round(s.get("win_rate", 0) * 100, 1),
                                "件数": s.get("count", 0),
                                "累積pips": round(s.get("total_pips", 0), 1),
                            })
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                with st.expander("損益推移グラフ"):
                    _markers_ap = calc_ai_signal(df, jst_offset, params=best_ap)
                    _show_pnl_chart(df, _markers_ap, eval_bars, jst_offset, pip_size)

        st.divider()
        if st.button(f"⚡ エントリー + AI 両方まとめて最適化 ({entry_n_iter + ai_n_iter}回)",
                     key="btn_auto_both", use_container_width=True, type="primary"):
            prog_e = st.progress(0, text="エントリー最適化中...")
            stat_e = st.empty()
            t0 = time.time()
            best_ep2, best_es2, top_ep2 = _optimize_entry(
                df, eval_bars, jst_offset, pip_size, default_eval,
                n_iter=entry_n_iter,
                progress_cb=_make_progress_cb(prog_e, stat_e, "エントリー"),
            )
            prog_e.progress(1.0, text=f"エントリー完了 ({time.time()-t0:.1f}秒)")

            prog_a = st.progress(0, text="AI最適化中...")
            stat_a = st.empty()
            t1 = time.time()
            best_ap2, best_as2, top_ap2 = _optimize_ai(
                df, eval_bars, jst_offset, pip_size, default_eval,
                n_iter=ai_n_iter,
                progress_cb=_make_progress_cb(prog_a, stat_a, "AI"),
            )
            prog_a.progress(1.0, text=f"AI完了 ({time.time()-t1:.1f}秒)")

            _save_params(timeframe, best_ep2, best_ap2)
            st.session_state[f"_opt_entry_{timeframe}"] = (best_ep2, best_es2, top_ep2)
            st.session_state[f"_opt_ai_{timeframe}"]    = (best_ap2, best_as2, top_ap2)
            st.success(f"✅ {timeframe} のエントリー + AI パラメーターを保存しました（{datetime.now().strftime('%H:%M:%S')}）。チャートを再読み込みすると反映されます。")
            st.rerun()

    # ============================================================
    # 全時間足一括最適化タブ
    # ============================================================
    with tab_all_tf:
        st.subheader("全時間足 一括自動最適化")
        st.markdown(
            "全8時間足（1M〜1W）を順番に自動最適化し、まとめて保存します。  \n"
            "⚠️ AI最適化は k-NN の計算が重いため、完了までに数分〜十数分かかります。"
        )

        _tf_targets = st.multiselect(
            "最適化する時間足", SUPPORTED_TIMEFRAMES,
            default=["1M", "5M", "15M"],
            help="使用する時間足のみ選択することを推奨（時間短縮）",
        )
        _do_entry_all = st.checkbox("エントリーシグナルを最適化", value=True)
        _do_ai_all    = st.checkbox("AIシグナルを最適化", value=True)

        # 試行回数は現在のサイドバー設定を使用
        st.caption(
            f"試行回数: エントリー {entry_n_iter}回 × {len(_tf_targets)}TF"
            + (f" + AI {ai_n_iter}回 × {len(_tf_targets)}TF" if _do_ai_all else "")
        )

        if st.button("🌐 全時間足まとめて最適化スタート",
                     key="btn_all_tf", use_container_width=True, type="primary"):
            if not _tf_targets:
                st.error("時間足を1つ以上選択してください。")
            else:
                all_tf_results = {}
                overall_prog = st.progress(0, text="全体進捗...")
                tf_status    = st.empty()
                total_tf     = len(_tf_targets)

                for tf_idx2, tf in enumerate(_tf_targets):
                    tf_status.info(f"⏳ [{tf_idx2+1}/{total_tf}] {tf} 最適化中...")
                    d_eval = EVAL_BARS_DEFAULT.get(tf, 5)
                    d_bars = {"1M": 1500, "5M": 800, "15M": 500}.get(tf, 300)

                    try:
                        df_tf, _ = get_ohlcv_dataframe(symbol, tf, count=d_bars)
                    except Exception:
                        all_tf_results[tf] = {"error": "データ取得失敗"}
                        continue

                    if df_tf is None or len(df_tf) < 50:
                        all_tf_results[tf] = {"error": "データ不足"}
                        continue

                    _ep_tf = get_entry_params(tf)
                    _ap_tf = get_ai_params(tf)

                    if _do_entry_all:
                        with st.spinner(f"{tf} エントリー最適化中..."):
                            best_ep_tf, best_es_tf, _ = _optimize_entry(
                                df_tf, d_eval, jst_offset, pip_size, d_eval,
                                n_iter=entry_n_iter,
                            )
                        if best_es_tf and best_es_tf.get("count", 0) > 0:
                            _ep_tf = best_ep_tf

                    if _do_ai_all:
                        with st.spinner(f"{tf} AI最適化中..."):
                            best_ap_tf, best_as_tf, _ = _optimize_ai(
                                df_tf, d_eval, jst_offset, pip_size, d_eval,
                                n_iter=ai_n_iter,
                            )
                        if best_as_tf and best_as_tf.get("count", 0) > 0:
                            _ap_tf = best_ap_tf

                    _save_params(tf, _ep_tf, _ap_tf)
                    all_tf_results[tf] = {
                        "entry_pf":  best_es_tf.get("profit_factor", 0) if _do_entry_all else None,
                        "entry_wr":  best_es_tf.get("win_rate", 0)      if _do_entry_all else None,
                        "ai_pf":     best_as_tf.get("profit_factor", 0) if _do_ai_all   else None,
                        "ai_wr":     best_as_tf.get("win_rate", 0)      if _do_ai_all   else None,
                    }
                    overall_prog.progress((tf_idx2 + 1) / total_tf,
                                          text=f"全体進捗: {tf_idx2+1}/{total_tf} 完了")

                tf_status.empty()
                overall_prog.progress(1.0, text="全時間足 最適化完了 ✓")
                st.success("✅ 全時間足のパラメーターを保存しました。チャートを再読み込みすると反映されます。")

                # 結果サマリー
                rows = []
                for tf, r in all_tf_results.items():
                    if "error" in r:
                        rows.append({"TF": tf, "状態": f"⚠️ {r['error']}",
                                     "Entry PF": "―", "Entry 勝率": "―",
                                     "AI PF": "―", "AI 勝率": "―"})
                    else:
                        def _fmt_pf(v):
                            return "∞" if v == float("inf") else f"{min(v,5):.2f}" if v else "―"
                        def _fmt_wr(v):
                            return f"{v*100:.1f}%" if v else "―"
                        rows.append({
                            "TF": tf, "状態": "✅ 保存済み",
                            "Entry PF": _fmt_pf(r.get("entry_pf")),
                            "Entry 勝率": _fmt_wr(r.get("entry_wr")),
                            "AI PF": _fmt_pf(r.get("ai_pf")),
                            "AI 勝率": _fmt_wr(r.get("ai_wr")),
                        })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ============================================================
    # エントリーシグナル手動タブ
    # ============================================================
    with tab_entry:
        col_param, col_result = st.columns([1, 2])

        with col_param:
            st.subheader(f"パラメーター（{timeframe}）")
            cd_max = max(ep["cooldown"] * 3, 10)
            adx_min = st.slider("ADX 最小値", 5, 40, ep["adx_min"], 1)
            score_thresh = st.slider("スコア閾値（最大9点）", 2, 8, ep["score_thresh"], 1)
            stoch_strong = st.slider("Stoch 強シグナル閾値", 10, 35, ep["stoch_strong"], 5)
            stoch_normal = st.slider("Stoch 通常シグナル閾値", 25, 45, ep["stoch_normal"], 5)
            body_mult    = st.slider("実体倍率", 0.05, 0.5, ep["body_mult"], 0.05)
            cooldown     = st.slider(f"クールダウン  ※30分={default_eval}本",
                                     1, cd_max, ep["cooldown"], 1)

        entry_params = {
            "adx_min": adx_min, "score_thresh": score_thresh,
            "stoch_strong": stoch_strong, "stoch_normal": stoch_normal,
            "body_mult": body_mult, "cooldown": cooldown,
        }

        with col_result:
            st.subheader("バックテスト結果")
            with st.spinner("計算中..."):
                try:
                    entry_markers = calc_entry_signals(df, jst_offset, params=entry_params)
                    lc = sum(1 for m in entry_markers if m["shape"] == "arrowUp")
                    sc = sum(1 for m in entry_markers if m["shape"] == "arrowDown")
                    st.caption(f"Long: {lc}件 / Short: {sc}件")
                    _display_stats(_evaluate_signals(df, entry_markers, eval_bars, jst_offset, pip_size))
                except Exception as e:
                    st.error(f"計算エラー: {e}"); entry_markers = []

        if entry_markers:
            st.subheader("損益推移")
            _show_pnl_chart(df, entry_markers, eval_bars, jst_offset, pip_size)

        st.divider()
        if st.button("エントリーパラメーターを保存", type="primary", key="save_entry"):
            _save_params(timeframe, entry_params, ap)
            st.success(f"{timeframe} のパラメーターを保存しました（{datetime.now().strftime('%H:%M:%S')}）。チャートを再読み込みすると反映されます。")

    # ============================================================
    # AI シグナル手動タブ
    # ============================================================
    with tab_ai:
        col_param2, col_result2 = st.columns([1, 2])

        with col_param2:
            st.subheader(f"パラメーター（{timeframe}）")
            cd_max_ai = max(ap["cooldown"] * 3, 10)
            forward      = st.slider(f"先読みバー数  ※30分={default_eval}本",
                                     1, max(default_eval * 3, 15), ap["forward"], 1)
            k_val        = st.slider("k（近傍数）", 5, 30, ap["k"], 1)
            min_prob     = st.slider("最低確率閾値", 0.55, 0.90, ap["min_prob"], 0.05)
            ai_cooldown  = st.slider(f"クールダウン  ※30分={default_eval}本",
                                     1, cd_max_ai, ap["cooldown"], 1)
            adx_min_ai   = st.slider("ADX 最小値", 5, 40, ap["adx_min"], 1)
            atr_mult     = st.slider("ラベリング ATR 倍率", 0.3, 3.0, ap["atr_mult"], 0.1)

        ai_params = {
            "forward": forward, "k": k_val, "min_prob": min_prob,
            "cooldown": ai_cooldown, "adx_min": adx_min_ai, "atr_mult": atr_mult,
        }

        with col_result2:
            st.subheader("バックテスト結果")
            with st.spinner("計算中（k-NN は少し時間がかかります）..."):
                try:
                    ai_markers = calc_ai_signal(df, jst_offset, params=ai_params)
                    lc2 = sum(1 for m in ai_markers if m["shape"] == "arrowUp")
                    sc2 = sum(1 for m in ai_markers if m["shape"] == "arrowDown")
                    st.caption(f"Long: {lc2}件 / Short: {sc2}件")
                    _display_stats(_evaluate_signals(df, ai_markers, eval_bars, jst_offset, pip_size))
                except Exception as e:
                    st.error(f"計算エラー: {e}"); ai_markers = []

        if ai_markers:
            st.subheader("損益推移")
            _show_pnl_chart(df, ai_markers, eval_bars, jst_offset, pip_size)

        st.divider()
        if st.button("AIパラメーターを保存", type="primary", key="save_ai"):
            _save_params(timeframe, ep, ai_params)
            st.success(f"{timeframe} のパラメーターを保存しました（{datetime.now().strftime('%H:%M:%S')}）。チャートを再読み込みすると反映されます。")


if __name__ == "__main__":
    main()
else:
    main()
