"""
dashboard/ai_learner.py

AI 自己学習モジュール。

シグナル発生後、N 本経過したら結果を自動評価し
output/ai_feedback.json に蓄積する。
正解パターンを k-NN の追加参照データとして渡すことで
時間とともに精度が向上する仕組み。

ユーザーの操作は一切不要。
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR       = Path(__file__).resolve().parent.parent
_FEEDBACK_DIR  = ROOT_DIR / "output"
MAX_FEEDBACK   = 1000   # 時間足ごとに保持する最大フィードバック数
CORRECT_WEIGHT = 3.0    # 正解パターンの参照重み
FORWARD_DEFAULT = 5     # 評価バー数（デフォルト）

# 後方互換：旧来の単一ファイルパス（learning.py 等が直接参照している場合用）
FEEDBACK_FILE = _FEEDBACK_DIR / "ai_feedback.json"


def _feedback_file(timeframe: str | None) -> Path:
    """時間足ごとのフィードバックファイルパスを返す。"""
    if timeframe:
        return _FEEDBACK_DIR / f"ai_feedback_{timeframe}.json"
    return FEEDBACK_FILE


def list_feedback_timeframes() -> list[str]:
    """フィードバックデータが存在する時間足リストを返す（保存順）。"""
    tfs = []
    for p in sorted(_FEEDBACK_DIR.glob("ai_feedback_*.json")):
        tf = p.stem.replace("ai_feedback_", "")
        if tf:
            tfs.append(tf)
    return tfs


# ============================================================
# 特徴量抽出（calc_ai_signal と同じ計算）
# ============================================================

def _compute_indicators(df: pd.DataFrame) -> dict:
    """バックテスト・評価に必要な全インジを計算して辞書で返す。"""
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n      = len(df)
    cl     = pd.Series(close)

    # RSI(9)
    _d   = cl.diff()
    rsi9 = (100 - 100 / (1 + _d.clip(lower=0).rolling(9).mean()
             / (-_d.clip(upper=0)).rolling(9).mean().replace(0, float("nan")))).values

    # Stoch %K(5,3,3)
    _rk      = 100 * (cl - cl.rolling(5).min()) / (
                   pd.Series(high).rolling(5).max() - pd.Series(low).rolling(5).min()
               ).replace(0, float("nan"))
    stoch_k  = _rk.rolling(3).mean().values

    # MACD(5,13,5) ヒスト
    _macd = cl.ewm(span=5, adjust=False).mean() - cl.ewm(span=13, adjust=False).mean()
    hist5 = (_macd - _macd.ewm(span=5, adjust=False).mean()).values

    # ATR7 / ATR14
    _tr   = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:]  - close[:-1]),
    ])
    _tr_s = np.concatenate([[float("nan")], _tr])
    atr7  = pd.Series(_tr_s).rolling(7).mean().values
    atr14 = pd.Series(_tr_s).rolling(14).mean().values

    # BB %B(14)
    bb_mid = cl.rolling(14).mean()
    bb_std = cl.rolling(14).std()
    bb_pct = ((cl - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, float("nan"))).values

    # EMA20
    ema20  = cl.ewm(span=20, adjust=False).mean().values

    # Volume ratio
    vol_ma10 = pd.Series(volume).rolling(10).mean().values

    return {
        "close": close, "high": high, "low": low, "volume": volume,
        "rsi9": rsi9, "stoch_k": stoch_k, "hist5": hist5,
        "atr7": atr7, "atr14": atr14, "bb_pct": bb_pct,
        "ema20": ema20, "vol_ma10": vol_ma10, "n": n,
    }


def _extract_feat(ind: dict, i: int) -> np.ndarray | None:
    """バーiの7次元特徴量を返す。計算不能なら None。"""
    a7  = ind["atr7"][i]
    a14 = ind["atr14"][i]
    if np.isnan(a7) or a7 <= 0:
        return None
    vals = [ind["rsi9"][i], ind["stoch_k"][i], ind["hist5"][i],
            ind["bb_pct"][i], ind["ema20"][i], ind["vol_ma10"][i]]
    if any(np.isnan(v) for v in vals):
        return None

    feat = np.zeros(7)
    feat[0] = np.clip(ind["rsi9"][i]   / 100.0, 0, 1)
    feat[1] = np.clip(ind["stoch_k"][i] / 100.0, 0, 1)
    feat[2] = np.clip(ind["hist5"][i] / a7, -3, 3) / 3.0
    feat[3] = np.clip(ind["bb_pct"][i], 0, 1)
    feat[4] = np.clip((ind["close"][i] - ind["ema20"][i]) / a7, -3, 3) / 3.0
    vol_r   = ind["volume"][i] / ind["vol_ma10"][i] if ind["vol_ma10"][i] > 0 else 1.0
    feat[5] = np.clip(vol_r, 0, 3) / 3.0
    feat[6] = np.clip((a7 / a14) - 1.0, -1, 1) if not np.isnan(a14) else 0.0
    return feat


# ============================================================
# 自動評価
# ============================================================

def auto_evaluate(
    df: pd.DataFrame,
    markers: list[dict],
    jst_offset: int,
    forward: int = FORWARD_DEFAULT,
    atr_mult: float = 1.5,
) -> list[dict]:
    """
    シグナルマーカーを自動評価する。

    forward 本後に結果が判明したシグナルのみ評価（まだ判明していないものはスキップ）。

    Returns:
        [
            {
                "ts":        int,    # シグナルのタイムスタンプ（JST オフセット済み）
                "direction": 1|-1,   # 1=LONG, -1=SHORT
                "correct":   bool,   # 正解かどうか
                "pips":      float,  # 結果（pips）
                "feat":      list,   # 特徴量ベクトル
            },
            ...
        ]
    """
    if not markers or len(df) < forward + 2:
        return []

    ind    = _compute_indicators(df)
    close  = ind["close"]
    high   = ind["high"]
    low    = ind["low"]
    atr7   = ind["atr7"]
    ts_seq = df.index
    # 価格が50以上なら JPY ペア（USDJPY≈150, EURJPY≈160 等）と判断
    pip_size = 0.01 if float(close[-1]) > 50 else 0.0001

    # timestamp → 行インデックス
    ts_to_idx = {int(t.timestamp()) + jst_offset: i for i, t in enumerate(ts_seq)}

    results = []
    for m in markers:
        t   = m["time"]
        idx = ts_to_idx.get(t)
        if idx is None:
            continue
        # まだ forward 本経過していない（未来）はスキップ
        if idx + forward >= len(df) - 1:
            continue

        a7 = atr7[idx]
        if np.isnan(a7) or a7 <= 0:
            continue

        direction  = 1 if m["shape"] == "arrowUp" else -1
        entry      = close[idx]
        fwd_high   = float(np.max(high[idx + 1: idx + forward + 1]))
        fwd_low    = float(np.min(low[idx + 1:  idx + forward + 1]))
        gain       = fwd_high - entry
        loss       = entry - fwd_low
        threshold  = atr_mult * a7

        if direction == 1:
            correct = gain >= threshold and gain >= 1.5 * loss
            pips    = (close[idx + forward] - entry) / pip_size
        else:
            correct = loss >= threshold and loss >= 1.5 * gain
            pips    = (entry - close[idx + forward]) / pip_size

        feat = _extract_feat(ind, idx)
        results.append({
            "ts":        t,
            "direction": direction,
            "correct":   bool(correct),
            "pips":      round(float(pips), 2),
            "feat":      feat.tolist() if feat is not None else None,
        })

    return results


# ============================================================
# フィードバック永続化
# ============================================================

def load_feedback(timeframe: str | None = None) -> list[dict]:
    """
    保存済みフィードバックを読み込む。

    timeframe を指定するとその時間足専用のファイルを読む。
    None の場合は旧来の共通ファイルを読む（後方互換）。
    """
    path = _feedback_file(timeframe)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_feedback(new_results: list[dict], timeframe: str | None = None) -> None:
    """
    新しい評価結果を追記する（重複排除・上限制限あり）。

    timeframe を指定するとその時間足専用のファイルへ保存する。
    """
    path        = _feedback_file(timeframe)
    existing    = load_feedback(timeframe)
    existing_ts = {r["ts"] for r in existing}
    to_add      = [r for r in new_results
                   if r["ts"] not in existing_ts and r.get("feat") is not None]
    if not to_add:
        return
    merged = existing + to_add
    merged = sorted(merged, key=lambda x: x["ts"], reverse=True)[:MAX_FEEDBACK]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")


def get_extra_labels(feedback: list[dict]) -> list[dict]:
    """
    フィードバックから k-NN 追加参照データを生成する。

    正解したシグナルのパターンを高ウェイトで返す。
    これを calc_ai_signal の extra_labels に渡すことで
    モデルが実績から学習する。

    Returns:
        [{"feat": np.ndarray, "label": 1|-1, "weight": float}, ...]
    """
    extras = []
    for r in feedback:
        if not r.get("feat") or not r["correct"]:
            continue
        feat = np.array(r["feat"])
        if np.any(np.isnan(feat)):
            continue
        extras.append({
            "feat":   feat,
            "label":  r["direction"],
            "weight": CORRECT_WEIGHT,
        })
    return extras


# ============================================================
# 学習統計
# ============================================================

def get_stats(feedback: list[dict]) -> dict:
    """
    蓄積フィードバックの統計を返す。

    Returns:
        {
            "total":          int,
            "correct":        int,
            "win_rate":       float,
            "avg_pips":       float,
            "recent_wr":      float,   # 直近20件の勝率
            "long_wr":        float,
            "short_wr":       float,
            "trend":          "improving" | "declining" | "stable",
        }
    """
    if not feedback:
        return {
            "total": 0, "correct": 0, "win_rate": 0.0, "avg_pips": 0.0,
            "recent_wr": 0.0, "long_wr": 0.0, "short_wr": 0.0, "trend": "stable",
        }

    total   = len(feedback)
    correct = sum(1 for r in feedback if r["correct"])
    win_rate = correct / total

    pips_list = [r["pips"] for r in feedback]
    avg_pips  = float(np.mean(pips_list)) if pips_list else 0.0

    # 直近20件
    recent   = feedback[:20]  # すでに新しい順にソート済み
    recent_wr = sum(1 for r in recent if r["correct"]) / len(recent) if recent else 0.0

    # Long / Short 別
    longs  = [r for r in feedback if r["direction"] == 1]
    shorts = [r for r in feedback if r["direction"] == -1]
    long_wr  = sum(1 for r in longs  if r["correct"]) / len(longs)  if longs  else 0.0
    short_wr = sum(1 for r in shorts if r["correct"]) / len(shorts) if shorts else 0.0

    # トレンド（直近 vs 直前の勝率比較）
    # 20件以上で計算開始（40件まで待たない）
    trend = "stable"
    if total >= 20:
        if total >= 40:
            prev_sample = feedback[20:40]
        else:
            # 20〜39件の間は後半 10件 vs 前半 10件で比較
            mid = total // 2
            prev_sample = feedback[mid:]
        prev_wr = sum(1 for r in prev_sample if r["correct"]) / len(prev_sample) if prev_sample else 0.5
        if recent_wr - prev_wr > 0.05:
            trend = "improving"
        elif recent_wr - prev_wr < -0.05:
            trend = "declining"

    return {
        "total":    total,
        "correct":  correct,
        "win_rate": win_rate,
        "avg_pips": avg_pips,
        "recent_wr": recent_wr,
        "long_wr":  long_wr,
        "short_wr": short_wr,
        "trend":    trend,
    }
