"""
config/signal_defaults.py

時間足ごとのシグナルデフォルトパラメーター。

【設計思想】
  1M / 5M : スキャルピング（30分以内決済）
             → FORWARD・COOLDOWN を分換算で30分相当に設定
             → ADX・スコア閾値は低め（短期は小さなモメンタムを拾う）
             → ATR 倍率は低め（小さな値動きでもラベリングできるように）

  15M / 30M: デイトレード（数時間）
             → 中間的な設定

  1H 〜 1D : スイング（数日）
             → 閾値高め・クールダウン短め（足数が少ない）
"""

from __future__ import annotations

# 時間足 → 30分相当のバー数（FORWARD / COOLDOWN の基準）
_BARS_30MIN: dict[str, int] = {
    "1M":  30,
    "5M":   6,
    "15M":  2,
    "30M":  1,
    "1H":   1,
    "4H":   1,
    "1D":   1,
    "1W":   1,
}

# ============================================================
# エントリーシグナル デフォルトパラメーター
# ============================================================
ENTRY_DEFAULTS: dict[str, dict] = {
    "1M": {
        "adx_min":      12,   # 1M は短期ノイズが多いため低め
        "score_thresh":  3,   # スキャルピングは頻繁にシグナルが必要
        "stoch_strong": 20,   # 強シグナル閾値（狭い逆張りゾーン）
        "stoch_normal": 30,
        "body_mult":   0.15,  # 小さな足でも通るよう低め
        "cooldown":     30,   # 30分（=30本）は再エントリーしない
    },
    "5M": {
        "adx_min":      12,
        "score_thresh":  3,
        "stoch_strong": 22,
        "stoch_normal": 32,
        "body_mult":   0.18,
        "cooldown":      6,   # 30分（=6本）
    },
    "15M": {
        "adx_min":      15,
        "score_thresh":  4,
        "stoch_strong": 25,
        "stoch_normal": 35,
        "body_mult":   0.20,
        "cooldown":      4,
    },
    "30M": {
        "adx_min":      15,
        "score_thresh":  4,
        "stoch_strong": 25,
        "stoch_normal": 35,
        "body_mult":   0.20,
        "cooldown":      3,
    },
    "1H": {
        "adx_min":      15,
        "score_thresh":  4,
        "stoch_strong": 25,
        "stoch_normal": 35,
        "body_mult":   0.20,
        "cooldown":      3,
    },
    "4H": {
        "adx_min":      20,
        "score_thresh":  5,
        "stoch_strong": 25,
        "stoch_normal": 35,
        "body_mult":   0.25,
        "cooldown":      2,
    },
    "1D": {
        "adx_min":      20,
        "score_thresh":  5,
        "stoch_strong": 25,
        "stoch_normal": 35,
        "body_mult":   0.30,
        "cooldown":      2,
    },
    "1W": {
        "adx_min":      20,
        "score_thresh":  5,
        "stoch_strong": 25,
        "stoch_normal": 35,
        "body_mult":   0.30,
        "cooldown":      2,
    },
}

# ============================================================
# AI シグナル デフォルトパラメーター
# ============================================================
AI_DEFAULTS: dict[str, dict] = {
    "1M": {
        "forward":        30,   # 30分後（=30本）の値動きでラベリング
        "k":              20,   # 1M はノイズが多いため近傍を多めに
        "min_prob":      0.65,  # 1M は予測難しいため閾値低め
        "cooldown":       30,   # 30分間は再シグナルなし
        "adx_min":        12,
        "atr_mult":       0.8,  # 1M の小さな値動きをラベリング
        "recency_decay":  3.0,  # 1M は相場変化が速いため直近重視
    },
    "5M": {
        "forward":         6,   # 30分後（=6本）
        "k":              20,
        "min_prob":      0.65,
        "cooldown":        6,   # 30分
        "adx_min":        12,
        "atr_mult":       1.0,
        "recency_decay":  2.5,
    },
    "15M": {
        "forward":         5,
        "k":              15,
        "min_prob":      0.70,
        "cooldown":        4,
        "adx_min":        15,
        "atr_mult":       1.5,
        "recency_decay":  2.0,
    },
    "30M": {
        "forward":         4,
        "k":              15,
        "min_prob":      0.70,
        "cooldown":        3,
        "adx_min":        15,
        "atr_mult":       1.5,
        "recency_decay":  2.0,
    },
    "1H": {
        "forward":         5,
        "k":              15,
        "min_prob":      0.70,
        "cooldown":        3,
        "adx_min":        15,
        "atr_mult":       1.5,
        "recency_decay":  2.0,
    },
    "4H": {
        "forward":         3,
        "k":              15,
        "min_prob":      0.72,
        "cooldown":        2,
        "adx_min":        20,
        "atr_mult":       1.8,
        "recency_decay":  1.5,
    },
    "1D": {
        "forward":         3,
        "k":              15,
        "min_prob":      0.72,
        "cooldown":        2,
        "adx_min":        20,
        "atr_mult":       2.0,
        "recency_decay":  1.0,
    },
    "1W": {
        "forward":         2,
        "k":              10,
        "min_prob":      0.72,
        "cooldown":        2,
        "adx_min":        20,
        "atr_mult":       2.0,
        "recency_decay":  1.0,
    },
}

# チューニングページの評価バー数デフォルト（30分相当）
EVAL_BARS_DEFAULT: dict[str, int] = _BARS_30MIN


def get_entry_params(timeframe: str, saved: dict | None = None) -> dict:
    """
    保存済みパラメーターがあればそれを、なければ時間足のデフォルトを返す。
    saved は signal_params.json の "entry" キー以下（timeframe をキーに持つ dict）。
    """
    defaults = ENTRY_DEFAULTS.get(timeframe, ENTRY_DEFAULTS["1H"])
    if saved and timeframe in saved:
        return {**defaults, **saved[timeframe]}
    return defaults


def get_ai_params(timeframe: str, saved: dict | None = None) -> dict:
    """
    保存済みパラメーターがあればそれを、なければ時間足のデフォルトを返す。
    saved は signal_params.json の "ai" キー以下（timeframe をキーに持つ dict）。
    """
    defaults = AI_DEFAULTS.get(timeframe, AI_DEFAULTS["1H"])
    if saved and timeframe in saved:
        return {**defaults, **saved[timeframe]}
    return defaults
