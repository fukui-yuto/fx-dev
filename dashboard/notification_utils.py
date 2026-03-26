"""
dashboard/notification_utils.py

iPhone へのプッシュ通知ユーティリティ。
対応サービス:
  - Telegram Bot  (完全無料・iOSプッシュ◎・トレーダーに最もポピュラー)
  - ntfy.sh       (完全無料・アカウント不要・App Store あり)
  - Pushover      (買い切り $5・ネイティブ iOS push・信頼性◎)

設定ファイル: output/notification_config.json
  (output/ は .gitignore 済みなので Bot Token が漏れない)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "output" / "notification_config.json"

# 同一シグナルの再通知を防ぐクールダウン（秒）
_COOLDOWN_SEC = 60
_last_sent: dict[str, float] = {}


# ============================================================
# 設定ファイル読み書き
# ============================================================

def load_config() -> dict:
    """output/notification_config.json を読み込む。なければ空の設定を返す。"""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    """設定を output/notification_config.json に保存する。"""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 各サービスの送信実装
# ============================================================

def _send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    """Telegram Bot API で通知を送る（完全無料）。"""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id":    chat_id,
                "text":       message,
                "parse_mode": "HTML",
            },
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def _send_ntfy(topic: str, message: str, title: str, priority: str = "high") -> bool:
    """ntfy.sh で通知を送る（完全無料・アカウント不要）。"""
    try:
        r = requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title":    title,
                "Priority": priority,
                "Tags":     "chart_with_upwards_trend",
            },
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def _send_pushover(user_key: str, api_token: str, message: str, title: str) -> bool:
    """Pushover で通知を送る（買い切り $5）。"""
    try:
        r = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token":   api_token,
                "user":    user_key,
                "title":   title,
                "message": message,
                "sound":   "cashregister",
            },
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


# ============================================================
# Telegram 設定ヘルパー
# ============================================================

def get_telegram_chat_id(bot_token: str) -> str | None:
    """
    Bot に話しかけた最初のユーザーの chat_id を自動取得する。
    セットアップ時に一度だけ呼び出す。

    Returns:
        chat_id 文字列。取得できなければ None。
    """
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getUpdates",
            timeout=5,
        )
        data = r.json()
        results = data.get("result", [])
        if results:
            return str(results[-1]["message"]["chat"]["id"])
    except Exception:
        pass
    return None


# ============================================================
# 統合送信関数
# ============================================================

def send_entry_notification(
    symbol: str,
    timeframe: str,
    direction: str,   # "long" | "short"
    entry: float,
    sl_pips: float,
    tp_pips: float,
    strategy: str = "",
    confidence: str = "",
) -> bool:
    """
    エントリーシグナル発生時に iPhone へ通知を送る。
    設定済みのサービス全てに並行送信する。
    クールダウン中（60秒以内に同じシンボル・方向）はスキップ。
    """
    cfg = load_config()
    if not cfg.get("enabled", False):
        return False

    cooldown_key = f"{symbol}_{direction}"
    now = time.time()
    # symbol="TEST" はテスト送信なのでクールダウンをスキップ
    if symbol != "TEST" and now - _last_sent.get(cooldown_key, 0) < _COOLDOWN_SEC:
        return False

    arrow = "▲ BUY" if direction == "long" else "▼ SELL"
    title = f"FX シグナル: {symbol} {arrow}"
    # Telegram は HTML 対応なので太字を使う
    tg_body = (
        f"<b>{symbol} {timeframe}  {arrow}</b>\n"
        f"Entry: <code>{entry}</code>\n"
        f"SL: {sl_pips:.1f}pips  TP: {tp_pips:.1f}pips\n"
    )
    plain_body = (
        f"{symbol} {timeframe}  {arrow}\n"
        f"Entry: {entry}\n"
        f"SL: {sl_pips:.1f}pips  TP: {tp_pips:.1f}pips\n"
    )
    if strategy:
        tg_body    += f"戦略: {strategy}\n"
        plain_body += f"戦略: {strategy}\n"
    if confidence:
        tg_body    += f"信頼度: {confidence}"
        plain_body += f"信頼度: {confidence}"

    sent = False

    # Telegram
    tg_token   = cfg.get("telegram_bot_token", "")
    tg_chat_id = cfg.get("telegram_chat_id", "")
    if tg_token and tg_chat_id:
        sent = _send_telegram(tg_token, tg_chat_id, tg_body) or sent

    # ntfy.sh
    ntfy_topic = cfg.get("ntfy_topic", "")
    if ntfy_topic:
        sent = _send_ntfy(ntfy_topic, plain_body, title) or sent

    # Pushover
    po_user  = cfg.get("pushover_user_key", "")
    po_token = cfg.get("pushover_api_token", "")
    if po_user and po_token:
        sent = _send_pushover(po_user, po_token, plain_body, title) or sent

    if sent:
        _last_sent[cooldown_key] = now

    return sent


def is_configured() -> bool:
    """通知設定が1つ以上有効かどうか。"""
    cfg = load_config()
    if not cfg.get("enabled", False):
        return False
    return bool(
        (cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id")) or
        cfg.get("ntfy_topic") or
        (cfg.get("pushover_user_key") and cfg.get("pushover_api_token"))
    )
