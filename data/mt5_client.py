"""
data/mt5_client.py

MetaTrader 5 クライアント。
MT5本体と接続してローソク足データを取得し pandas DataFrame で返す。

認証情報は .env ファイルで管理する（git管理外）。
.env が未設定の場合は起動中のMT5に自動接続を試みる。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# プロジェクトルートの .env を読み込む
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


# ============================================================
# 定数マッピング
# ============================================================

# 時間足 → MT5 タイムフレーム定数（MT5未接続時のフォールバック用）
TIMEFRAME_MAP: dict[str, int] = {
    "1M":  1,      # mt5.TIMEFRAME_M1
    "5M":  5,      # mt5.TIMEFRAME_M5
    "15M": 15,     # mt5.TIMEFRAME_M15
    "30M": 30,     # mt5.TIMEFRAME_M30
    "1H":  16385,  # mt5.TIMEFRAME_H1
    "4H":  16388,  # mt5.TIMEFRAME_H4
    "1D":  16408,  # mt5.TIMEFRAME_D1
    "1W":  32769,  # mt5.TIMEFRAME_W1
}


def _get_mt5_timeframe(timeframe: str) -> int:
    """時間足文字列を MT5 タイムフレーム定数に変換する。"""
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 ライブラリがインストールされていません")
    mapping = {
        "1M":  mt5.TIMEFRAME_M1,
        "5M":  mt5.TIMEFRAME_M5,
        "15M": mt5.TIMEFRAME_M15,
        "30M": mt5.TIMEFRAME_M30,
        "1H":  mt5.TIMEFRAME_H1,
        "4H":  mt5.TIMEFRAME_H4,
        "1D":  mt5.TIMEFRAME_D1,
        "1W":  mt5.TIMEFRAME_W1,
    }
    if timeframe not in mapping:
        raise KeyError(f"未対応の時間足: {timeframe}")
    return mapping[timeframe]


# ============================================================
# MT5Client
# ============================================================

# 自動検出を試みる MT5 インストールパス候補（優先順）
_CANDIDATE_PATHS: list[str] = [
    r"C:\Program Files\OANDA MetaTrader 5\terminal64.exe",
    r"C:\Program Files\OANDA MetaTrader 5 x64\terminal64.exe",
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    r"C:\Program Files\Forex.com MetaTrader 5\terminal64.exe",
    r"C:\Program Files\Saxo Bank MetaTrader 5\terminal64.exe",
    r"C:\Program Files\IC Markets MetaTrader 5\terminal64.exe",
    r"C:\Program Files\XM Global MetaTrader 5\terminal64.exe",
    r"C:\Program Files\FXGT MetaTrader 5\terminal64.exe",
    r"C:\Program Files\Titan FX MetaTrader 5\terminal64.exe",
]


def _find_mt5_path() -> str | None:
    """インストール済みの MT5 実行ファイルパスを自動検出する。"""
    env_path = os.getenv("MT5_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for p in _CANDIDATE_PATHS:
        if Path(p).exists():
            return p
    return None


class MT5Client:
    """
    MetaTrader 5 に接続してローソク足を取得するクライアント。

    .env に MT5_LOGIN / MT5_PASSWORD / MT5_SERVER が設定されていれば
    Python から直接ログインする。未設定の場合は起動中の MT5 に自動接続する。
    MT5 のパスは自動検出する（複数ブローカーに対応）。
    """

    def __init__(self) -> None:
        if not MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 ライブラリがインポートできません。"
                " pipenv install MetaTrader5 を実行してください。"
            )

        login    = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server   = os.getenv("MT5_SERVER")
        mt5_path = _find_mt5_path()

        kwargs: dict = {}
        if mt5_path:
            kwargs["path"] = mt5_path
        if login and password and server:
            kwargs["login"]    = int(login)
            kwargs["password"] = password
            kwargs["server"]   = server

        ok = mt5.initialize(**kwargs)

        if not ok:
            err = mt5.last_error()
            hint = (
                f"MT5 の初期化に失敗しました: {err}\n"
                "・MetaTrader 5 アプリを起動してログインしてください。\n"
                "・または .env に MT5_LOGIN / MT5_PASSWORD / MT5_SERVER を設定してください。"
            )
            if not mt5_path:
                hint += "\n・MT5 がインストールされていないか、パスが不明です。.env に MT5_PATH を設定してください。"
            raise ConnectionError(hint)

    def __del__(self) -> None:
        if MT5_AVAILABLE:
            mt5.shutdown()

    # ----------------------------------------------------------

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500,
    ) -> pd.DataFrame:
        """
        ローソク足を取得して DataFrame で返す。

        Args:
            symbol:    通貨ペア（例: "USDJPY"）
            timeframe: 時間足（例: "1H"）
            count:     取得本数

        Returns:
            pd.DataFrame: index=timestamp(UTC), columns=[open, high, low, close, volume]

        Raises:
            ValueError: データ取得に失敗した場合
        """
        tf = _get_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            raise ValueError(
                f"MT5からデータを取得できませんでした: {err}\n"
                f"シンボル '{symbol}' がブローカーで利用可能か確認してください。"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "time":       "timestamp",
            "open":       "open",
            "high":       "high",
            "low":        "low",
            "close":      "close",
            "tick_volume": "volume",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.set_index("timestamp")
        df.sort_index(inplace=True)
        return df

    # ----------------------------------------------------------

    def fetch_candles_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: "datetime",
        date_to:   "datetime",
    ) -> "pd.DataFrame":
        """
        日付範囲を指定してローソク足を取得する。

        Args:
            symbol:    通貨ペア
            timeframe: 時間足
            date_from: 開始日時（UTC aware datetime）
            date_to:   終了日時（UTC aware datetime）
        """
        from datetime import datetime
        tf    = _get_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)

        if rates is None or len(rates) == 0:
            raise ValueError(
                f"MT5からデータを取得できませんでした: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "time": "timestamp", "tick_volume": "volume"
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.set_index("timestamp")
        df.sort_index(inplace=True)
        return df

    def fetch_candles_max(
        self,
        symbol: str,
        timeframe: str,
        max_bars: int = 99999,
    ) -> "pd.DataFrame":
        """
        取得可能な最大本数のローソク足を取得する。
        ブローカーが保持している全履歴を引き出す用途。
        """
        tf    = _get_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, max_bars)

        if rates is None or len(rates) == 0:
            raise ValueError(
                f"MT5からデータを取得できませんでした: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "time": "timestamp", "tick_volume": "volume"
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.set_index("timestamp")
        df.sort_index(inplace=True)
        return df

    def get_available_symbols(self) -> list[str]:
        """ブローカーで利用可能なシンボル一覧を返す。"""
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return [s.name for s in symbols]

    def is_symbol_available(self, symbol: str) -> bool:
        """シンボルが利用可能か確認する。"""
        info = mt5.symbol_info(symbol)
        return info is not None


# ============================================================
# モジュールレベルのシングルトン
# ============================================================

_client: MT5Client | None = None


def get_client() -> MT5Client:
    """MT5Client のシングルトンを返す（接続済みを使い回す）。"""
    global _client
    if _client is None:
        _client = MT5Client()
    return _client


def reset_client() -> None:
    """クライアントをリセットする（再接続用）。"""
    global _client
    _client = None


def is_available() -> bool:
    """MT5ライブラリが使用可能か確認する（接続はしない）。"""
    return MT5_AVAILABLE


def is_connected() -> bool:
    """MT5に接続済みか確認する（未初期化・切断時は再接続を試みる）。"""
    if not MT5_AVAILABLE:
        return False
    global _client
    try:
        # terminal_info() が None なら切断されている → クライアントをリセットして再試行
        if _client is not None and mt5.terminal_info() is None:
            _client = None
        get_client()  # シングルトン経由で初期化・接続
        return mt5.terminal_info() is not None
    except Exception:
        _client = None  # 次回の呼び出しで再試行できるようリセット
        return False


# 最後に接続を試みた時刻（頻繁な再試行を防ぐ）
_last_attempt: float = 0.0
_RETRY_INTERVAL: float = 5.0  # 秒


def try_connect() -> bool:
    """
    MT5 への接続を試みる。
    _RETRY_INTERVAL 秒以内の連続呼び出しはスキップする。

    Returns:
        True = 接続成功, False = 失敗
    """
    global _last_attempt
    now = time.time()
    if now - _last_attempt < _RETRY_INTERVAL:
        return is_connected()
    _last_attempt = now
    reset_client()
    return is_connected()


# ============================================================
# 動作確認
# ============================================================

if __name__ == "__main__":
    print("MT5 接続テスト中...")
    try:
        client = MT5Client()
        print("接続成功")
        df = client.fetch_candles("USDJPY", "1H", count=5)
        print(df)
        print(f"\n取得件数: {len(df)}")
    except Exception as e:
        print(f"エラー: {e}")
