"""
config/settings.py

プロジェクト全体の設定定数を管理するモジュール。
新しい通貨ペア・時間足・パスを追加する場合はこのファイルのみ変更する。
"""

from __future__ import annotations  # Python 3.9未満での list[str] 等を有効化

from pathlib import Path

# ============================================================
# プロジェクトルート
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# 対応通貨ペア
# ============================================================
# 将来的に追加する場合はリストに追記するだけでOK
# 例: "EURUSD", "GBPJPY", "AUDUSD"

# 通貨ペアのグループ分け（サイドバーUI用）
SYMBOL_GROUPS: dict[str, list[str]] = {
    "円ペア (JPY)": [
        "USDJPY", "EURJPY", "GBPJPY", "AUDJPY",
        "CADJPY", "NZDJPY", "CHFJPY",
    ],
    "ドルペア (USD)": [
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
        "USDCAD", "USDCHF",
    ],
    "クロス円以外": [
        "EURGBP", "EURAUD", "EURCAD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPCHF",
        "AUDCAD", "AUDNZD", "AUDCHF",
        "NZDCAD", "NZDCHF", "CADCHF",
    ],
}

# フラットなリスト（グループ順を維持）
SUPPORTED_SYMBOLS: list[str] = [
    s for group in SYMBOL_GROUPS.values() for s in group
]


# ============================================================
# 対応時間足
# ============================================================
# 短い時間足から長い時間足の順に並べておく（可読性のため）

SUPPORTED_TIMEFRAMES: list[str] = [
    "1M",   # 1分足
    "5M",   # 5分足
    "15M",  # 15分足
    "30M",  # 30分足
    "1H",   # 1時間足
    "4H",   # 4時間足
    "1D",   # 日足
    "1W",   # 週足
]

# 時間足 → 分数のマッピング（ソート・比較・リサンプルに使用）
TIMEFRAME_MINUTES: dict[str, int] = {
    "1M":  1,
    "5M":  5,
    "15M": 15,
    "30M": 30,
    "1H":  60,
    "4H":  240,
    "1D":  1440,
    "1W":  10080,
}


# ============================================================
# データ保存パス
# ============================================================

DATA_DIR = ROOT_DIR / "output" / "data"
REPORT_DIR = ROOT_DIR / "output" / "reports"
CHART_DIR = ROOT_DIR / "output" / "charts"
LOG_DIR = ROOT_DIR / "output" / "logs"

# データは output/data/{symbol}/{timeframe}/ に保存
# 例: output/data/USDJPY/1H/2024-01.parquet
def get_data_path(symbol: str, timeframe: str) -> Path:
    """指定シンボル・時間足のデータ保存ディレクトリを返す。"""
    return DATA_DIR / symbol / timeframe


# ============================================================
# 松井証券API
# ============================================================

MATSUI_API_BASE_URL = "https://api.matsui.co.jp"  # 要確認・変更
MATSUI_API_RATE_LIMIT_SECONDS = 1.0   # リクエスト間隔（秒）
MATSUI_API_RETRY_COUNT = 3            # リトライ回数
MATSUI_API_TIMEOUT_SECONDS = 30       # タイムアウト（秒）


# ============================================================
# バックテスト デフォルト値
# ============================================================

BACKTEST_DEFAULT_CAPITAL = 1_000_000  # 初期資金（円）
BACKTEST_DEFAULT_SPREAD_PIPS = 0.3    # デフォルトスプレッド（pips）
BACKTEST_DEFAULT_LOT_SIZE = 10_000    # 1ロットのサイズ（通貨単位）


# ============================================================
# バリデーション用ヘルパー
# ============================================================

def is_valid_symbol(symbol: str) -> bool:
    """通貨ペアが対応リストに含まれるか確認する。"""
    return symbol in SUPPORTED_SYMBOLS


def is_valid_timeframe(timeframe: str) -> bool:
    """時間足が対応リストに含まれるか確認する。"""
    return timeframe in SUPPORTED_TIMEFRAMES


# ============================================================
# 起動時に出力ディレクトリをまとめて作成するユーティリティ
# ============================================================

def ensure_output_dirs() -> None:
    """output/ 以下の必要なディレクトリをすべて作成する。"""
    for symbol in SUPPORTED_SYMBOLS:
        for timeframe in SUPPORTED_TIMEFRAMES:
            get_data_path(symbol, timeframe).mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 動作確認用エントリポイント
# ============================================================

if __name__ == "__main__":
    print("=== Settings ===")
    print(f"ROOT_DIR       : {ROOT_DIR}")
    print(f"DATA_DIR       : {DATA_DIR}")
    print(f"SYMBOLS        : {SUPPORTED_SYMBOLS}")
    print(f"TIMEFRAMES     : {SUPPORTED_TIMEFRAMES}")
    print()
    print("=== Data Paths ===")
    for symbol in SUPPORTED_SYMBOLS:
        for tf in SUPPORTED_TIMEFRAMES:
            print(f"  {symbol}/{tf} -> {get_data_path(symbol, tf)}")
    print()
    print("Creating output directories...")
    ensure_output_dirs()
    print("Done.")
