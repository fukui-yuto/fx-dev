"""
fetch_data.py

MT5から全履歴データを取得してローカルDBに保存するスクリプト。
バックテスト前に一度実行しておく。

使い方:
    pipenv run python fetch_data.py
    pipenv run python fetch_data.py --symbols USDJPY --timeframes 1M 5M 1H
    pipenv run python fetch_data.py --symbols USDJPY --timeframes 1M --max-bars 50000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from data.local_store import get_latest_timestamp, get_stats, upsert


def fetch_all(
    symbols: list[str],
    timeframes: list[str],
    max_bars: int,
    diff_only: bool,
) -> None:
    from data.mt5_client import MT5Client, is_available

    if not is_available():
        print("[ERROR] MetaTrader5ライブラリが利用できません。")
        print("        pipenv install MetaTrader5 を実行してください。")
        sys.exit(1)

    print("MT5に接続中...")
    try:
        client = MT5Client()
    except Exception as e:
        print(f"[ERROR] MT5接続失敗: {e}")
        sys.exit(1)

    print("接続成功。\n")
    total_saved = 0
    errors: list[str] = []

    for sym in symbols:
        for tf in timeframes:
            latest = get_latest_timestamp(sym, tf)
            label  = f"{sym:>8} / {tf:<4}"

            if diff_only and latest is not None:
                # 差分更新：最新タイムスタンプ以降だけ取得
                try:
                    df = client.fetch_candles_range(
                        sym, tf, latest, datetime.now(timezone.utc)
                    )
                    cnt = upsert(sym, tf, df)
                    total_saved += cnt
                    print(f"  {label}  差分更新  +{cnt:>7,} 件  （最新: {latest.strftime('%Y-%m-%d %H:%M')}）")
                except Exception as e:
                    msg = f"  {label}  [SKIP] {e}"
                    print(msg)
                    errors.append(msg)
            else:
                # 全履歴取得
                try:
                    df = client.fetch_candles_max(sym, tf, max_bars=max_bars)
                    cnt = upsert(sym, tf, df)
                    total_saved += cnt
                    if not df.empty:
                        first = df.index.min().strftime("%Y-%m-%d")
                        last  = df.index.max().strftime("%Y-%m-%d")
                        print(f"  {label}  全履歴    {cnt:>7,} 件  （{first} 〜 {last}）")
                    else:
                        print(f"  {label}  [SKIP] データなし")
                except Exception as e:
                    msg = f"  {label}  [SKIP] {e}"
                    print(msg)
                    errors.append(msg)

    print(f"\n合計 {total_saved:,} 件を保存しました。")

    if errors:
        print(f"\n失敗した組み合わせ ({len(errors)}件):")
        for e in errors:
            print(e)

    print("\n--- 保存済みデータ一覧 ---")
    stats = get_stats()
    if stats.empty:
        print("（なし）")
    else:
        print(stats.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MT5から全履歴OHLCVデータを取得してローカルDBに保存する"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=SUPPORTED_SYMBOLS,
        help=f"取得する通貨ペア（デフォルト: {SUPPORTED_SYMBOLS}）",
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=SUPPORTED_TIMEFRAMES,
        help=f"取得する時間足（デフォルト: {SUPPORTED_TIMEFRAMES}）",
    )
    parser.add_argument(
        "--max-bars", type=int, default=99999,
        help="1シンボル・1時間足あたりの最大取得本数（デフォルト: 99999）",
    )
    parser.add_argument(
        "--diff", action="store_true",
        help="差分更新モード（最終保存日時以降のみ取得）",
    )
    args = parser.parse_args()

    invalid_tf = [tf for tf in args.timeframes if tf not in SUPPORTED_TIMEFRAMES]
    if invalid_tf:
        print(f"[ERROR] 未対応の時間足: {invalid_tf}")
        print(f"        対応時間足: {SUPPORTED_TIMEFRAMES}")
        sys.exit(1)

    mode = "差分更新" if args.diff else "全履歴取得"
    print(f"=== OHLCVデータ取得 ({mode}) ===")
    print(f"通貨ペア  : {args.symbols}")
    print(f"時間足    : {args.timeframes}")
    if not args.diff:
        print(f"最大本数  : {args.max_bars:,}")
    print()

    fetch_all(args.symbols, args.timeframes, args.max_bars, args.diff)


if __name__ == "__main__":
    main()
