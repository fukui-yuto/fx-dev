"""
data/local_store.py

OHLCVデータのローカルSQLiteストレージ。
- data/ohlcv.db に保存（gitignore推奨）
- MT5から取得したデータを蓄積・差分更新
- バックテスト等で長期データを参照可能
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "ohlcv.db"


# ============================================================
# DB初期化
# ============================================================

def init_db() -> None:
    """テーブルとインデックスを作成する（既存はスキップ）。"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol    TEXT    NOT NULL,
                timeframe TEXT    NOT NULL,
                timestamp INTEGER NOT NULL,
                open      REAL    NOT NULL,
                high      REAL    NOT NULL,
                low       REAL    NOT NULL,
                close     REAL    NOT NULL,
                volume    REAL    NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stf
            ON ohlcv (symbol, timeframe, timestamp)
        """)
        conn.commit()


# ============================================================
# 書き込み
# ============================================================

def upsert(symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """DataFrame を DB に UPSERT する。戻り値は挿入/更新件数。"""
    if df.empty:
        return 0
    init_db()
    records = [
        (symbol, timeframe,
         int(ts.timestamp()),
         float(row["open"]), float(row["high"]),
         float(row["low"]),  float(row["close"]),
         float(row["volume"]))
        for ts, row in df.iterrows()
    ]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv "
            "(symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()
    return len(records)


# ============================================================
# 読み込み
# ============================================================

def query(
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end:   datetime | None = None,
) -> pd.DataFrame:
    """
    DBからOHLCVを取得して DataFrame で返す。
    index は UTC DatetimeTZDtype。
    """
    init_db()
    sql    = ("SELECT timestamp, open, high, low, close, volume "
              "FROM ohlcv WHERE symbol=? AND timeframe=?")
    params: list = [symbol, timeframe]
    if start:
        sql += " AND timestamp >= ?"
        params.append(int(start.timestamp()))
    if end:
        sql += " AND timestamp <= ?"
        params.append(int(end.timestamp()))
    sql += " ORDER BY timestamp"

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp")
    return df


def get_latest_timestamp(symbol: str, timeframe: str) -> datetime | None:
    """指定シンボル/時間足の最新タイムスタンプを返す。なければ None。"""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT MAX(timestamp) FROM ohlcv WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        )
        row = cur.fetchone()
    if row and row[0] is not None:
        return datetime.fromtimestamp(row[0], tz=timezone.utc)
    return None


# ============================================================
# 統計
# ============================================================

def get_stats() -> pd.DataFrame:
    """保存済みデータの一覧（シンボル・時間足・件数・期間）を返す。"""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT symbol, timeframe,
                   COUNT(*)    AS bars,
                   MIN(timestamp) AS first_ts,
                   MAX(timestamp) AS last_ts
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
            """,
            conn,
        )
    if df.empty:
        return pd.DataFrame(columns=["symbol", "timeframe", "bars", "開始", "終了"])

    df["開始"] = pd.to_datetime(df["first_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    df["終了"] = pd.to_datetime(df["last_ts"],  unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    return df[["symbol", "timeframe", "bars", "開始", "終了"]].rename(
        columns={"bars": "件数"}
    )


def delete_data(symbol: str, timeframe: str) -> int:
    """指定シンボル/時間足のデータをDBから削除する。"""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "DELETE FROM ohlcv WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        )
        conn.commit()
        return cur.rowcount
