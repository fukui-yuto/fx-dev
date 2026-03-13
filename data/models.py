"""
data/models.py

市場データの型定義とバリデーションを担うモジュール。
OHLCVData は Pydantic BaseModel を継承し、受け取ったデータの
整合性（高値≥低値など）を生成時に自動検証する。
"""

from __future__ import annotations  # Python 3.9未満での list[X] | None 等を有効化

import sys
from datetime import datetime
from typing import Dict, List, Optional

# Annotated は Python 3.9+ / typing_extensions で利用可能
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ============================================================
# 型エイリアス（可読性向上）
# ============================================================

Price = Annotated[float, Field(gt=0, description="価格（正の実数）")]
Volume = Annotated[float, Field(ge=0, description="出来高（0以上の実数）")]


# ============================================================
# OHLCVData — 1本のローソク足データ
# ============================================================

class OHLCVData(BaseModel):
    """
    1本のローソク足（OHLCV）を表すデータモデル。

    Pydantic によりフィールドの型・値域・OHLCV整合性を
    インスタンス生成時に自動検証する。

    Attributes:
        symbol:    通貨ペア (例: "USDJPY")
        timeframe: 時間足   (例: "1H", "4H", "1D")
        timestamp: 足の開始時刻 (UTC推奨)
        open:      始値
        high:      高値
        low:       安値
        close:     終値
        volume:    出来高（通貨単位）
    """

    symbol: str = Field(
        ...,
        description="通貨ペア (例: 'USDJPY')",
        examples=["USDJPY", "EURUSD"],
    )
    timeframe: str = Field(
        ...,
        description="時間足 (例: '1H', '4H', '1D')",
        examples=["1M", "5M", "15M", "1H", "4H", "1D"],
    )
    timestamp: datetime = Field(
        ...,
        description="ローソク足の開始時刻（UTC推奨）",
    )
    open: Price = Field(
        ...,
        description="始値 — その足の最初の取引価格",
    )
    high: Price = Field(
        ...,
        description="高値 — その足の最高取引価格",
    )
    low: Price = Field(
        ...,
        description="安値 — その足の最低取引価格",
    )
    close: Price = Field(
        ...,
        description="終値 — その足の最後の取引価格",
    )
    volume: Volume = Field(
        default=0.0,
        description="出来高 — その足の取引量（通貨単位）",
    )

    # ----------------------------------------------------------
    # OHLCV 整合性バリデーション
    # ----------------------------------------------------------

    @model_validator(mode="after")
    def validate_ohlcv_consistency(self) -> OHLCVData:
        """
        OHLCV の論理的整合性を検証する。

        Rules:
            - high >= open, close, low
            - low  <= open, close, high
        """
        if self.high < self.open:
            raise ValueError(
                f"high ({self.high}) は open ({self.open}) 以上でなければなりません"
            )
        if self.high < self.close:
            raise ValueError(
                f"high ({self.high}) は close ({self.close}) 以上でなければなりません"
            )
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) は low ({self.low}) 以上でなければなりません"
            )
        if self.low > self.open:
            raise ValueError(
                f"low ({self.low}) は open ({self.open}) 以下でなければなりません"
            )
        if self.low > self.close:
            raise ValueError(
                f"low ({self.low}) は close ({self.close}) 以下でなければなりません"
            )
        return self

    # ----------------------------------------------------------
    # 便利メソッド
    # ----------------------------------------------------------

    @property
    def body(self) -> float:
        """実体の大きさ（終値 − 始値の絶対値）。"""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> float:
        """上ひげの長さ（高値 − max(始値, 終値)）。"""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """下ひげの長さ（min(始値, 終値) − 安値）。"""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """陽線（終値 > 始値）なら True。"""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """陰線（終値 < 始値）なら True。"""
        return self.close < self.open

    def to_dict(self) -> dict:
        """DataFrame 生成などに使いやすい辞書形式で返す。"""
        return {
            "timestamp": self.timestamp,
            "open":      self.open,
            "high":      self.high,
            "low":       self.low,
            "close":     self.close,
            "volume":    self.volume,
        }


# ============================================================
# OHLCVSeries — 複数本のローソク足をまとめるコンテナ
# ============================================================

class OHLCVSeries(BaseModel):
    """
    同一シンボル・時間足の OHLCVData リストとメタ情報をまとめるコンテナ。

    Attributes:
        symbol:    通貨ペア
        timeframe: 時間足
        bars:      ローソク足のリスト（古い順）
    """

    symbol: str
    timeframe: str
    bars: List[OHLCVData] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_series_consistency(self) -> OHLCVSeries:
        """全 bar の symbol・timeframe が自身と一致するか検証する。"""
        for bar in self.bars:
            if bar.symbol != self.symbol:
                raise ValueError(
                    f"bar.symbol '{bar.symbol}' が series.symbol '{self.symbol}' と不一致"
                )
            if bar.timeframe != self.timeframe:
                raise ValueError(
                    f"bar.timeframe '{bar.timeframe}' が series.timeframe '{self.timeframe}' と不一致"
                )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """
        bars を pandas DataFrame に変換する。

        Returns:
            pd.DataFrame: index=timestamp, columns=[open, high, low, close, volume]
        """
        if not self.bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        records = [bar.to_dict() for bar in self.bars]
        df = pd.DataFrame(records).set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> OHLCVSeries:
        """
        DataFrame から OHLCVSeries を生成する。

        Args:
            df:        index=timestamp, columns=[open, high, low, close, volume]
            symbol:    通貨ペア
            timeframe: 時間足

        Returns:
            OHLCVSeries
        """
        bars = [
            OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=idx,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0.0),
            )
            for idx, row in df.iterrows()
        ]
        return cls(symbol=symbol, timeframe=timeframe, bars=bars)

    @property
    def length(self) -> int:
        """ローソク足の本数。"""
        return len(self.bars)

    @property
    def start(self) -> Optional[datetime]:
        """最初の足のタイムスタンプ。"""
        return self.bars[0].timestamp if self.bars else None

    @property
    def end(self) -> Optional[datetime]:
        """最後の足のタイムスタンプ。"""
        return self.bars[-1].timestamp if self.bars else None


# ============================================================
# 動作確認用エントリポイント
# ============================================================

if __name__ == "__main__":
    from datetime import timezone

    print("=== OHLCVData バリデーションテスト ===\n")

    # --- 正常ケース ---
    bar = OHLCVData(
        symbol="USDJPY",
        timeframe="1H",
        timestamp=datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
        open=148.50,
        high=149.20,
        low=148.30,
        close=149.00,
        volume=1500.0,
    )
    print(f"[OK] bar={bar}")
    print(f"     body={bar.body:.4f}  upper={bar.upper_shadow:.4f}  lower={bar.lower_shadow:.4f}")
    print(f"     is_bullish={bar.is_bullish}")

    # --- 異常ケース: high < close ---
    print("\n[異常ケース: high < close]")
    try:
        bad = OHLCVData(
            symbol="USDJPY",
            timeframe="1H",
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            open=148.50,
            high=148.00,   # ← closeより低い
            low=148.30,
            close=149.00,
            volume=0.0,
        )
    except Exception as e:
        print(f"  ValidationError (期待通り): {e}")

    # --- OHLCVSeries → DataFrame ---
    print("\n=== OHLCVSeries → DataFrame ===")
    series = OHLCVSeries(symbol="USDJPY", timeframe="1H", bars=[bar])
    df = series.to_dataframe()
    print(df)
    print(f"\nlength={series.length}  start={series.start}  end={series.end}")
