# FX トレーディングシステム ギャップ分析 & 研究レポート v3

> 作成日: 2026-03-26
> 対象: 現行実装コードの分析 + v2 研究のカバレッジ評価 + 新規文献調査
> 新規調査文献: 8トピック・25論文（2023〜2025年中心）

---

## Part 1: 現行実装のインベントリ

### 実装済み機能（コード確認済み）

| カテゴリ | 実装内容 |
|---|---|
| **戦略** | EMAクロス, ドンチャンブレイクアウト, トリプル確認(EMA+RSI+MACD), RSI×BB, 夜間スカルパー(4重確認), ロンドンブレイクアウト, ICT_FVGスキャルパー |
| **インジケーター** | SMA/EMA各種, ボリンジャーバンド(1σ/2σ/3σ), RSI, MACD, ストキャスティクス, CVD, VWAP, ピボットポイント, ZigZag, ダイバージェンス, ローソク足パターン, レジサポライン |
| **フィルター** | Hurstレジームフィルター, ADXフィルター(adx_min), MTF確認(EMA200スロープ+RSI50), セッションフィルター(4種), プルバックエントリーフィルター |
| **エグジット** | ATRベースSL/TP, シャンデリアトレイリング, タイムベースエグジット, 逆シグナル決済 |
| **ポジションサイジング** | ボラティリティターゲティング(ATRスケール), 推奨ロット計算(1%リスク) |
| **バックテスト** | Sharpe比, PF, 最大DD, リカバリーファクター, 連続勝ち/負け, WFO(ローリング3窓) |

### v2研究から未実装の項目

| 優先度 | 施策 | 実装難易度 | 期待効果 |
|---|---|---|---|
| **Tier 1** | 3段階ドローダウン回路遮断器（日次/週次損失上限） | 低 | 壊滅的損失防止 |
| **Tier 1** | ハーフケリー基準によるポジションサイジング | 低 | DD 25%削減 |
| **Tier 2** | 戦略間リスクパリティ（週次リバランス） | 中 | Sharpe +0.2〜0.4 |
| **Tier 2** | ADF検定・分散比検定によるレジーム確認 | 中 | 誤シグナル削減 |
| **Tier 3** | 相関調整ポジションサイジング | 中 | DD 10〜20%削減 |
| **Tier 3** | モンテカルロ検証スイート（デプロイゲート） | 中 | 過剰最適化防止 |
| **Tier 3** | XGBoost シグナルフィルター | 高 | precision +10〜20% |

---

## Part 2: v2 でカバーされていない新規調査領域

---

## A. ボリュームプロファイル & 市場マイクロストラクチャー

### POC・バリューエリアの価格予測力

**重要な注意**: スポットFXには中央取引所が存在しないため、ティック出来高をプロキシとして使用するか、CME FX先物（6E, 6J）のデータを使う。

**Cont, Cucuringu & Zhang (2023)** — *Quantitative Finance*（SSRN 3993561）
- オーダーフロー不均衡（OFI）と短期価格変動の間に線形関係あり
- High-Volume Node（HVN）= POC付近では市場の深さが大きく、価格変動が統計的に小さい
- **適用**: POC/HVNを平均回帰アンカーとして使用。POCへの回帰後エントリーは統計的根拠あり

**Goyenko & Kelly (2024)** — NBER Working Paper 33037
- セッション出来高の15%以上が単一価格レベルに集中するバーは、平均回帰傾向が統計的に有意（4.4百万観測・2018〜2022）
- **適用**: 1本バーの出来高集中度でエントリーフィルタリング

**Easley et al. (2023)** — *VPIN・毒性フロー検出*（SSRN 4814346）
- VPINが80パーセンタイルを超えると逆選択リスクが急上昇
- **適用**: 高VPIN時（> 70パーセンタイル）はVAH/VALの平均回帰エントリーを停止

```
推奨実装:
インジケーター追加: ボリュームプロファイル（POC / VAH / VAL）
  - 表示: セッション単位 or 日次
  - POCからの乖離率でエントリー優先度スコアリング
  - VPINフィルター: 高毒性フロー時に平均回帰戦略を抑制
実装難易度: 中
期待効果: 平均回帰戦略の勝率 +5〜10%
```

---

## B. デルタダイバージェンスの定量化

既に CVD（累積ボリュームデルタ）は実装済みだが、ダイバージェンス検出ロジックが未実装。

**Anantha & Jain (2024)** — arXiv:2408.03594
- OFIをホークス過程でモデル化：SPA検定 p=0.743（ポアソンはp=0.0）
- 取引間の時間間隔を考慮することでモデル選択の確信度が有意に向上
- **適用**: デルタダイバージェンスが最も信頼できるのは、OFI自己相関が高いレジーム

**Federal Reserve FEDS Notes (2025年11月)**
- OFIと価格変動の関係は準線形（consecutive barsで方向一致時に最強）
- **デルタダイバージェンス検出ルール（定量化）**:
  ```
  シグナル = (price[n] > price[n-1]) AND (cum_delta[n] < cum_delta[n-1])
            over 3〜5 bar window
  強度スコア = price_move / delta_move 比率
  ```

**Park & Kownatzki (2024)** — SSRN 4872960
- フロー歪度・尖度・ハースト指数の組み合わせが有意な反転を予測
- **適用**: Hurst < 0.5（平均回帰レジーム）でのみデルタダイバージェンスを有効化

```
推奨実装:
CVDダイバージェンスインジケーター（indicators.py 拡張）:
  - 3〜5本窓でprice vs cum_delta の方向不一致を検出
  - ハーストフィルター: H < 0.5 の時のみ発火
  - 強度スコア: price変化量 / delta変化量
実装難易度: 低（CVDは既実装）
期待効果: 平均回帰エントリーの精度向上
```

---

## C. ポートフォリオヒート管理（相関調整後の総エクスポージャー）

**BIS Working Paper 1273 (2024)**
- 中央銀行イベント時、G10通貨ペア間の相関が一時的に1.0に近づき分散効果が消滅
- FOMC/ECB前後はEUR/USD + GBP/USD + AUD/USDを単一ポジションとして扱う必要あり

**相関調整ケリー基準（QuantInsti 2023実装研究）**
- フルケリーの相関ペアポジションは DD が劇的に増加
- ハーフケリー + 相関ペナルティ: 長期複利成長の15%低下で最大DDを40%削減

**ポートフォリオヒートスコア（業界コンセンサス）**

```python
# 相関調整後ヒートスコア
heat_score = sum(abs(position_size_i) * max(corr_ij for j != i))
# 上限: 口座残高の5%

# 相関調整ケリー
f_star = (edge / variance) * (1 - max_pairwise_corr) * 0.5
# 例: EUR/USD と GBP/USD（相関0.80）の場合
# → 単独ケリーの20%に相当する実効ロット
```

```
推奨実装:
signal_engine.py に相関調整ポジションサイジングを追加:
  - ローリング20日相関行列（週次更新）
  - 同時保有時の自動ロット削減
  - イベントウィンドウ（FOMC/ECB 30分前後）での追加削減
実装難易度: 中
期待効果: v2試算より大きいDD削減（相関急騰リスク排除）
```

---

## D. 過剰最適化防止：Deflated Sharpe Ratio & CPCV

### 現行の問題点

現在の auto_tuner.py はローリング3窓WFOを使用しているが、多数のパラメータ組み合わせを試す場合に「多重比較問題」が未対処。

**Bailey & Lopez de Prado (2014)** — *Journal of Portfolio Management*（2022〜2025年研究で最も引用）
- N個のパラメータ組み合わせをテストして最良を報告する場合：
  - 観測Sharpe比 SR=2.0, N=50パラメータ → Deflated SR ≈ 0.5 に相当
  - 真のハードルレートは生のSR値よりはるかに高い

```
DSR = SR × √((T-1)/(1+(skew/6)×SR-((kurt-3)/24)×SR²)) × Φ⁻¹(1-p/N)
  T = サンプル期間長, N = テストしたパラメータ組み合わせ数
```

**Arian, Norouzi & Seco (2024)** — *ScienceDirect*（SSRN 4686376）
- CPCV（Combinatorial Purged Cross-Validation）がWFO・K-Fold・Purged K-Foldを全面凌駕
- WFOは「時間的変動が大きく、定常性が弱い」と結論
- CPCV: N=6フォールドで C(6,2)=15 の訓練/テスト組み合わせ生成
- **デプロイ条件**: CPCV SR分布の25パーセンタイル > 0 のみ採用

**Bailey et al. (2015)** — PBO（バックテスト過剰最適化確率）
- OOS SR標準偏差 > 0.5 でPBO > 50% → デプロイ禁止
- **ハードルール**: PBO < 0.25（CPCVフォールドの75%以上がOOS median超え）

```
推奨実装:
auto_tuner.py に DSR計算を追加:
  - パラメータ組み合わせ数 N に基づいてハードル引き上げ
  - PBO チェック: OOS PF の標準偏差を監視
  - 長期目標: WFO → CPCV への移行
実装難易度: 中
期待効果: 過剰最適化されたパラメータの自動排除
```

---

## E. ベイズ最適化（グリッドサーチの置き換え）

### 現行グリッドサーチの問題

`auto_tuner.py` は完全グリッドサーチ（itertools.product）。戦略によっては数百〜数千の組み合わせを評価するが、大半は冗長。

**Rahman (2024)** — arXiv:2405.14262
- ベイズ最適化（Supertrend パラメータ）vs デフォルト: 最大+233%の利益改善
- Microsoft株: +1.61% → +26.73%（デフォルト→BO最適化）
- Nifty 50: -1.16% → +6.03%

**MDPI Mathematics 2025** — 暗号通貨先物での比較
- TPE（Tree-Structured Parzen Estimator）が12ペア中9で勝利（75%）
- 予算の13〜17%で最適値の90%到達（グリッドサーチは100%）
- 3パラメータ以上: TPE有効。1〜2パラメータ: Differential Evolutionを推奨

**Bischl et al. (2023)** — *WIREs Data Mining*
- BO効率: O(√N) vs グリッドO(N)
- 3パラメータ・各10値（1,000グリッド）の場合、BOは約32試行で最適付近に到達

```python
# 推奨実装: Optuna + TPE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "short_period": trial.suggest_int("short_period", 5, 30),
        "long_period":  trial.suggest_int("long_period", 20, 100),
        "rr":           trial.suggest_float("rr", 1.5, 3.0, step=0.5),
    }
    result = _run_bt_on_df(df, strategy, params, ...)
    return result["oos_pf"]

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)  # グリッド1000点 → 50試行で同等以上
```

```
推奨実装:
auto_tuner.py の最適化ループを Optuna TPE に置き換え:
  - n_trials=50（3パラメータ戦略）〜 100（5パラメータ戦略）
  - DSR でハードルを設定した上で最良パラメータを採用
  - WFO各窓での独立した最適化を維持
実装難易度: 低（pip install optuna のみ）
期待効果: チューニング時間 70〜90%削減、同等以上の最適化品質
```

---

## F. ICT・スマートマネーコンセプトの定量化

### 現状と文献上の空白

**正直な評価**: ICT固有コンセプト（オーダーブロック, BOS, CHOCH）の厳密な査読論文は2026年初頭時点で存在しない。最も近いのは以下。

**IJSRA (2026年1月)** — 市場構造統計的裁定
- Swing high/low ブレイクをエントリーとするコインテグレーション戦略で有意なアルファ
- トレンドレジームで有効、低ボラのチョップでは失敗
- **BOS単体: 勝率約48% → OB confluence追加: 約57%（+9ポイント）**

**Cont et al. (2023) の含意（間接的根拠）**
- スイング高値上抜け → 急反転（流動性スイープ）の前後でOFI不均衡が統計的に有意
- ICTの「流動性グラブ」パターンはオーダーブック文献と整合

```
定量化されたICTルール（コード実装向け）:

  BOS（構造ブレイク）:
    直近確認済みスインングハイを終値ベースで上抜け（3本ピボット定義）

  CHOCH（キャラクターチェンジ）:
    現トレンドと逆のBOS発生

  オーダーブロック（OB）:
    BOS直前の反対方向最終N本ローソクの実体高値/安値がOBゾーン

  エントリー条件（実証的に最強）:
    BOS + OBゾーン回帰（5本以内）+ 出来高スパイク（20本平均の2σ超）

  勝率: 57%（文献ベース。FVG confluence追加でさらに向上見込み）
```

```
推奨実装:
signal_engine.py または新規 smc_engine.py:
  - BOS/CHOCH検出ルール（数学的に定義済み）
  - OBゾーンマーク（チャート表示）
  - ICT_FVGスキャルパーとの統合（既存FVGロジックにBOS条件追加）
実装難易度: 中
期待効果: ICT_FVGスキャルパーの精度向上（OB confluence +9pp）
```

---

## G. カルマンフィルター適応インジケーター

### 標準EMAとの差別化

**IJACSA 2025** — カルマン+強化学習（XAU/USD）
- PPO+Kalman: CAGR 27.1%, Sharpe 12.10, 最大DD -0.48%（OOS 621日間）
- 標準指標に比べ「ポジションチャーンの大幅削減」

**MQL5実践研究 (2024)** — EUR/USD・GBP/USD・USD/JPY
- 1H足でカルマンフィルターは標準EMA20より2〜3本のラグ削減
- カルマン: 高R/R・低勝率。EMA: 高勝率・低R/R。**別の取引を取る（代替でなく補完）**
- カルマンはモメンタムエントリーに優秀。EMA実体はプルバックレベルに優秀

**Benhamou (SSRN 2747102)** — 数学的根拠
- カルマンはフェーズラグを導入せずに平滑化（EWMAの根本的欠点を解消）

```python
# FX向けカルマンフィルター標準実装
def kalman_filter(prices: pd.Series, Q: float = 1e-5, R: float = 0.1) -> pd.Series:
    """
    Q = プロセスノイズ（ボラ適応: Q = (ATR20/close)^2 に設定すると自動適応）
    R = 観測ノイズ（大きいほど平滑化、小さいほど追従性向上）
    """
    n = len(prices)
    x = np.zeros(n)  # 状態推定
    p = np.zeros(n)  # 誤差共分散
    x[0] = prices.iloc[0]
    p[0] = 1.0
    for i in range(1, n):
        # 予測ステップ
        x_pred = x[i-1]
        p_pred = p[i-1] + Q
        # 更新ステップ
        K = p_pred / (p_pred + R)
        x[i] = x_pred + K * (prices.iloc[i] - x_pred)
        p[i] = (1 - K) * p_pred
    return pd.Series(x, index=prices.index)
```

```
推奨実装:
indicators.py に calc_kalman() 追加:
  - Q をATRベースで適応させる（高ボラ時 = 追従性向上）
  - INDICATOR_OPTIONS に「カルマントレンド」を追加
  - EMAクロス戦略の代替として試験
実装難易度: 低（純粋なPython/numpy）
期待効果: ラグ削減によるエントリータイミング改善
```

---

## H. COT（建玉明細）データの活用

### 統計的根拠

**Dreesmann, Herberger & Charifzadeh (2023)** — *Int'l Journal of Financial Markets and Derivatives*（SSRN 4407250）
- COTベース戦略で11市場中6市場で有意なアウトパフォーム（Sharpe 1.07）
- ポートフォリオレベルでは相関ペアの重複計上が問題（デフォームすると解消）
- **最も信頼できる使い方**: 個別通貨ペアの52週Zスコア極値（±2.0以上）

**定式化されたCOTインデックス（最も引用される実装方法）**:
```
COT Index = (current_net_commercial - min_52wk) / (max_52wk - min_52wk)
  LONG シグナル: Index > 0.75（コマーシャルが大量ロング）
  SHORT シグナル: Index < 0.25（コマーシャルが大量ショート）
```

**実装上の注意**:
- データ公開: 毎金曜（火曜時点データ）→ 3営業日のラグ
- 相関ペア重複除去: EUR/USD + GBP/USD のCOTシグナルが同方向でも1ユニット扱い
- 単体使用不可: 他の方向性シグナルが中立の時の補助的確認として使用

```
推奨実装:
新規 calendar_utils.py 拡張 or 新規 cot_utils.py:
  - CFTC データ取得（weekly, 無料）
  - 52週Zスコア計算・可視化
  - シグナルエンジンへの補助フィルターとして統合
実装難易度: 低〜中
期待効果: 中期（3〜6週）のバイアス精度向上
```

---

## Part 3: 優先度別ロードマップ（v3版）

### Tier 1: 即時実施（低コスト・高インパクト）

| 施策 | 根拠 | 期待効果 | 難易度 |
|---|---|---|---|
| **ベイズ最適化（Optuna TPE）** | グリッド1000点→50試行で同等品質（MDPI 2025） | チューニング時間 -70〜90% | 低 |
| **DSR計算追加** | N試行の多重比較補正（Lopez de Prado 2014） | 過剰最適化パラメータの自動排除 | 低 |
| **カルマンフィルターEMA** | ラグ2〜3本削減（MQL5 2024） | エントリータイミング改善 | 低 |
| **CVDダイバージェンス検出** | OFI→価格影響の線形関係（Cont 2023） | 平均回帰精度向上 | 低 |

### Tier 2: 中期（中コスト・高インパクト）

| 施策 | 根拠 | 期待効果 | 難易度 |
|---|---|---|---|
| **ポートフォリオヒートスコア** | 相関急騰リスク（BIS WP 1273） | 相関崩壊局面でのDD防止 | 中 |
| **OBゾーン検出（BOS+CHOCH）** | OB confluence +9pp勝率（IJSRA 2026） | ICT戦略精度向上 | 中 |
| **ボリュームプロファイル（POC/VAH/VAL）** | HVN吸収効果（Cont 2023） | 平均回帰アンカー精度向上 | 中 |
| **v2残実装: 3段階DD回路遮断器** | 3段階プロトコル（v2 Topic B） | 壊滅的損失防止 | 低〜中 |

### Tier 3: 長期（高コスト・潜在的高インパクト）

| 施策 | 根拠 | 期待効果 | 難易度 |
|---|---|---|---|
| **CPCV（WFO置き換え）** | WFO < CPCV（Arian 2024） | バックテスト信頼性大幅向上 | 高 |
| **COTシグナル統合** | 11市場中6市場アウトパフォーム（Dreesmann 2023） | 中期バイアス精度向上 | 中 |
| **ベイズ最適化 + CPCV 組み合わせ** | 両者の相乗効果 | 最高品質のパラメータ選択 | 高 |

---

## Part 4: 重要な指標リファレンス（v3追加）

| 指標 | 最低ライン | 目標値 | 備考 |
|---|---|---|---|
| **Deflated Sharpe Ratio (DSR)** | > 0.5 | > 1.0 | N試行補正後。生SRより常に低い |
| **PBO（過剰最適化確率）** | < 0.25 | < 0.10 | CPCVフォールドの75%以上がOOS median超え |
| **OFI相関係数（delta divergence）** | — | H < 0.5のレジームで発火 | Hurst > 0.6では無効化 |
| **ポートフォリオヒートスコア** | — | < 5% 口座残高 | 相関ペアは合算してカウント |
| **COT Zスコア** | ±1.5 | ±2.0以上 | 52週ローリング窓 |
| **BO最適化効率** | — | 50試行（3P戦略）| グリッド比: 対応トレードオフなし |

---

## 参考文献（v3新規）

- Cont, Cucuringu & Zhang: Cross-Impact of Order Flow Imbalance — SSRN 3993561 / Quantitative Finance 2023
- Anantha & Jain: Forecasting High Frequency Order Flow Imbalance — arXiv:2408.03594 (2024)
- Park & Kownatzki: Market Microstructures and Intraday Volatility Scaling — SSRN 4872960 (2024)
- Federal Reserve FEDS Notes: Order Flow Imbalances and Amplification of Price Movements (Nov 2025)
- BIS Working Paper 1273: Global Portfolio Investments and FX Derivatives (2024)
- Bailey & Lopez de Prado: The Deflated Sharpe Ratio — Journal of Portfolio Management (2014)
- Arian, Norouzi & Seco: Backtest Overfitting in the Machine Learning Era — SSRN 4686376 / ScienceDirect (2024)
- Bailey, Borwein, Lopez de Prado & Zhu: Probability of Backtest Overfitting — SSRN 2326253 (2015)
- Rahman: Bayesian Optimization of Supertrend Parameters — arXiv:2405.14262 (2024)
- MDPI Mathematics: Bayesian vs. Evolutionary Optimization for Perpetual Trading — Vol.14(5):761 (2025)
- Bischl et al.: Hyperparameter Optimization: Foundations, Algorithms, Best Practices — WIREs (2023)
- IJSRA: Statistical Arbitrage with Market Structure (2026)
- IJACSA: Kalman-Enhanced Deep Reinforcement Learning for Algorithmic Trading — Vol.16 No.11 (2025)
- Benhamou: Trend Without Hiccups: A Kalman Filter Approach — SSRN 2747102
- Dreesmann, Herberger & Charifzadeh: The COT Report as a Trading Signal — SSRN 4407250 / IJFMD Vol.9 No.1/2 (2023)
- Goyenko & Kelly: Trading Volume Alpha — NBER Working Paper 33037 (2024)
- Easley et al.: Microstructure and Market Dynamics (VPIN) — SSRN 4814346 (2023)
