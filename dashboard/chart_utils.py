"""
dashboard/chart_utils.py

チャート構築ロジック。
- ローソク足: TradingView Lightweight Charts（JSポーリングでズーム保持）
- 損益曲線:  Plotly
"""

from __future__ import annotations

import json
import os as _os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# OANDA Japan MT5 サーバーは UTC+3。JST(UTC+9)への差分は +6時間
JST_OFFSET = 6 * 3600

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

COLOR_BULL   = "#26a69a"
COLOR_BEAR   = "#ef5350"
COLOR_EQUITY = "#29b6f6"


# ============================================================
# ヘルパー
# ============================================================

def _price_format(symbol: str) -> tuple[int, float]:
    if symbol.endswith("JPY"):
        return 3, 0.001
    return 5, 0.00001


def _df_to_candles(df: pd.DataFrame) -> list:
    candles = []
    for ts, row in df.iterrows():
        t = int(ts.timestamp()) + JST_OFFSET
        candles.append({
            "time":  t,
            "open":  round(float(row["open"]),  5),
            "high":  round(float(row["high"]),  5),
            "low":   round(float(row["low"]),   5),
            "close": round(float(row["close"]), 5),
        })
    return candles


def _scale_margins(has_rsi: bool, has_macd: bool, has_stoch: bool = False, has_cvd: bool = False) -> dict:
    """サブインジケーターの有無に応じてscaleMarginsを返す。"""
    subs: list[str] = []
    if has_rsi:   subs.append("rsi")
    if has_stoch: subs.append("stoch")
    if has_macd:  subs.append("macd")
    if has_cvd:   subs.append("cvd")

    n = len(subs)
    if n == 0:
        return {"candles": (0.0, 0.0)}
    elif n == 1:
        return {"candles": (0.0, 0.30), subs[0]: (0.75, 0.0)}
    elif n == 2:
        return {"candles": (0.0, 0.44), subs[0]: (0.60, 0.22), subs[1]: (0.80, 0.0)}
    elif n == 3:
        return {
            "candles": (0.0, 0.55),
            subs[0]:   (0.60, 0.30),
            subs[1]:   (0.78, 0.12),
            subs[2]:   (0.90, 0.0),
        }
    else:  # n == 4
        return {
            "candles": (0.0, 0.60),
            subs[0]:   (0.65, 0.45),
            subs[1]:   (0.75, 0.32),
            subs[2]:   (0.85, 0.18),
            subs[3]:   (0.93, 0.0),
        }


# ============================================================
# JSON書き込み（fragment が毎秒呼ぶ）
# ============================================================

def write_panel_json(
    df: pd.DataFrame,
    panel_id: int,
    indicators_data: dict,
    events: list | None = None,
    signal_lines: dict | None = None,
    notification: dict | None = None,
    cvd_scale: int | None = None,
) -> None:
    """static/panel_{panel_id}.json に最新足のみ書き込む（ポーリング更新用）。
    全データでなく最新1本のみにすることでファイルサイズを最小化し、
    Tornado の stat→read 競合によるエラーを防ぐ。
    """
    _STATIC_DIR.mkdir(exist_ok=True)
    ts  = df.index[-1]
    row = df.iloc[-1]
    t   = int(ts.timestamp()) + JST_OFFSET
    last_candle = {
        "time":  t,
        "open":  round(float(row["open"]),  5),
        "high":  round(float(row["high"]),  5),
        "low":   round(float(row["low"]),   5),
        "close": round(float(row["close"]), 5),
    }
    # インジケーターも最新1点のみ（価格レベル系は全件）
    _FULL_DATA_KEYS = {"SR_lines", "Pivot_lines", "Session_markers", "Recent_HL", "ZigZag_line"}
    ind_last: dict = {}
    for k, v in indicators_data.items():
        data = v.get("data", [])
        if k in _FULL_DATA_KEYS:
            ind_last[k] = data
        elif data:
            ind_last[k] = data[-1]
    # チャートの時間範囲外のイベントを除去し、時刻順ソート
    # LightweightCharts は setMarkers() に時刻昇順ソートを要求する。
    # 未ソートだとズームや再描画時にマーカーが消える。
    t_min = int(df.index[0].timestamp())  + JST_OFFSET
    t_max = int(df.index[-1].timestamp()) + JST_OFFSET
    filtered_events = sorted(
        [e for e in (events or []) if t_min <= e["time"] <= t_max],
        key=lambda e: e["time"],
    )
    payload = {"candle": last_candle, "indicators": ind_last, "events": filtered_events, "signal_lines": signal_lines or {}, "notification": notification, "cvd_scale": cvd_scale}
    path = _STATIC_DIR / f"panel_{panel_id}.json"
    # 固定サイズ＋インプレース上書き:
    #   write_bytes() は O_TRUNC で一瞬 0 バイトにするため Tornado が
    #   Content-Length 不一致エラーを起こす。
    #   os.open を O_CREAT のみ（O_TRUNC なし）で開き、固定サイズで上書きすることで
    #   ファイルが 0 バイトになる瞬間をなくす。
    #   _FIXED_SIZE はペイロード最大値の上限。events が多い場合でも収まるよう
    #   65536（64KB）を確保する。これを超えると末尾に旧データが残り JSON が壊れる。
    _FIXED_SIZE = 65536
    raw = json.dumps(payload).encode("utf-8")
    if len(raw) < _FIXED_SIZE:
        raw = raw + b" " * (_FIXED_SIZE - len(raw))
    try:
        fd = _os.open(str(path), _os.O_WRONLY | _os.O_CREAT, 0o666)
        try:
            _os.write(fd, raw)
        finally:
            _os.close(fd)
    except PermissionError:
        pass  # Tornado が読み取り中 → 次の書き込みで再試行


# ============================================================
# チャートHTML生成
# ============================================================

def build_panel_html(
    df: pd.DataFrame,
    port: int,
    panel_id: int,
    symbol: str,
    indicators_data: dict,
    height: int = 500,
    initial_events: list | None = None,
    cvd_scale: int = 1,
) -> str:
    """
    LightweightChartsを埋め込んだHTMLを返す。
    JSが /app/static/panel_{panel_id}.json を1秒ごとにfetchする。
    initial_events を渡すとポーリング前から即座にマーカーを表示する。
    """
    candles             = _df_to_candles(df)
    ind_json            = {k: v["data"] for k, v in indicators_data.items()}
    initial             = json.dumps({"candles": candles, "indicators": ind_json, "events": initial_events or []})
    precision, min_move = _price_format(symbol)
    data_url            = f"http://localhost:{port}/app/static/panel_{panel_id}.json"

    has_rsi      = "RSI"        in indicators_data
    has_macd     = "MACD"       in indicators_data
    has_sr       = "SR_lines"   in indicators_data
    has_pivot    = "Pivot_lines" in indicators_data
    has_stoch     = "Stoch_K"     in indicators_data
    has_recent_hl = "Recent_HL"  in indicators_data
    has_zigzag    = "ZigZag_line" in indicators_data
    has_cvd       = "CVD_delta"   in indicators_data
    margins  = _scale_margins(has_rsi, has_macd, has_stoch, has_cvd)
    cm       = margins["candles"]

    # ---- overlay indicator JS ----
    # series宣言とsetDataを分離（initより前にsetDataを呼べないため）
    overlay_series = ""
    overlay_init   = ""
    overlay_update = ""
    for key, ind in indicators_data.items():
        if ind["type"] != "overlay":
            continue
        safe       = key.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("σ", "s")
        line_style = ind.get("lineStyle", 0)
        color      = ind["color"]
        overlay_series += f"""
const s_{safe} = chart.addLineSeries({{
  color:'{color}', lineWidth:1, lineStyle:{line_style},
  priceScaleId:'right', lastValueVisible:false, priceLineVisible:false,
}});"""
        overlay_init += f"""
s_{safe}.setData(init.indicators[{json.dumps(key)}] || []);"""
        overlay_update += f"""
    if (d.indicators?.[{json.dumps(key)}]) s_{safe}.update(d.indicators[{json.dumps(key)}]);"""

    # ---- RSI JS ----
    rsi_series = rsi_init = rsi_update = ""
    if has_rsi:
        rm = margins["rsi"]
        rsi_series = f"""
const rsiS = chart.addLineSeries({{color:'#7e57c2', lineWidth:1, priceScaleId:'rsi', lastValueVisible:false, priceLineVisible:false}});
chart.priceScale('rsi').applyOptions({{ scaleMargins:{{ top:{rm[0]}, bottom:{rm[1]} }} }});
rsiS.createPriceLine({{ price:70, color:'#ef535088', lineWidth:1, lineStyle:2, axisLabelVisible:true }});
rsiS.createPriceLine({{ price:30, color:'#26a69a88', lineWidth:1, lineStyle:2, axisLabelVisible:true }});
rsiS.createPriceLine({{ price:50, color:'#55555588', lineWidth:1, lineStyle:2, axisLabelVisible:false }});"""
        rsi_init   = "\nrsiS.setData(init.indicators.RSI || []);"
        rsi_update = "\n    if (d.indicators?.RSI) rsiS.update(d.indicators.RSI);"

    # ---- MACD JS ----
    macd_series = macd_init = macd_update = ""
    if has_macd:
        mm = margins["macd"]
        macd_series = f"""
const macdL = chart.addLineSeries({{color:'#2196f3', lineWidth:1, priceScaleId:'macd', lastValueVisible:false, priceLineVisible:false}});
const macdS = chart.addLineSeries({{color:'#ff9800', lineWidth:1, priceScaleId:'macd', lastValueVisible:false, priceLineVisible:false}});
const macdH = chart.addHistogramSeries({{priceScaleId:'macd', lastValueVisible:false}});
chart.priceScale('macd').applyOptions({{ scaleMargins:{{ top:{mm[0]}, bottom:{mm[1]} }} }});"""
        macd_init = """
macdL.setData(init.indicators.MACD || []);
macdS.setData(init.indicators.MACD_signal || []);
macdH.setData(init.indicators.MACD_hist || []);"""
        macd_update = """
    if (d.indicators?.MACD)        macdL.update(d.indicators.MACD);
    if (d.indicators?.MACD_signal) macdS.update(d.indicators.MACD_signal);
    if (d.indicators?.MACD_hist)   macdH.update(d.indicators.MACD_hist);"""

    # ---- レジサポライン JS ----
    sr_series = sr_init = sr_update = ""
    if has_sr:
        sr_series = """
let srLines = [];
function setSRLines(levels) {
  srLines.forEach(l => cSeries.removePriceLine(l));
  srLines = [];
  if (!levels) return;
  levels.forEach(lvl => {
    srLines.push(cSeries.createPriceLine({
      price: lvl.price,
      color: lvl.type === 'resistance' ? '#ef535099' : '#26a69a99',
      lineWidth: 1, lineStyle: 0, axisLabelVisible: true,
      title: lvl.type === 'resistance' ? 'R' : 'S',
    }));
  });
}"""
        sr_init   = "\nsetSRLines(init.indicators.SR_lines);"
        sr_update = "\n    if (d.indicators?.SR_lines) setSRLines(d.indicators.SR_lines);"

    # ---- ZigZag JS ----
    zigzag_series = zigzag_init = zigzag_update = ""
    if has_zigzag:
        zigzag_series = """
const zigzagS = chart.addLineSeries({
  color:'#ffeb3b', lineWidth:1, lineStyle:0,
  priceScaleId:'right', lastValueVisible:false, priceLineVisible:false,
  crosshairMarkerVisible:false,
});
"""
        zigzag_init   = "\nzigzagS.setData(init.indicators.ZigZag_line || []);"
        zigzag_update = "\n    if (d.indicators?.ZigZag_line) zigzagS.setData(d.indicators.ZigZag_line);"

    # ---- ストキャスティクス JS ----
    stoch_series = stoch_init = stoch_update = ""
    if has_stoch:
        sm = margins.get("stoch", (0.75, 0.0))
        stoch_series = f"""
const stochK = chart.addLineSeries({{color:'#26c6da', lineWidth:1, priceScaleId:'stoch', lastValueVisible:false, priceLineVisible:false}});
const stochD = chart.addLineSeries({{color:'#ff7043', lineWidth:1, priceScaleId:'stoch', lastValueVisible:false, priceLineVisible:false}});
chart.priceScale('stoch').applyOptions({{ scaleMargins:{{ top:{sm[0]}, bottom:{sm[1]} }} }});
stochK.createPriceLine({{ price:80, color:'#ef535088', lineWidth:1, lineStyle:2, axisLabelVisible:true }});
stochK.createPriceLine({{ price:20, color:'#26a69a88', lineWidth:1, lineStyle:2, axisLabelVisible:true }});
stochK.createPriceLine({{ price:50, color:'#55555588', lineWidth:1, lineStyle:2, axisLabelVisible:false }});"""
        stoch_init   = "\nstochK.setData(init.indicators.Stoch_K || []);\nstochD.setData(init.indicators.Stoch_D || []);"
        stoch_update = "\n    if (d.indicators?.Stoch_K) stochK.update(d.indicators.Stoch_K);\n    if (d.indicators?.Stoch_D) stochD.update(d.indicators.Stoch_D);"

    # ---- CVD JS ----
    cvd_series = cvd_init = cvd_update = ""
    if has_cvd:
        cvm = margins.get("cvd", (0.75, 0.0))
        cvd_series = f"""
let cvdScaleFactor = {cvd_scale};
let cvdRaw = [], cvdMaxAbs = 1;
const cvdH = chart.addHistogramSeries({{priceScaleId:'cvd', lastValueVisible:false, base:0}});
const cvdL = chart.addLineSeries({{color:'#fff176', lineWidth:1, priceScaleId:'cvd_cum', lastValueVisible:false, priceLineVisible:false}});
chart.priceScale('cvd').applyOptions({{ scaleMargins:{{ top:{cvm[0]}, bottom:{cvm[1]} }} }});
chart.priceScale('cvd_cum').applyOptions({{ scaleMargins:{{ top:{cvm[0]}, bottom:{cvm[1]} }}, visible:false }});
cvdH.createPriceLine({{ price:0, color:'#888888', lineWidth:1, lineStyle:0, axisLabelVisible:true, title:'0' }});
function _cvdApplyScale() {{
  const h = cvdMaxAbs / cvdScaleFactor;
  const p = function() {{ return {{priceRange: {{minValue: -h, maxValue: h}}}}; }};
  cvdH.applyOptions({{autoscaleInfoProvider: p}});
}}"""
        cvd_init = (
            "\ncvdRaw = (init.indicators.CVD_delta || []);"
            "\ncvdH.setData(cvdRaw);"
            "\ncvdL.setData(init.indicators.CVD_line || []);"
            "\nconst _absVals = cvdRaw.map(function(_d){return Math.abs(_d.value);});"
            "\ncvdMaxAbs = _absVals.length > 0 ? Math.max.apply(null, _absVals) : 1;"
            "\nif (!cvdMaxAbs || cvdMaxAbs === 0) cvdMaxAbs = 1;"
            "\n_cvdApplyScale();"
        )
        cvd_update = (
            "\n    if (d.cvd_scale != null && d.cvd_scale !== cvdScaleFactor) {"
            "\n        cvdScaleFactor = d.cvd_scale; _cvdApplyScale();"
            "\n    }"
            "\n    if (d.indicators?.CVD_delta) {"
            "\n        const _d = d.indicators.CVD_delta;"
            "\n        if (cvdRaw.length > 0 && cvdRaw[cvdRaw.length-1].time === _d.time) { cvdRaw[cvdRaw.length-1] = _d; } else { cvdRaw.push(_d); cvdMaxAbs = Math.max(cvdMaxAbs, Math.abs(_d.value)); }"
            "\n        cvdH.update(_d);"
            "\n    }"
            "\n    if (d.indicators?.CVD_line) cvdL.update(d.indicators.CVD_line);"
        )

    # ---- 直近高値/安値 JS ----
    recent_hl_series = recent_hl_init = recent_hl_update = ""
    if has_recent_hl:
        recent_hl_series = """
let recentHLLines = [];
function setRecentHL(levels) {
  recentHLLines.forEach(l => cSeries.removePriceLine(l));
  recentHLLines = [];
  if (!levels) return;
  const hlColors = {H:'#ff8a65',L:'#80deea'};
  levels.forEach(lvl => {
    const isHigh = lvl.label.startsWith('H');
    recentHLLines.push(cSeries.createPriceLine({
      price: lvl.price,
      color: isHigh ? '#ff8a6599' : '#80deea99',
      lineWidth: 1, lineStyle: 1,
      axisLabelVisible: true,
      title: lvl.label,
    }));
  });
}"""
        recent_hl_init   = "\nsetRecentHL(init.indicators.Recent_HL);"
        recent_hl_update = "\n    if (d.indicators?.Recent_HL) setRecentHL(d.indicators.Recent_HL);"

    # ---- ピボットポイント JS ----
    pivot_series = pivot_init = pivot_update = ""
    if has_pivot:
        pivot_series = """
let pivotLines = [];
function setPivotLines(levels) {
  pivotLines.forEach(l => cSeries.removePriceLine(l));
  pivotLines = [];
  if (!levels) return;
  const pColors = {
    'P':'#e0e0e0',
    'R1':'#ef9a9a','R2':'#ef5350','R3':'#b71c1c',
    'S1':'#80cbc4','S2':'#26a69a','S3':'#00695c'
  };
  const pStyles = {'P':0,'R1':2,'R2':1,'R3':0,'S1':2,'S2':1,'S3':0};
  levels.forEach(lvl => {
    pivotLines.push(cSeries.createPriceLine({
      price: lvl.price,
      color: pColors[lvl.label] || '#888888',
      lineWidth: 1,
      lineStyle: pStyles[lvl.label] !== undefined ? pStyles[lvl.label] : 2,
      axisLabelVisible: true,
      title: lvl.label,
    }));
  });
}"""
        pivot_init   = "\nsetPivotLines(init.indicators.Pivot_lines);"
        pivot_update = "\n    if (d.indicators?.Pivot_lines) setPivotLines(d.indicators.Pivot_lines);"

    # ---- シグナルライン JS ----
    signal_lines_js = """
let _slLine = null, _tpLine = null, _entryLine = null;
function setSignalLines(lines) {
  if (_slLine)    { cSeries.removePriceLine(_slLine);    _slLine = null; }
  if (_tpLine)    { cSeries.removePriceLine(_tpLine);    _tpLine = null; }
  if (_entryLine) { cSeries.removePriceLine(_entryLine); _entryLine = null; }
  if (!lines || !lines.direction || lines.direction === 'neutral') return;
  const isLong = lines.direction === 'long';
  if (lines.entry) {
    _entryLine = cSeries.createPriceLine({
      price: lines.entry, color: '#ffeb3bcc', lineWidth: 1, lineStyle: 2,
      axisLabelVisible: true, title: isLong ? '\\u25b2 ENTRY' : '\\u25bc ENTRY',
    });
  }
  if (lines.sl) {
    _slLine = cSeries.createPriceLine({
      price: lines.sl, color: '#ef5350cc', lineWidth: 2, lineStyle: 0,
      axisLabelVisible: true, title: ('SL ' + (lines.sl_pips||0).toFixed(1) + 'p'),
    });
  }
  if (lines.tp) {
    _tpLine = cSeries.createPriceLine({
      price: lines.tp, color: '#26a69acc', lineWidth: 2, lineStyle: 0,
      axisLabelVisible: true, title: ('TP ' + (lines.tp_pips||0).toFixed(1) + 'p'),
    });
  }
}"""
    signal_lines_init   = "\nif (init.signal_lines) setSignalLines(init.signal_lines);"
    signal_lines_update = "\n    if (d.signal_lines !== undefined) setSignalLines(d.signal_lines);"

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#0e1117;overflow:hidden;}}
  #chart{{width:100%;}}
</style>
</head>
<body>
<div id="chart"></div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  width: window.innerWidth, height: {height},
  layout: {{ background:{{type:'solid',color:'#0e1117'}}, textColor:'#f0f2f6', fontSize:11 }},
  grid:   {{ vertLines:{{color:'#1f2937'}}, horzLines:{{color:'#1f2937'}} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor:'#374151' }},
  timeScale: {{
    borderColor:'#374151', timeVisible:true, secondsVisible:false,
    tickMarkFormatter: (time, type) => {{
      const d=new Date(time*1000);
      const mo=String(d.getUTCMonth()+1).padStart(2,'0');
      const day=String(d.getUTCDate()).padStart(2,'0');
      const h=String(d.getUTCHours()).padStart(2,'0');
      const mi=String(d.getUTCMinutes()).padStart(2,'0');
      if(type===0) return String(d.getUTCFullYear());
      if(type===1) return mo+'月';
      if(type===2) return day+'日';
      return h+':'+mi;
    }},
  }},
  localization: {{
    timeFormatter: (time) => {{
      const d=new Date(time*1000);
      return d.getUTCFullYear()+'/'+String(d.getUTCMonth()+1).padStart(2,'0')+'/'+
             String(d.getUTCDate()).padStart(2,'0')+' '+
             String(d.getUTCHours()).padStart(2,'0')+':'+
             String(d.getUTCMinutes()).padStart(2,'0')+' JST';
    }},
  }},
}});

const cSeries = chart.addCandlestickSeries({{
  upColor:'#26a69a', downColor:'#ef5350',
  borderUpColor:'#26a69a', borderDownColor:'#ef5350',
  wickUpColor:'#26a69a', wickDownColor:'#ef5350',
  priceFormat: {{ type:'price', precision:{precision}, minMove:{min_move} }},
}});
chart.priceScale('right').applyOptions({{ scaleMargins:{{ top:{cm[0]}, bottom:{cm[1]} }} }});
{overlay_series}
{zigzag_series}
{rsi_series}
{stoch_series}
{macd_series}
{cvd_series}
{sr_series}
{recent_hl_series}
{pivot_series}
{signal_lines_js}

const init = {initial};
cSeries.setData(init.candles);
if (init.events?.length) cSeries.setMarkers(init.events);
{overlay_init}{zigzag_init}{rsi_init}{stoch_init}{macd_init}{cvd_init}{sr_init}{recent_hl_init}{pivot_init}{signal_lines_init}

window.addEventListener('resize', () => {{ chart.applyOptions({{ width:window.innerWidth }}); }});

// ---- 通知サウンド ----
let _notifLastTs = 0;
function playNotifSound(type) {{
  try {{
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function beep(freq, start, dur, vol) {{
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(vol, ctx.currentTime + start);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
      osc.start(ctx.currentTime + start);
      osc.stop(ctx.currentTime + start + dur + 0.05);
    }}
    if (type === 'entry_long') {{
      // 上昇2音: 低→高
      beep(660, 0.00, 0.12, 0.35);
      beep(880, 0.14, 0.18, 0.35);
    }} else if (type === 'entry_short') {{
      // 下降2音: 高→低
      beep(880, 0.00, 0.12, 0.35);
      beep(550, 0.14, 0.18, 0.35);
    }} else {{
      // エグジット: 短い単音
      beep(440, 0.00, 0.20, 0.25);
    }}
  }} catch(e) {{}}
}}

setInterval(async () => {{
  try {{
    const r = await fetch('{data_url}?_='+Date.now());
    if (!r.ok) return;
    const d = await r.json();
    if (!d.candle) return;
    cSeries.update(d.candle);
    if (d.events?.length) cSeries.setMarkers(d.events);{overlay_update}{zigzag_update}{rsi_update}{stoch_update}{macd_update}{cvd_update}{sr_update}{recent_hl_update}{pivot_update}{signal_lines_update}
    if (d.notification?.ts && d.notification.ts !== _notifLastTs) {{
      _notifLastTs = d.notification.ts;
      playNotifSound(d.notification.type);
    }}
  }} catch(e) {{}}
}}, 100);
</script>
</body>
</html>"""


# ============================================================
# 損益曲線（Plotly）
# ============================================================

def build_equity_curve(equity_df: pd.DataFrame) -> go.Figure:
    equity   = equity_df["equity"]
    positive = equity >= 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity.where(positive),
                             fill="tozeroy", fillcolor="rgba(38,166,154,0.2)",
                             line=dict(color=COLOR_BULL, width=1.5), name="利益"))
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity.where(~positive),
                             fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
                             line=dict(color=COLOR_BEAR, width=1.5), name="損失"))
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
    fig.update_layout(
        title="損益曲線（累積損益）", template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=60, r=20, t=50, b=20), height=350,
        legend=dict(orientation="h", y=1.02), yaxis_title="累積損益（円）",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937")
    return fig
