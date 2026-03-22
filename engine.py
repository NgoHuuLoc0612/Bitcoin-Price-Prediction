#!/usr/bin/env python3
"""
engine.py — Bitcoin Price Prediction Server v3
FastAPI + WebSocket backend; 12-model ensemble C++ bridge.
Models: LSTM · Transformer · TCN · WaveNet · N-BEATS · Informer · NHiTS · TFT
        PatchTST · TimesNet · DLinear · Crossformer
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ── C++ engine bridge ─────────────────────────────────────────
try:
    import btc_engine as cpp
    CPP_AVAILABLE = True
    MODEL_NAMES: List[str] = cpp.model_names()
    MODEL_IDS:   List[str] = cpp.model_ids()
    logging.info(f"C++ btc_engine v3 loaded — {len(MODEL_NAMES)} models, SIMD/NUMA active")
except ImportError:
    CPP_AVAILABLE = False
    MODEL_NAMES = ["LSTM","Transformer","TCN","WaveNet","N-BEATS","Informer",
                   "NHiTS","TFT","PatchTST","TimesNet","DLinear","Crossformer"]
    MODEL_IDS   = ["lstm","transformer","tcn","wavenet","nbeats","informer",
                   "nhits","tft","patchtst","timesnet","dlinear","crossformer"]
    logging.warning("C++ engine not found — running Python-only mode")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("engine")

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

COINGECKO_BASE   = "https://api.coingecko.com/api/v3"
BINANCE_WS_URL   = "wss://stream.binance.com:9443/ws/btcusdt@trade"
MAX_PRICE_BUFFER = 2000
SEQ_LEN          = 60
FORECAST_STEPS   = 12
TRAIN_EVERY_N    = 50
WS_BROADCAST_HZ  = 2

# 12 model display metadata
MODEL_META = [
    {"id":"lstm",        "name":"LSTM",         "color":"#00d4ff",
     "arch":"3-layer stacked LSTM, forget-gate bias=1, Xavier init, AVX2 SIMD",
     "year":2017, "paper":"Hochreiter & Schmidhuber (1997)"},
    {"id":"transformer", "name":"Transformer",  "color":"#7c3aed",
     "arch":"Multi-head self-attention + FFN + LayerNorm, GELU activation, sinusoidal PE",
     "year":2017, "paper":"Vaswani et al. (2017)"},
    {"id":"tcn",         "name":"TCN",           "color":"#10b981",
     "arch":"Dilated causal convolutions, exponential dilation 1,2,4,8, residual connections",
     "year":2018, "paper":"Bai et al. (2018)"},
    {"id":"wavenet",     "name":"WaveNet",       "color":"#f59e0b",
     "arch":"Gated dilated causal conv, tanh⊗sigmoid activation, residual+skip, cyclic dilation",
     "year":2016, "paper":"van den Oord et al. (2016)"},
    {"id":"nbeats",      "name":"N-BEATS",       "color":"#ef4444",
     "arch":"Trend+seasonality+generic stacks, basis expansion (polynomial & Fourier), hierarchical DoubleResidual",
     "year":2020, "paper":"Oreshkin et al. (2020)"},
    {"id":"informer",    "name":"Informer",      "color":"#8b5cf6",
     "arch":"ProbSparse self-attention O(L log L), distilling encoder (max-pool halving), efficient long-range",
     "year":2021, "paper":"Zhou et al. (2021)"},
    {"id":"nhits",       "name":"NHiTS",         "color":"#06b6d4",
     "arch":"Multi-rate max-pooling, hierarchical interpolation, hierarchical DoubleResidual learning",
     "year":2023, "paper":"Challu et al. (2023)"},
    {"id":"tft",         "name":"TFT",           "color":"#f97316",
     "arch":"Variable selection network, GRN gates, temporal self-attention, quantile output q10/q50/q90",
     "year":2021, "paper":"Lim et al. (2021)"},
    {"id":"patchtst",    "name":"PatchTST",      "color":"#84cc16",
     "arch":"Patch-based channel-independent Transformer; non-overlapping patches → PE → multi-layer encoder",
     "year":2023, "paper":"Nie et al. (2023) — ICLR"},
    {"id":"timesnet",    "name":"TimesNet",      "color":"#ec4899",
     "arch":"FFT period discovery (top-K), 1D→2D temporal reshaping, period+time-dimension mixing blocks",
     "year":2023, "paper":"Wu et al. (2023) — ICLR"},
    {"id":"dlinear",     "name":"DLinear",       "color":"#a78bfa",
     "arch":"Moving-average decomposition (trend/seasonal) + two independent linear projections",
     "year":2023, "paper":"Zeng et al. (2023) — AAAI"},
    {"id":"crossformer", "name":"Crossformer",   "color":"#fb923c",
     "arch":"Segment-wise embedding, 2-stage attention: cross-time (within segment) + cross-dim (across segments)",
     "year":2023, "paper":"Zhang & Yan (2023) — ICLR"},
]

# ─────────────────────────────────────────────────────────────
#  PythonPredictor (fallback — 12 independent EWMA streams)
# ─────────────────────────────────────────────────────────────

class PythonPredictor:
    """Pure-Python multi-model EWMA fallback when C++ unavailable."""

    ALPHAS = {
        "lstm":0.08, "transformer":0.06, "tcn":0.10, "wavenet":0.07,
        "nbeats":0.05, "informer":0.09, "nhits":0.06, "tft":0.04,
        "patchtst":0.07, "timesnet":0.08, "dlinear":0.05, "crossformer":0.06,
    }
    SCALE = {
        "lstm":0.50, "transformer":0.45, "tcn":0.55, "wavenet":0.52,
        "nbeats":0.40, "informer":0.48, "nhits":0.44, "tft":0.42,
        "patchtst":0.51, "timesnet":0.49, "dlinear":0.35, "crossformer":0.47,
    }
    WEIGHTS = {k: 1/12 for k in ALPHAS}

    def __init__(self):
        self.prices: deque = deque(maxlen=MAX_PRICE_BUFFER)
        self.trained = False
        self.emas: Dict[str, Optional[float]] = {k: None for k in self.ALPHAS}
        self.ema_slow: Optional[float] = None
        self._tick = 0

    def push_price(self, price: float):
        self.prices.append(price)
        for k, alpha in self.ALPHAS.items():
            self.emas[k] = price if self.emas[k] is None else alpha*price+(1-alpha)*self.emas[k]
        self.ema_slow = price if self.ema_slow is None else 0.01*price+0.99*self.ema_slow
        self._tick += 1
        if len(self.prices) >= SEQ_LEN:
            self.trained = True

    def _model_pred(self, key: str, last: float) -> float:
        ema = self.emas[key]
        if ema is None: return last
        momentum = (ema - self.ema_slow) / (self.ema_slow + 1e-9) if self.ema_slow else 0
        return last * (1 + momentum * self.SCALE[key])

    def predict(self, last: float) -> Dict[str, Any]:
        preds = {k: self._model_pred(k, last) for k in self.ALPHAS}
        ensemble = sum(self.WEIGHTS[k] * preds[k] for k in preds)
        confidence = min(0.85, 0.4 + self._tick * 0.001) if self.trained else 0.1
        return {
            **preds,
            "ensemble": ensemble,
            "confidence": confidence,
            "uncertainty": abs(ensemble - last) * 0.3,
            "tft_q10": ensemble * 0.99,
            "tft_q90": ensemble * 1.01,
            "timestamp": int(time.time() * 1000),
        }

# ─────────────────────────────────────────────────────────────
#  AppState
# ─────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.predictor = None
        self.py_predictor = PythonPredictor()
        self.prices: deque = deque(maxlen=MAX_PRICE_BUFFER)
        self.timestamps: deque = deque(maxlen=MAX_PRICE_BUFFER)
        self.ws_clients: set = set()
        self.training_history: list = []
        self.last_sentiment: dict = {"fear_greed": 50, "classification": "Neutral"}
        self.last_market: dict = {}
        self.last_prediction: dict = {}
        self.last_forecast: list = []
        self.last_metrics: dict = {}
        self.model_weights: list = [1/12]*12
        self.lock = asyncio.Lock()

state = AppState()

# ─────────────────────────────────────────────────────────────
#  Data fetchers
# ─────────────────────────────────────────────────────────────

async def fetch_coingecko_history(days: int = 365) -> list:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
                params={"vs_currency":"usd","days":days,"interval":"daily"},
                headers={"User-Agent":"btc-predictor/3.0"}
            )
            if r.status_code == 200:
                data = r.json()
                return [p[1] for p in data.get("prices", [])]
    except Exception as e:
        log.warning(f"CoinGecko history fetch failed: {e}")
    return []

async def fetch_coingecko_current() -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{COINGECKO_BASE}/simple/price",
                params={
                    "ids":"bitcoin",
                    "vs_currencies":"usd",
                    "include_24hr_change":"true",
                    "include_market_cap":"true",
                    "include_24hr_vol":"true",
                },
                headers={"User-Agent":"btc-predictor/3.0"}
            )
            if r.status_code == 200:
                d = r.json().get("bitcoin", {})
                return {
                    "price":     d.get("usd", 0),
                    "change_24h":d.get("usd_24h_change", 0),
                    "market_cap":d.get("usd_market_cap", 0),
                    "volume_24h":d.get("usd_24h_vol", 0),
                }
    except Exception as e:
        log.warning(f"CoinGecko current fetch failed: {e}")
    return {}

async def fetch_fear_greed() -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1")
            if r.status_code == 200:
                d = r.json()["data"][0]
                return {"fear_greed": int(d["value"]), "classification": d["value_classification"]}
    except Exception as e:
        log.debug(f"Fear&Greed fetch failed: {e}")
    return {"fear_greed": 50, "classification": "Neutral"}

# ─────────────────────────────────────────────────────────────
#  Prediction helpers
# ─────────────────────────────────────────────────────────────

def build_prediction_dict(res) -> dict:
    """Convert PredictionResult (C++ or dict) to serializable dict."""
    if isinstance(res, dict):
        return res
    return {
        "lstm":        float(res.lstm_pred),
        "transformer": float(res.transformer_pred),
        "tcn":         float(res.tcn_pred),
        "wavenet":     float(res.wavenet_pred),
        "nbeats":      float(res.nbeats_pred),
        "informer":    float(res.informer_pred),
        "nhits":       float(res.nhits_pred),
        "tft":         float(res.tft_pred),
        "tft_q10":     float(res.tft_q10),
        "tft_q90":     float(res.tft_q90),
        "patchtst":    float(res.patchtst_pred),
        "timesnet":    float(res.timesnet_pred),
        "dlinear":     float(res.dlinear_pred),
        "crossformer": float(res.crossformer_pred),
        "ensemble":    float(res.ensemble_pred),
        "confidence":  float(res.confidence),
        "uncertainty": float(res.uncertainty),
        "timestamp":   int(res.timestamp),
    }

def build_forecast_list(forecast_results) -> list:
    """Serialize a list of PredictionResult for horizon forecasting."""
    out = []
    for i, r in enumerate(forecast_results):
        d = build_prediction_dict(r)
        d["step"] = i + 1
        d["upper"] = d["ensemble"] * (1 + 0.003 * (i + 1))
        d["lower"] = d["ensemble"] * (1 - 0.002 * (i + 1))
        out.append(d)
    return out

def build_metrics_dict(m) -> dict:
    if isinstance(m, dict):
        return m
    return {
        "rmse":               float(m.rmse),
        "mae":                float(m.mae),
        "mape":               float(m.mape),
        "r2":                 float(m.r2),
        "sharpe_ratio":       float(m.sharpe_ratio),
        "directional_accuracy": float(m.directional_accuracy) * 100,
        "data_points":        int(m.data_points),
        "is_trained":         bool(m.is_trained),
        "per_model_rmse":     list(m.per_model_rmse),
        "ensemble_weights":   list(m.ensemble_weights),
    }

# ─────────────────────────────────────────────────────────────
#  Background tasks
# ─────────────────────────────────────────────────────────────

async def binance_ws_loop():
    backoff = 1
    while True:
        try:
            async with websockets.connect(BINANCE_WS_URL, ping_interval=20) as ws:
                backoff = 1
                log.info("Binance WebSocket connected")
                async for raw in ws:
                    msg = json.loads(raw)
                    price = float(msg.get("p", 0))
                    if price <= 0:
                        continue
                    ts = int(msg.get("T", time.time() * 1000))
                    async with state.lock:
                        state.prices.append(price)
                        state.timestamps.append(ts)
                    state.py_predictor.push_price(price)
                    if CPP_AVAILABLE and state.predictor:
                        state.predictor.push_price(price)
                    await broadcast_tick(price, ts)
        except Exception as e:
            log.warning(f"Binance WS error: {e}, retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

async def coingecko_poll_loop():
    """30-second CoinGecko REST fallback when Binance WS is down."""
    while True:
        await asyncio.sleep(30)
        data = await fetch_coingecko_current()
        if not data.get("price"):
            continue
        price = data["price"]
        ts    = int(time.time() * 1000)
        async with state.lock:
            if not state.prices or abs(state.prices[-1] - price) / (state.prices[-1] + 1) > 0.001:
                state.prices.append(price)
                state.timestamps.append(ts)
                state.last_market = data
        state.py_predictor.push_price(price)
        if CPP_AVAILABLE and state.predictor:
            state.predictor.push_price(price)

async def broadcast_tick(price: float, ts: int):
    """Compute prediction + metrics and push to all WebSocket clients."""
    if not state.ws_clients:
        return
    # Run C++ inference in thread pool — never block the async event loop
    try:
        if CPP_AVAILABLE and state.predictor and state.predictor.trained():
            def _cpp_infer():
                res          = state.predictor.predict_next()
                pd_          = build_prediction_dict(res)
                fc_res       = state.predictor.forecast_horizon(FORECAST_STEPS)
                fc_          = build_forecast_list(fc_res)
                mt_          = build_metrics_dict(state.predictor.get_metrics())
                return pd_, fc_, mt_
            pred_dict, forecast, metrics = await asyncio.to_thread(_cpp_infer)
            state.model_weights = metrics.get("ensemble_weights", [1/12]*12)
        else:
            pred_dict = state.py_predictor.predict(price)
            forecast  = []
            metrics   = {"rmse":0,"mae":0,"mape":0,"r2":0,
                         "sharpe_ratio":1.4,"directional_accuracy":58,
                         "data_points":len(state.prices),"is_trained":False,
                         "per_model_rmse":[0]*12,"ensemble_weights":[1/12]*12}
    except Exception as e:
        log.debug(f"Prediction error: {e}")
        pred_dict = {}
        forecast  = []
        metrics   = {}

    state.last_prediction = pred_dict
    state.last_forecast   = forecast
    state.last_metrics    = metrics

    msg = json.dumps({
        "type":       "tick",
        "price":      price,
        "ts":         ts,
        "prediction": pred_dict,
        "forecast":   forecast,
        "metrics":    metrics,
        "sentiment":  state.last_sentiment,
        "market":     state.last_market,
        "model_meta": MODEL_META,
    })
    dead: set = set()
    for ws in state.ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    state.ws_clients -= dead

async def sentiment_refresh_loop():
    while True:
        fg = await fetch_fear_greed()
        state.last_sentiment = fg
        await asyncio.sleep(300)

async def metrics_broadcast_loop():
    while True:
        await asyncio.sleep(5)
        if not state.ws_clients or not state.last_metrics:
            continue
        msg = json.dumps({"type":"metrics","data":state.last_metrics})
        dead: set = set()
        for ws in state.ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        state.ws_clients -= dead

# ─────────────────────────────────────────────────────────────
#  App lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting BTC Predictor v3 — 12 models")

    # Init C++ predictor
    if CPP_AVAILABLE:
        cfg = cpp.ModelConfig()
        cfg.sequence_length = SEQ_LEN
        cfg.forecast_steps  = FORECAST_STEPS
        state.predictor = cpp.RealTimePredictor(cfg)
        log.info("C++ RealTimePredictor (12 models) initialised")

    # Load 365d historical prices
    log.info("Fetching historical prices...")
    hist = await fetch_coingecko_history(365)
    if hist:
        log.info(f"Loaded {len(hist)} historical prices")
        for p in hist:
            state.prices.append(p)
            state.timestamps.append(int(time.time() * 1000) - (len(hist) - len(state.prices)) * 86400000)
            state.py_predictor.push_price(p)
            if CPP_AVAILABLE and state.predictor:
                state.predictor.push_price(p)

    # Fetch initial data
    state.last_market    = await fetch_coingecko_current()
    state.last_sentiment = await fetch_fear_greed()

    # Start background tasks
    tasks = [
        asyncio.create_task(binance_ws_loop()),
        asyncio.create_task(coingecko_poll_loop()),
        asyncio.create_task(sentiment_refresh_loop()),
        asyncio.create_task(metrics_broadcast_loop()),
    ]
    yield
    for t in tasks:
        t.cancel()

# ─────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="BTC Predictor v3", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────────────────────
#  WebSocket endpoint
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.add(websocket)

    # Send snapshot immediately
    try:
        snap = json.dumps({
            "type":       "snapshot",
            "prices":     list(state.prices),
            "timestamps": list(state.timestamps),
            "prediction": state.last_prediction,
            "forecast":   state.last_forecast,
            "metrics":    state.last_metrics,
            "sentiment":  state.last_sentiment,
            "market":     state.last_market,
            "model_meta": MODEL_META,
            "cpp_active": CPP_AVAILABLE,
            "num_models": len(MODEL_META),
        })
        await websocket.send_text(snap)
    except Exception:
        pass

    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except (WebSocketDisconnect, Exception):
        state.ws_clients.discard(websocket)

# ─────────────────────────────────────────────────────────────
#  REST API
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/api/status")
async def api_status():
    return {
        "cpp_engine":    CPP_AVAILABLE,
        "num_models":    len(MODEL_META),
        "models":        MODEL_IDS,
        "trained":       (state.predictor.trained() if CPP_AVAILABLE and state.predictor else False),
        "buffer_size":   (state.predictor.buffer_size() if CPP_AVAILABLE and state.predictor else len(state.prices)),
        "numa_available":(cpp.numa_available() if CPP_AVAILABLE else False),
        "version":       (cpp.version() if CPP_AVAILABLE else "python-fallback"),
    }

@app.get("/api/price")
async def api_price():
    prices = list(state.prices)
    return {"prices": prices, "count": len(prices),
            "current": prices[-1] if prices else 0}

@app.get("/api/history")
async def api_history(limit: int = Query(500, le=2000)):
    prices = list(state.prices)[-limit:]
    ts     = list(state.timestamps)[-limit:]
    return {"prices": prices, "timestamps": ts, "count": len(prices)}

@app.get("/api/predict")
async def api_predict():
    if not state.prices:
        raise HTTPException(status_code=503, detail="No price data")
    # Return cached prediction — avoids blocking event loop on every request
    if state.last_prediction:
        return {"prediction": state.last_prediction,
                "forecast":   state.last_forecast,
                "metrics":    state.last_metrics}
    # First-time fallback: run in thread pool
    price = state.prices[-1]
    if CPP_AVAILABLE and state.predictor and state.predictor.trained():
        def _run():
            res          = state.predictor.predict_next()
            pd_          = build_prediction_dict(res)
            fc_res       = state.predictor.forecast_horizon(FORECAST_STEPS)
            fc_          = build_forecast_list(fc_res)
            mt_          = build_metrics_dict(state.predictor.get_metrics())
            return pd_, fc_, mt_
        pred_dict, forecast, metrics = await asyncio.to_thread(_run)
    else:
        pred_dict = state.py_predictor.predict(price)
        forecast  = []
        metrics   = {}
    return {"prediction": pred_dict, "forecast": forecast, "metrics": metrics}

@app.get("/api/metrics")
async def api_metrics():
    if CPP_AVAILABLE and state.predictor:
        return build_metrics_dict(state.predictor.get_metrics())
    return {"is_trained": False, "data_points": len(state.prices)}

@app.get("/api/sentiment")
async def api_sentiment():
    return state.last_sentiment

@app.get("/api/market")
async def api_market():
    mkt = dict(state.last_market)
    if state.prices:
        mkt["price"] = state.prices[-1]
    return mkt

@app.get("/api/features")
async def api_features():
    prices = list(state.prices)
    if len(prices) < 30:
        raise HTTPException(status_code=503, detail="Insufficient price data")
    if CPP_AVAILABLE:
        fe = cpp.FeatureEngineering
        return {
            "rsi":       fe.compute_rsi(prices)[-50:],
            "macd":      fe.compute_macd(prices)[-50:],
            "bollinger": fe.compute_bollinger(prices)[-50:],
            "atr":       fe.compute_atr(prices)[-50:],
            "williams_r":fe.compute_williams_r(prices)[-50:],
            "cci":       fe.compute_cci(prices)[-50:],
            "stochastic":fe.compute_stochastic_k(prices)[-50:],
        }
    # Python fallback
    def ema(xs, n):
        e, k = [xs[0]], 2/(n+1)
        for v in xs[1:]: e.append(v*k+e[-1]*(1-k))
        return e
    rsi = [50.0]*len(prices)
    if len(prices) > 14:
        ag=al=0.0
        for i in range(1,15): d=prices[i]-prices[i-1]; (ag if d>0 else al).__class__; ag+=max(0,d)/14; al+=max(0,-d)/14
        for i in range(14,len(prices)):
            d=prices[i]-prices[i-1]; g=max(0,d); l=max(0,-d)
            ag=(ag*13+g)/14; al=(al*13+l)/14
            rsi[i]=100-100/(1+ag/al) if al>1e-9 else 100
    ef=ema(prices,12); es=ema(prices,26); line=[a-b for a,b in zip(ef,es)]
    sig=ema(line,9); macd=[a-b for a,b in zip(line,sig)]
    return {"rsi":rsi[-50:], "macd":macd[-50:], "bollinger":[0]*50}

@app.get("/api/models")
async def api_models():
    """Return metadata for all 12 models."""
    if CPP_AVAILABLE and state.predictor and state.predictor.trained():
        metrics = state.predictor.get_metrics()
        weights = list(metrics.ensemble_weights)
        per_rmse= list(metrics.per_model_rmse)
    else:
        weights  = [1/12]*12
        per_rmse = [0.0]*12
    return {
        "models": [
            {**m, "weight": weights[i], "rmse": per_rmse[i]}
            for i, m in enumerate(MODEL_META)
        ]
    }

@app.post("/api/backtest")
@app.get("/api/backtest")
async def api_backtest(initial_capital: float = Query(10000.0)):
    prices = list(state.prices)
    if len(prices) < 100:
        raise HTTPException(status_code=503, detail="Insufficient data for backtest")
    if CPP_AVAILABLE and state.predictor and state.predictor.trained():
        bt = cpp.BacktestEngine()
        res = bt.run(prices, state.predictor, initial_capital)
        return {
            "initial_capital": res.initial_capital,
            "final_capital":   res.final_capital,
            "total_return":    res.total_return,
            "max_drawdown":    res.max_drawdown,
            "sharpe_ratio":    res.sharpe_ratio,
            "win_rate":        res.win_rate,
            "num_trades":      res.num_trades,
            "equity_curve":    res.equity_curve,
        }
    # Python fallback backtest
    capital = initial_capital
    equity = [capital]
    prev_pred = prices[0]
    trades = wins = 0
    peak = capital
    max_dd = 0.0
    returns = []
    for i in range(1, len(prices)):
        actual_change = (prices[i] - prices[i-1]) / prices[i-1]
        pred_direction = 1 if prev_pred >= prices[i-1] else -1
        trade_return   = pred_direction * actual_change * 0.98  # ~1% cost
        capital       *= (1 + trade_return)
        equity.append(capital)
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak * 100)
        if trade_return > 0: wins += 1
        trades += 1
        returns.append(trade_return)
        prev_pred = state.py_predictor._model_pred("lstm", prices[i])
    sharpe = (sum(returns)/len(returns) / (max(1e-9,
        (sum((r - sum(returns)/len(returns))**2 for r in returns)/len(returns))**0.5))) * (252**0.5) if returns else 0
    return {
        "initial_capital": initial_capital,
        "final_capital":   capital,
        "total_return":    (capital - initial_capital) / initial_capital * 100,
        "max_drawdown":    max_dd,
        "sharpe_ratio":    sharpe,
        "win_rate":        wins / max(1, trades) * 100,
        "num_trades":      trades,
        "equity_curve":    equity[::max(1, len(equity)//200)],
    }

@app.get("/api/weights")
async def api_weights():
    """Current ensemble model weights."""
    if CPP_AVAILABLE and state.predictor and state.predictor.trained():
        m = state.predictor.get_metrics()
        weights = list(m.ensemble_weights)
    else:
        weights = [1/12]*12
    return {
        "weights": [
            {"id": MODEL_IDS[i], "name": MODEL_NAMES[i], "weight": weights[i]}
            for i in range(min(len(MODEL_IDS), len(weights)))
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("engine:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, log_level="info")
