# ₿ Bitcoin Price Prediction Engine v3

A C++/Python ML pipeline that runs 12 time-series models in parallel, streams live BTC price from Binance, and serves predictions through a FastAPI/WebSocket backend with a 3D WebGL frontend.

The C++ engine is compiled as a pybind11 extension (`.pyd` / `.so`) and handles all inference. Python handles the HTTP/WebSocket layer and data plumbing. No frameworks like PyTorch or TensorFlow — every model is implemented from scratch in C++ with AVX2 SIMD and OpenBLAS.

---

## Models

| # | Model | Architecture | Paper |
|---|-------|-------------|-------|
| 1 | **LSTM** | 3-layer stacked LSTM, forget-gate bias=1, Xavier init, AVX2 dot product | Hochreiter & Schmidhuber (1997) |
| 2 | **Transformer** | Multi-head self-attention + FFN + LayerNorm, GELU, sinusoidal PE | Vaswani et al. (2017) |
| 3 | **TCN** | Dilated causal conv, exponential dilation 1→2→4→8, residual connections | Bai et al. (2018) |
| 4 | **WaveNet** | Gated dilated causal conv (tanh⊗sigmoid), residual+skip, cyclic dilation | van den Oord et al. (2016) |
| 5 | **N-BEATS** | Trend+seasonality+generic stacks, polynomial+Fourier basis expansion | Oreshkin et al. (2020) |
| 6 | **Informer** | ProbSparse self-attention O(L log L), distilling encoder with max-pool | Zhou et al. (2021) |
| 7 | **NHiTS** | Multi-rate max-pooling, hierarchical interpolation, DoubleResidual | Challu et al. (2023) |
| 8 | **TFT** | Variable selection GRN, temporal self-attention, quantile q10/q50/q90 | Lim et al. (2021) |
| 9 | **PatchTST** | Patch-based channel-independent Transformer, patch_len=16 stride=8 | Nie et al., ICLR 2023 |
| 10 | **TimesNet** | FFT top-5 period discovery → 1D→2D reshape → periodic+temporal mixing | Wu et al., ICLR 2023 |
| 11 | **DLinear** | Moving-avg decomposition (k=25) + two independent linear projections | Zeng et al., AAAI 2023 |
| 12 | **Crossformer** | Segment embedding (seg=6), cross-time + cross-dim two-stage attention | Zhang & Yan, ICLR 2023 |

Ensemble weights are updated live using inverse-error weighting — models that have been more accurate recently get higher weight.

---

## Stack

```
btc_predictor/
├── core.cpp          # 12-model ML engine — all inference logic
├── core.hpp
├── optimization.cpp  # Adam, LBFGS, RMSProp, NUMA allocator, thread pool
├── optimization.hpp
├── bindings.cpp      # pybind11 bindings
├── external.cpp      # CoinGecko REST, Binance WebSocket, order book, sentiment
├── external.hpp
├── engine.py         # FastAPI server + WebSocket broadcast
├── index.html        # Frontend — 3D WebGL, charts, model panel
├── CMakeLists.txt
└── requirements.txt
```

---

## Build

### Linux (Ubuntu/Debian)

```bash
sudo apt install cmake build-essential libopenblas-dev libcurl4-openssl-dev \
                 libnuma-dev python3-dev pybind11-dev

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cp build/btc_engine*.so .
```

### Windows (MSYS2 MinGW64)

```bash
# Install dependencies
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc \
          mingw-w64-x86_64-openblas mingw-w64-x86_64-curl \
          mingw-w64-x86_64-python mingw-w64-x86_64-pybind11

cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE="C:/Users/.../Python311/python.exe"
cmake --build build -j4
copy build\btc_engine*.pyd .
```

After building, copy the required MinGW runtime DLLs into the project folder so Python can find them:

```
libstdc++-6.dll
libgcc_s_seh-1.dll
libwinpthread-1.dll
libgomp-1.dll
libopenblas.dll
libgfortran-5.dll
libquadmath-0.dll
libbtc_core.dll  ← from build/
```

---

## Run

```bash
pip install -r requirements.txt
python engine.py
# → http://localhost:8000
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend |
| WS | `/ws` | Live price feed + all 12 predictions |
| GET | `/api/predict` | Latest prediction from all 12 models |
| GET | `/api/models` | Model metadata + current weights |
| GET | `/api/weights` | Ensemble weights |
| GET | `/api/history` | Historical price buffer |
| GET | `/api/metrics` | RMSE / MAE / R² / Sharpe per model |
| GET | `/api/sentiment` | Fear & Greed index |
| GET | `/api/market` | CoinGecko market data |
| GET | `/api/features` | RSI, MACD, Bollinger, ATR, Williams %R |
| POST | `/api/backtest` | Walk-forward backtest |
| GET | `/api/status` | Engine status, SIMD/NUMA info |

---

## Frontend

- 3D WebGL scene — Three.js rotating BTC crystal with 12 model particles in orbit
- Live prediction panel — per-model values, color-coded, toggleable
- Ensemble weight bar chart — updates as models converge
- Radar chart — model comparison across speed, accuracy, trend, seasonality dimensions
- Architecture browser — collapsible tech details and paper citations per model
- 12-step forecast horizon with confidence intervals
- Technical indicators — RSI, MACD histogram, Bollinger %B
- Backtest equity curve — walk-forward PnL
- Fear & Greed gauge with animated needle
- Training log — live epoch/loss stream over WebSocket
- On-chain metrics — hash rate, SOPR, NVT ratio, mempool size

---

## Notes

- The C++ engine has no BLAS dependency if `cblas.h` is not found at build time — it falls back to an OpenMP-parallelised naive GEMM. Performance is lower but the build still works.
- All C++ inference runs in a thread pool (`asyncio.to_thread`) so the async event loop is never blocked.
- `api/predict` returns the cached result from the background prediction loop rather than triggering a fresh inference on every request.
