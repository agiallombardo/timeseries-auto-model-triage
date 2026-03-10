# Time Series Auto Model Triage
Trying to manually figure out the best prediction model and hyperparameters can take hours per dataset. With it being the same process each time, why not automate it?

This framework automatically tests and evaluates multiple forecasting models on time series data, including statistical, machine learning, and deep learning approaches. It provides comprehensive visualizations and model selection capabilities.

For the given time series data and the corresponding hyperparameter tuning results, the top 3 performing models will be selected by default.

The framework is also extensible by defining your own models within the corresponding `src/models` or `src/tuning` modules.

## Author
Anthony Giallombardo & Assistant Claude

## Features
- **15 forecasting models** — statistical, ML, and deep learning
- Automatic hyperparameter tuning for top-performing models
- Comprehensive visualization of results
- Detailed performance metrics and comparisons
- Dataset-specific result organization
- Optional GPU acceleration (Metal on Apple Silicon), .env config, and parallel runs

## Requirements
- **Python 3.9+**
- CSV (or Excel) input with a **time/date column** and a **numeric column** to forecast

## Installation
```bash
git clone <repo-url>
cd timeseries-auto-model-triage
pip install -r requirements.txt
```

On **macOS (Apple Silicon)**, `tensorflow-metal` is installed automatically for GPU-accelerated deep learning. Optional: `pip install -r requirements-mac-gpu.txt` if you installed base requirements on another OS and then run on a Mac.

For **Jupyter** support: `pip install jupyter ipykernel` (or use the optional deps in `requirements.txt`).

## Input data
- **Format:** CSV (or Excel). You must have one **time/date column** and one **numeric column** to forecast.
- **Arguments:** `--file` (path), `--time_col` (name of date column), `--data_col` (name of value column). Optional: `--date_format` if parsing fails (e.g. `%Y-%m-%d`).
- You can set `DATA_FILE`, `TIME_COL`, and `DATA_COL` in a `.env` file and then run `python main.py` with no arguments.

## Sample data
Sample datasets are in `data/samples/`:
- **Retail sales** — daily, seasonal and holiday effects
- **Energy consumption** — hourly, multiple seasonality

Generate them:
```bash
python data/samples/generate_retail_sales.py
python data/samples/generate_energy_consumption.py
```

## Basic usage
Run all models on the retail sales sample:
```bash
python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales
```

By default this runs **ML and DL models only** (10 models: rf, svr, xgb, lr, rnn, lstm, mlp, lstm_feat, rnn_feat, cnn1d), 3 variations each, tunes the top 3, and saves under `results/retail_sales_daily/`. Use `--models all` to include statistical models (arima, sarima, ma, es, prophet).

## Example scenarios

**Retail sales (daily):**
```bash
python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales
```
ML models (e.g. `rf`, `xgb`) often do well here due to seasonal and holiday effects.

**Energy consumption (hourly):**
```bash
python main.py --file data/samples/energy_consumption_hourly.csv --time_col timestamp --data_col energy_consumption
```
High-frequency data with multiple seasonality; LSTM and Prophet are often competitive.

**Run only selected models:** `--models rf xgb prophet mlp`  
**Tune all models:** `--tune_all`  
**Tune top 5 instead of 3:** `--tune_top 5`


## Understanding results

Results are written under `results/{dataset_name}/` (or `--output_dir`):

| Output | Description |
|--------|-------------|
| `model_comparison_results.csv` | One row per model, sorted by composite score (RMSE, MAE, R², etc.). |
| `model_configs.json` | Best variation config per model (hyperparameters, normalization, lags). |
| `results_summary.json` | Dataset info, best-judgment text, full results, optional best_per_run and tuned params. |
| `all_models_trellis.png` | Grid of forecasts for top 9 models (skipped with `--minimal-output` or `--no-charts`). |
| `top_3_models_comparison.png` | Comparison of the top 3 models. |
| `model_performance.png` | Bar charts of metrics. |
| `model_radar.png` | Radar chart (skipped with `--minimal-output`). |
| `top_3_residuals.png` | Residuals for top 3 (skipped with `--minimal-output`). |
| `feature_importance.png` | Feature importance when available. |

## Supported Models

Use the `--models` flag with the keys below (e.g. `--models rf xgb lstm`). **Default: ML and DL only** (10 models). Use `--models all` to run all 15 including statistical (arima, sarima, ma, es, prophet).

| Key | Display name | Family |
|-----|--------------|--------|
| `arima` | ARIMA | Statistical |
| `sarima` | SARIMA (Seasonal ARIMA) | Statistical |
| `ma` | Moving Average | Statistical |
| `es` | Exponential Smoothing | Statistical |
| `prophet` | Prophet | Statistical |
| `rf` | Random Forest | ML |
| `svr` | SVR (Support Vector Regression) | ML |
| `xgb` | XGBoost | ML |
| `lr` | Linear Regression | ML |
| `rnn` | RNN (Recurrent Neural Network) | DL |
| `lstm` | LSTM (Long Short-Term Memory) | DL |
| `mlp` | MLP (Multi-Layer Perceptron) | DL |
| `lstm_feat` | LSTM-feat (LSTM on lag/features) | DL |
| `rnn_feat` | RNN-feat (RNN on lag/features) | DL |
| `cnn1d` | CNN-1D (1D Convolutional) | DL |

**Statistical:** ARIMA, SARIMA, Moving Average, Exponential Smoothing, Prophet — good baselines and for seasonal series.

**ML:** Random Forest, SVR, XGBoost, Linear Regression — support multiple loss functions (L1, L2, Huber, quantile) and lag-based features.

**DL:** RNN, LSTM, MLP, LSTM-feat, RNN-feat, CNN-1D — TensorFlow/Keras; GPU-accelerated on Apple Silicon with `tensorflow-metal`.


## Performance and GPU (MacBook Pro)

**Apple Silicon (M1/M2/M3/M4):** The Metal GPU plugin (`tensorflow-metal`) is included by default when you install requirements on macOS (`pip install -r requirements.txt`). TensorFlow will use Metal for Keras models automatically; no extra steps or code changes are required.

**Intel Macs:** TensorFlow on macOS does not use the Intel iGPU for training. Use CPU or consider a cloud GPU for heavy DL workloads.

**Troubleshooting: "Library not loaded: … _pywrap_tensorflow_internal.so" (Metal plugin)**  
This usually means `tensorflow-metal` is incompatible with your TensorFlow version. To run without GPU and avoid the crash:

```bash
pip uninstall tensorflow-metal
```

Then reinstall a [tensorflow-metal version](https://pypi.org/project/tensorflow-metal/) that matches your TensorFlow version, or keep it uninstalled to use CPU-only for DL models. You can still run **ML-only** models without TensorFlow by using e.g. `--models rf svr xgb lr`.

**TensorFlow threading (CPU or GPU):** You can tune how many threads TensorFlow uses for intra-op and inter-op parallelism via environment variables (e.g. before running `main.py`):

- `TF_NUM_INTRAOP_THREADS` – threads used within an op (default: auto).
- `TF_NUM_INTEROP_THREADS` – threads used across ops (default: auto).

Example (use 4 and 2 threads):

```bash
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales
```

These can also be set in a `.env` file (see [Configuration via .env](#configuration-via-env)).

## Configuration via .env

You can set default options in a `.env` file in the project root so you don’t have to pass them on the CLI every time. Options in `.env` override code defaults; CLI arguments override `.env`.

Create a `.env` file (it is gitignored). Example:

```env
# Data (optional; if set, you can run: python main.py)
DATA_FILE=data/samples/retail_sales_daily.csv
TIME_COL=date
DATA_COL=sales

# Performance
N_RUNS=1
MODELS=rf,xgb,mlp
JOBS=1
TUNE_TOP=3
TUNE_ALL=false
OUTPUT_DIR=results

# Optional: fast tuning (fewer grid points / CV splits)
# TUNING_FAST=false
# TUNING_N_SPLITS=3

# Optional: deep learning (epochs / early stopping patience)
# DL_EPOCHS_GRID=50
# DL_EPOCHS_REFIT=200
# DL_PATIENCE=10

# Optional: minimal output (skip extra charts)
# MINIMAL_OUTPUT=false
# NO_CHARTS=false

# Optional: TensorFlow threading (see Performance and GPU above)
# TF_NUM_INTRAOP_THREADS=
# TF_NUM_INTEROP_THREADS=
```

With `DATA_FILE`, `TIME_COL`, and `DATA_COL` set, you can run:

```bash
python main.py
```

Or override for a single run: `python main.py --file other.csv --time_col ts --data_col value`.

## Advanced configuration

| Flag | Description |
|------|-------------|
| `--models` | Models to run. Default: ML + DL only (`rf`, `svr`, `xgb`, `lr`, `rnn`, `lstm`, `mlp`, `lstm_feat`, `rnn_feat`, `cnn1d`). Use `all` to include statistical (`arima`, `sarima`, `ma`, `es`, `prophet`). |
| `--n-runs` | Number of program runs to aggregate (default 3). Use `1` for faster iteration. |
| `--tune_top` | Number of top models to tune (default 3). Use `0` to disable tuning. |
| `--tune_all` | Tune every model (time-consuming). |
| `--jobs` | Run up to N (model × variation) tasks in parallel (default 1). Use `2` on MacBook with Metal to reduce GPU contention. |
| `--output_dir` | Directory for results (default `results`). |
| `--minimal-output` | Skip trellis/radar/residuals; keep CSV, config, and main charts. |
| `--no-charts` | Skip all chart generation. |
| `--losses` | Restrict loss variants: `l1`, `l2`, `huber`, `quantile` (for models that support them). |

These can also be set in `.env` (e.g. `N_RUNS=1`, `MODELS=rf,xgb,mlp`, `JOBS=2`).