# Time Series Auto Model Triage
Trying to manually figure out the best prediction model and hyperparameters can take hours per dataset. With it being the same process each time, why not automate it?

This framework automatically tests and evaluates multiple forecasting models on time series data, including statistical, machine learning, and deep learning approaches. It provides comprehensive visualizations and model selection capabilities.

For the given time series data and the corresponding hyperparameter tuning results, the top 3 performing models will be selected by default.

The framework is also extensible by defining your own models within the corresponding models or tuning .py   
  
## Author
Anthony Giallombardo & Assistant Claude

## Features
 - Supports 10 different forecasting models 
 - Automatic hyperparameter tuning for top-performing models 
 - Comprehensive visualization of results 
 - Detailed performance metrics and comparisons 
 - Dataset-specific result organization

## Sample Data

**The package includes sample datasets in the data/samples directory:** \
***Retail Sales Data*** - Daily retail sales with seasonal patterns, trends, and holiday effects \
***Energy Consumption Data*** - Hourly energy usage with multiple seasonal patterns and temperature effects

**You can generate these sample datasets using:** \
`python data/samples/generate_retail_sales.py` \
`python data/samples/generate_energy_consumption.py`

**Basic Usage** \
Run the forecaster with default settings on the retail sales sample: \
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales` 

**This will**
- Run all available forecasting models
- Automatically tune the top 3 performing models
- Generate visualizations and performance metrics
- Save results in results/retail_sales_daily/

## Example Scenarios
**Forecasting Retail Sales** \
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales` 

For this dataset, machine learning models like Random Forest and XGBoost typically outperform statistical models because they can capture complex patterns, including seasonal effects and holiday periods. 

**Forecasting Energy Consumption** \
`python main.py --file data/samples/energy_consumption_hourly.csv --time_col timestamp --data_col energy_consumption` 

This high-frequency data with multiple seasonality patterns (daily, weekly, yearly) benefits from models like LSTM and Prophet that can capture these complex temporal dependencies. 

**Forecasting with Selections**

***To test only specific models:*** \
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --models rf xgb prophet` 

***Tune all models (can be time-consuming):*** \
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --tune_all` 

***Tune only the top 5 models (default is 3):*** \
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --tune_top 5` 

  
## Understanding Results

Each dataset's results are saved in a dedicated directory (results/{dataset_name}/):
- all_models_trellis.png - Grid of plots showing each model's performance
- top_3_models_comparison.png - Detailed comparison of the best models
- model_performance.png - Bar charts comparing metrics across models
- model_comparison_results.csv - Tabular performance metrics (one row per model, sorted by composite score)
- model_configs.json - One config per model (best variation), same order as results
- results_summary.json - Dataset, best_judgment, results (all models in order with metrics and params), optional best_per_run and tuned_best_params

## Supported Models

***Statistical Models***
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)
- Moving Average
- Exponential Smoothing

***Machine Learning Models***
- Random Forest
- SVR (Support Vector Regression)
- XGBoost

***Deep Learning Models***
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)

***Other Models***
- Facebook Prophet


## Performance and GPU (MacBook Pro)

**Apple Silicon (M1/M2/M3/M4):** For faster deep learning training, install the optional Metal plugin so TensorFlow uses the GPU:

```bash
pip install tensorflow-metal
```

Or install from the project’s optional requirements (includes `tensorflow-metal`):

```bash
pip install -r requirements-mac-gpu.txt
```

TensorFlow will then use Metal for Keras models by default; no code changes are required.

**Intel Macs:** TensorFlow on macOS does not use the Intel iGPU for training. Use CPU or consider a cloud GPU for heavy DL workloads.

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

## Advanced Configuration

Run a subset of models, control runs and tuning, or enable parallel model runs:

- `--models rf xgb mlp` – run only these models.
- `--n-runs 1` – single sweep (faster; default from config is 3).
- `--tune_top 5` – tune top 5 models (default 3).
- `--tune_all` – tune all models (time-consuming).
- `--jobs 2` – run up to 2 (model × variation) tasks in parallel (default 1).
- `--output_dir results` – where to save results.
- `--minimal-output` – skip non-essential charts; keep CSV and config.
- `--no-charts` – skip all chart generation.