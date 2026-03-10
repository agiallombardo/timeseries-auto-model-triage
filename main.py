#!/usr/bin/env python3
import argparse
import os
import re
import json
import logging
import time
import traceback
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from dotenv import load_dotenv

from src.utils import setup_logging
from src.config import resolve_data_args, resolve_run_args, apply_tuning_setup_from_env
from src.device import init_tensorflow_device_and_threading
from src.data_handling import load_data, prepare_features, split_data
from src.evaluation import (
    evaluate_model, plot_results, create_trellis_plot,
    create_top_models_plot, create_performance_chart, compute_best_judgment,
    create_radar_chart, create_residuals_plot, plot_feature_importance,
)
from src.report import generate_html_report
from src.models.registry import get_available_models, get_tuning_functions
from src.losses import LOSS_SUPPORTED_MODELS, LOSS_KEYS
from src.model_config import (
    get_default_setup,
    get_variations_for_model,
    get_metadata_for_model,
    build_normalization_entry,
)

# Results reporting: column order and numeric format (same for CSV, console, and summary)
REPORT_COLUMNS = ['model', 'composite_score', 'rmse', 'mae', 'r2', 'mase', 'mape']
REPORT_FORMAT = {
    'composite_score': '{:.4f}',
    'rmse': '{:.4f}',
    'mae': '{:.4f}',
    'r2': '{:.4f}',
    'mase': '{:.3f}',
    'mape': '{:.2f}',
}

# Map display name first word to registry key (for "tune top N" selection)
DISPLAY_NAME_TO_REGISTRY_KEY = {
    'moving': 'ma',
    'exponential': 'es',
    'random': 'rf',
    'xgboost': 'xgb',
    'lstm-feat': 'lstm_feat',
    'rnn-feat': 'rnn_feat',
    'cnn-1d': 'cnn1d',
    'linear': 'lr',
}

# Model key -> family for console and judgment
MODEL_FAMILY = {
    'arima': 'Statistical', 'sarima': 'Statistical', 'ma': 'Statistical',
    'es': 'Statistical', 'prophet': 'Statistical',
    'rf': 'ML', 'svr': 'ML', 'xgb': 'ML', 'lr': 'ML',
    'rnn': 'DL', 'lstm': 'DL', 'mlp': 'DL', 'lstm_feat': 'DL', 'rnn_feat': 'DL', 'cnn1d': 'DL',
}


def _format_results_for_display(df):
    """Return a copy of the results DataFrame with numeric columns formatted for consistent console/CSV display."""
    cols = [c for c in REPORT_COLUMNS if c in df.columns]
    out = df[cols].copy()
    for c in REPORT_FORMAT:
        if c not in out.columns:
            continue
        fmt = REPORT_FORMAT[c]
        out[c] = out[c].apply(lambda x: fmt.format(x) if pd.notna(x) and not np.isinf(x) else "")
    return out


def _model_name_to_family_map(results_df_agg):
    """Build dict model display name -> family (Statistical, ML, DL) from aggregated results."""
    out = {}
    for _, row in results_df_agg.iterrows():
        name = row['model']
        model_key, _ = _parse_display_name(name)
        out[name] = MODEL_FAMILY.get(model_key, "Other")
    return out


def _print_results_report(results_df_agg, top_models_df, judgment_text):
    """Print a consistent results report: table, best judgment, best-by-family, and top 3 models."""
    cols = [c for c in REPORT_COLUMNS if c in results_df_agg.columns]
    if not cols:
        return
    # Table with consistent formatting
    disp = _format_results_for_display(results_df_agg)
    sep = "=" * 100
    print("\n" + sep)
    print("RESULTS (median across variations, sorted by composite score; test metrics = final evaluation only)")
    print(sep)
    print(disp.to_string(index=False))
    print("-" * 100)
    print("Best:", judgment_text)
    print(sep)
    # Best by family
    family_best = {}
    name_to_family = _model_name_to_family_map(results_df_agg)
    for _, row in results_df_agg.iterrows():
        name = row['model']
        fam = name_to_family.get(name, "Other")
        if fam not in family_best:
            family_best[fam] = name
    if family_best:
        parts = [f"  {fam} → {name}" for fam, name in sorted(family_best.items())]
        print("\nBest by family: " + "  |  ".join(parts))
    # Top 3
    print("\nTop 3 models:")
    for i, (_, row) in enumerate(top_models_df.iterrows(), 1):
        parts = [f"{i}. {row['model']}", f"composite={row['composite_score']:.4f}", f"RMSE={row['rmse']:.4f}", f"MAE={row['mae']:.4f}", f"R²={row['r2']:.4f}"]
        if 'mase' in row and pd.notna(row.get('mase')) and not np.isinf(row.get('mase', 0)):
            parts.append(f"MASE={row['mase']:.3f}")
        if 'mape' in row and pd.notna(row.get('mape')) and not np.isinf(row.get('mape', 0)):
            parts.append(f"MAPE={row['mape']:.2f}%")
        print("  " + " | ".join(parts))


def _parse_display_name(display_name):
    """Parse display name back to (model_key, loss_key|None).

    Handles all display name formats:
      'XGBoost (L2)'                       -> ('xgb', 'l2')
      'ARIMA(1, 0, 0)'                     -> ('arima', None)
      'SARIMA[1,1,0]x[0,1,1,12] (Tuned)'  -> ('sarima', None)
      'SVR (C=1.0, rbf)'                   -> ('svr', None)
      'Random Forest (L2) (n=200)'         -> ('rf', 'l2')
      'Linear Regression (L1) (α=0.1)'     -> ('lr', 'l1')
      'CNN-1D (Huber)'                     -> ('cnn1d', 'huber')
    """
    loss_map = {'l1': 'l1', 'l2': 'l2', 'huber': 'huber', 'quantile': 'quantile'}

    # Isolate model base name by removing order/args that appear as inline (digits) or [list] right after
    base = re.sub(r'\([\d, ]+\).*$', '', display_name).strip()  # ARIMA(1,0,0)... → ARIMA
    base = re.sub(r'\[.*$', '', base).strip()                    # SARIMA[...]... → SARIMA
    base = base.split(' (')[0].strip()                           # strip loss/suffix like " (L2)"

    first_word = base.lower().split()[0] if base else display_name.lower().split()[0]
    model_key = DISPLAY_NAME_TO_REGISTRY_KEY.get(first_word, first_word)

    # Scan parenthesised segments for a recognised loss key
    loss = None
    for segment in display_name.split('(')[1:]:
        candidate = segment.strip().rstrip(')').strip().lower()
        if candidate in loss_map:
            loss = loss_map[candidate]
            break
    return model_key, loss


def _aggregate_run_results(run_results, model_name):
    """Aggregate n_runs evaluation dicts into one with mean metrics and rmse_std for reranking."""
    if not run_results:
        return {"model": model_name, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mse": np.nan, "mase": np.nan, "mape": np.nan, "rmse_std": np.nan}
    keys = ["mse", "rmse", "mae", "r2", "mase", "mape"]
    out = {"model": model_name}
    for k in keys:
        if k not in run_results[0]:
            continue
        vals = [r[k] for r in run_results if k in r]
        if not vals:
            continue
        out[k] = float(np.nanmean(vals))
    rmse_vals = [r["rmse"] for r in run_results if "rmse" in r]
    out["rmse_std"] = float(np.nanstd(rmse_vals)) if len(rmse_vals) > 1 else 0.0
    return out


def _dataset_results_dir(args):
    """Results path: output_dir / dataset_name / date (YYYY-MM-DD)."""
    dataset_name = os.path.basename(args.file).split(".")[0]
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(args.output_dir, dataset_name, date_str)


def _print_phase(n: int, total: int, message: str) -> None:
    """Print a phase banner to the console."""
    print(f"\n[{n}/{total}] {message}", flush=True)


def _get_losses_for_model(model_key, requested_losses):
    """Return the list of loss keys to run for this model. requested_losses is from args.losses (None = all)."""
    if model_key not in LOSS_SUPPORTED_MODELS:
        return [None]
    supported = LOSS_SUPPORTED_MODELS[model_key]
    if requested_losses is None:
        return supported
    return [l for l in supported if l in requested_losses]


def _build_run_config(
    args, default_setup, model_key, variation_index, variation_spec,
    display_name, tuned, hyperparameters, metadata, n_train, n_test,
):
    """Build a single run config dict for model_configs.json."""
    effective_lags = variation_spec.get("lags", default_setup["lags"]) if variation_spec else default_setup["lags"]
    setup = {
        "data_file": args.file,
        "time_column": args.time_col,
        "data_column": args.data_col,
        "test_size": default_setup["test_size"],
        "lags": effective_lags,
        "lags_built": default_setup["lags"],
        "rolling_window": default_setup["rolling_window"],
        "ma_window": default_setup["ma_window"],
        "n_runs": default_setup.get("n_runs", 3),
        "total_samples": n_train + n_test,
        "training_samples": n_train,
        "testing_samples": n_test,
    }
    time_steps = None
    if metadata and metadata.get("uses_n_steps"):
        n_steps = hyperparameters.get("n_steps") if isinstance(hyperparameters, dict) else default_setup.get("n_steps_univariate") or default_setup.get("n_steps_feature")
        if n_steps is None:
            n_steps = metadata.get("default_n_steps")
        time_steps = {"n_steps": n_steps}
    normalization = build_normalization_entry(metadata, hyperparameters)
    shape_and_reshaping = (metadata.get("shape_and_reshaping") if metadata else None) or None
    config = {
        "model_key": model_key,
        "variation_index": variation_index,
        "variation_spec": variation_spec,
        "display_name": display_name,
        "tuned": tuned,
        "setup": setup,
        "time_steps": time_steps,
        "normalization": normalization,
        "shape_and_reshaping": shape_and_reshaping,
        "hyperparameters": hyperparameters if isinstance(hyperparameters, dict) else {},
    }
    if tuned:
        config["selection_metric"] = "rmse"
        config["hyperparameters_note"] = "Validation-selected (held-out test not used for parameter selection)."
    return config


# Module-level logger
logger = logging.getLogger(__name__)

def main():

    """Main function to run the time series forecasting comparison."""
    load_dotenv(os.environ.get("TRIAGE_ENV", ".env"))

    parser = argparse.ArgumentParser(
        description='Time Series Forecasting Model Comparison. Each model runs 3 variations (different params) to find the best.'
    )
    parser.add_argument('--file', required=False, default=None, help='Path to the data file (or set DATA_FILE in .env)')
    parser.add_argument('--time_col', required=False, default=None, help='Name of the time/date column (or set TIME_COL in .env)')
    parser.add_argument('--data_col', required=False, default=None, help='Name of the data column to forecast (or set DATA_COL in .env)')
    parser.add_argument('--date_format', default=None, help='Format of the date string (or set DATE_FORMAT in .env)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to run (or set MODELS=rf,xgb,mlp in .env). Default: all.')
    parser.add_argument('--losses', nargs='+', default=None,
                        help='Loss variants to run (default: all). Choices: l1, l2, huber, quantile. Set LOSSES in .env for comma-separated list.')
    parser.add_argument('--tune_top', type=int, default=None,
                        help='Number of top models to tune (default: 3, or TUNE_TOP in .env). Set to 0 to disable.')
    parser.add_argument('--tune_all', action='store_true',
                        help='Tune all models instead of just the top performing ones (or TUNE_ALL=true in .env).')
    parser.add_argument('--output_dir', default=None, help='Directory to save results (or OUTPUT_DIR in .env). Default: results.')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Parallel jobs for (model × variation) runs (default: 1, or JOBS in .env).')
    parser.add_argument('--n-runs', type=int, default=None,
                        help='Number of program runs to aggregate (default from config/.env, usually 3). Use 1 for faster iteration.')
    parser.add_argument('--minimal-output', action='store_true',
                        help='Skip non-essential charts; keep CSV and config (or MINIMAL_OUTPUT=true in .env).')
    parser.add_argument('--no-charts', action='store_true',
                        help='Skip all chart generation (or NO_CHARTS=true in .env).')

    args = parser.parse_args()

    resolve_data_args(args)
    if args.file is None or args.time_col is None or args.data_col is None:
        parser.error("Data source required: provide --file, --time_col, --data_col or set DATA_FILE, TIME_COL, DATA_COL in .env")

    default_setup = get_default_setup()
    resolve_run_args(args, default_setup)
    apply_tuning_setup_from_env()

    # Setup logging
    logger = setup_logging(args.output_dir)

    # TensorFlow device + threading init — only when DL models will actually run
    _DL_MODELS = {"rnn", "lstm", "mlp", "lstm_feat", "rnn_feat", "cnn1d"}
    _ML_ONLY_MODELS = {"rf", "svr", "xgb", "lr"}
    _active_models = set(args.models) if args.models and args.models != ["all"] else _DL_MODELS
    _requests_dl = bool(_active_models & _DL_MODELS or "all" in (args.models or []))
    if _requests_dl:
        tf_ready = init_tensorflow_device_and_threading()
        if not tf_ready:
            logger.warning(
                "TensorFlow/Metal initialization failed. "
                "Skipping DL models and continuing with non-DL models."
            )
            if args.models and "all" in args.models:
                args.models = sorted(_ML_ONLY_MODELS)
            else:
                args.models = [m for m in (args.models or []) if m not in _DL_MODELS]
            if not args.models:
                logger.error(
                    "No runnable models remain after disabling DL models. "
                    "Install compatible TensorFlow packages or run with non-DL models."
                )
                return

    # Validate --losses if provided; keep only valid keys and warn once
    if args.losses is not None:
        invalid = [l for l in args.losses if l not in LOSS_KEYS]
        if invalid:
            logger.warning(
                "Ignoring invalid --losses: %s. Valid choices: %s", invalid, LOSS_KEYS
            )
        args.losses = [l for l in args.losses if l in LOSS_KEYS] if args.losses else None

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Suppress noisy convergence and parameter warnings from statsmodels, sklearn, and Keras
    warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
    warnings.filterwarnings("ignore", message=".*Non-stationary starting autoregressive parameters.*")
    warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*")
    warnings.filterwarnings("ignore", message=".*Objective did not converge.*", module="sklearn.linear_model")
    warnings.filterwarnings("ignore", message=".*conditioned on metric.*val_loss.*")

    run_start_time = time.time()

    _print_phase(1, 4, f"Loading data from {args.file}…")
    logger.info(f"Loading data from {args.file}")
    logger.info(f"Time column: {args.time_col}")
    logger.info(f"Data column: {args.data_col}")

    logger.info(f"Setup: test_size={default_setup['test_size']}, lags_options={default_setup.get('lags_options')} (built with max={default_setup['lags']}, subsetted per variation), n_steps={default_setup.get('n_steps_univariate')}, ma_window={default_setup['ma_window']} (3 variations per model)")

    # Load and prepare data (use default setup, not CLI)
    try:
        df = load_data(args.file, args.time_col, args.data_col, args.date_format)
        logger.info(f"Data loaded successfully with {len(df)} records")

        df_features = prepare_features(
            df, args.data_col,
            lags=default_setup["lags"],
            rolling_window=default_setup["rolling_window"],
        )
        logger.info(f"Features prepared with {len(df_features)} records after handling lags")
        if len(df_features) < 30:
            logger.warning(
                f"Only {len(df_features)} samples after feature preparation. "
                "Many models need more data for reliable results; consider using a longer series."
            )

        X_train, X_test, y_train, y_test, train_df, test_df = split_data(
            df_features, args.data_col, test_size=default_setup["test_size"]
        )
        logger.info(f"Data split into {len(train_df)} training and {len(test_df)} testing records")
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        return
    
    # Get available models and tuning functions from registry
    available_models = get_available_models()
    tuning_functions = get_tuning_functions()
    models_to_run_list = list(available_models.keys()) if "all" in args.models else [m.lower() for m in args.models]
    models_to_run_list = [m for m in models_to_run_list if m in available_models]
    n_models = len(models_to_run_list)
    _DL_KEYS = {"rnn", "lstm", "mlp", "lstm_feat", "rnn_feat", "cnn1d"}
    _dl_in_run = [m for m in models_to_run_list if m in _DL_KEYS]
    if _dl_in_run:
        logger.info("Models to run: %s (DL: %s)", models_to_run_list, _dl_in_run)
        print(f"Running {n_models} models (including DL: {', '.join(_dl_in_run)})", flush=True)
    else:
        logger.info("Models to run (ML/statistical only): %s", models_to_run_list)
        print(f"Running {n_models} models (ML/statistical only; no DL)", flush=True)
    n_runs = default_setup.get("n_runs", 3)
    if args.tune_all:
        _print_phase(2, 4, f"Tuning all models (3 variations each)…")
    else:
        total_configs = n_runs * n_models * 3
        _print_phase(2, 4, f"Sweeping {n_models} models × 3 variations × {n_runs} runs ({total_configs} configs)…")

    all_predictions, model_names, results, run_configs, best_per_run = run_models(
        args, default_setup, available_models, tuning_functions,
        X_train, X_test, y_train, y_test, train_df, test_df
    )
    
    # Create initial results DataFrame and compute composite scores for ranking
    results_df = pd.DataFrame(results)
    _, _, results_df = compute_best_judgment(results_df)

    # Determine which models to tune
    models_to_tune = []
    
    if args.tune_all:
        # Tune all models: each (model_key, loss) for loss-supported models
        models_to_tune = []
        for model_key in tuning_functions:
            for loss_key in _get_losses_for_model(model_key, args.losses):
                models_to_tune.append((model_key, loss_key))
    elif args.tune_top > 0:
        # Top N rows: parse each to (model_key, loss), skip if already tuned
        top_models = results_df.head(args.tune_top)
        seen = set()
        for _, row in top_models.iterrows():
            if 'tuned' in row['model'].lower():
                continue
            model_key, loss = _parse_display_name(row['model'])
            if model_key not in tuning_functions:
                continue
            key = (model_key, loss)
            if key not in seen:
                seen.add(key)
                models_to_tune.append(key)
    
    # Tune selected models (each entry is (model_key, loss) or (model_key, None))
    all_tuned_params = {}
    if models_to_tune:
        _print_phase(3, 4, f"Tuning {len(models_to_tune)} model(s)…")
        logger.info(f"Tuning {len(models_to_tune)} model/loss combination(s)...")
        tuned_predictions, tuned_names, tuned_results, all_tuned_params = tune_selected_models(
            args, default_setup, models_to_tune, tuning_functions,
            X_train, X_test, y_train, y_test, train_df, test_df
        )
        all_predictions.extend(tuned_predictions)
        model_names.extend(tuned_names)
        results.extend(tuned_results)

        for i, name in enumerate(tuned_names):
            if " (Tuned)" not in name:
                continue
            display_name = name.replace(" (Tuned)", "").strip()
            model_key, loss_key = _parse_display_name(display_name)
            metadata = get_metadata_for_model(model_key)
            params_suffix = f"{model_key}_{loss_key}" if loss_key else model_key
            best_params = all_tuned_params.get(params_suffix, {})
            if isinstance(best_params, int):
                best_params = {"window": best_params}
                run_configs.append(_build_run_config(
                args, default_setup, model_key, variation_index=0, variation_spec={},
                display_name=name, tuned=True, hyperparameters=best_params, metadata=metadata,
                n_train=len(y_train), n_test=len(y_test),
            ))
    _print_phase(4, 4, "Saving results and generating charts…")
    process_results(
        args, default_setup, all_predictions, model_names, results, run_configs, best_per_run,
        y_train, y_test, X_train, X_test, tuning_functions, tuned_best_params=all_tuned_params
    )
    elapsed = time.time() - run_start_time
    logger.info("Total run time: %.1f s (%.1f min)", elapsed, elapsed / 60.0)
    print(f"\nTotal run time: {elapsed:.1f} s ({elapsed / 60:.1f} min)", flush=True)

def run_models(args, default_setup, available_models, tuning_functions, X_train, X_test, y_train, y_test, train_df, test_df):
    """Run exactly 3 variations per model (built-in defaults), no tuning. Returns predictions, names, results, run_configs, best_per_run."""
    logger = logging.getLogger(__name__)

    # Suppress noisy warnings during model runs (statsmodels frequency, Keras input_shape)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No frequency information was provided")
        warnings.filterwarnings("ignore", message=".*input_shape.*input_dim.*", category=UserWarning)
        return _run_models_impl(args, default_setup, available_models, tuning_functions, X_train, X_test, y_train, y_test, train_df, test_df)


def _run_models_impl(args, default_setup, available_models, tuning_functions, X_train, X_test, y_train, y_test, train_df, test_df):
    """Run exactly 3 variations per model using built-in defaults (no CLI for variations)."""
    logger = logging.getLogger(__name__)

    models_to_run = list(available_models.keys()) if "all" in args.models else [m.lower() for m in args.models]
    seasonal_periods = determine_seasonal_periods(y_train)

    all_predictions = []
    model_names = []
    results = []
    run_configs = []

    # If tune_all: run tuning for each (model, variation) - same 3 variations per model
    if args.tune_all:
        logger.info("Performing hyperparameter tuning for all models (3 variations each)")
        tuning_tasks = []
        for model_key in models_to_run:
            if model_key not in tuning_functions:
                continue
            for variation_index, variation_spec in enumerate(get_variations_for_model(model_key)):
                tuning_tasks.append((model_key, variation_index, variation_spec))
        for model_key, variation_index, variation_spec in tqdm(
            tuning_tasks, desc="Tuning all", unit="config"
        ):
            try:
                loss_key = variation_spec.get("loss")
                logger.info(f"Tuning {model_key.upper()}" + (f" ({loss_key.upper()})" if loss_key else "") + " ...")
                var_lags = variation_spec.get("lags")
                var_X_train = _subset_lag_features(X_train, var_lags) if var_lags else X_train
                var_X_test = _subset_lag_features(X_test, var_lags) if var_lags else X_test
                model_params = _base_params(default_setup, model_key, y_train, y_test, var_X_train, var_X_test, seasonal_periods)
                model_params.update(variation_spec)
                pred, name, best_params = execute_tuning(model_key, tuning_functions, model_params)
                all_predictions.append(pred)
                model_names.append(name)
                results.append(evaluate_model(y_test, pred, name, y_train))
                metadata = get_metadata_for_model(model_key)
                hp = best_params if isinstance(best_params, dict) else {}
                run_configs.append(_build_run_config(
                    args, default_setup, model_key, variation_index, variation_spec,
                    name, tuned=True, hyperparameters=hp, metadata=metadata,
                    n_train=len(y_train), n_test=len(y_test),
                ))
            except Exception as e:
                logger.error(f"Error tuning {model_key.upper()}: {e}")
                logger.error(traceback.format_exc())
        return all_predictions, model_names, results, run_configs, []

    # Default: loop the entire program n_runs times, then aggregate and rerank by mean metrics
    n_runs = args.n_runs if getattr(args, "n_runs", None) is not None else default_setup.get("n_runs", 3)
    all_runs_preds = []
    all_runs_results = []
    first_run_names = []
    best_per_run = []
    run_config_args = {"file": args.file, "time_col": args.time_col, "data_col": args.data_col}

    if getattr(args, "jobs", 1) > 1:
        # Parallel: build task list and run with joblib (loky backend for fork safety)
        tasks = []
        for run_idx in range(n_runs):
            for model_key in models_to_run:
                if model_key not in available_models:
                    continue
                variations = get_variations_for_model(model_key)
                for variation_index, variation_spec in enumerate(variations):
                    tasks.append((
                        run_idx, model_key, variation_index, variation_spec, default_setup,
                        X_train, X_test, y_train, y_test, seasonal_periods, run_config_args,
                    ))
        logger.info("Running %d (model × variation) tasks with --jobs=%s", len(tasks), args.jobs)
        raw_results = []
        with tqdm(total=len(tasks), desc="Running", unit="task") as pbar:
            for i in range(0, len(tasks), args.jobs):
                chunk = tasks[i : i + args.jobs]
                chunk_results = Parallel(n_jobs=len(chunk), backend="loky")(
                    delayed(_run_one_model_variation)(t) for t in chunk
                )
                raw_results.extend(chunk_results)
                pbar.update(len(chunk))
                pbar.set_postfix_str(f"{pbar.n}/{len(tasks)}")
        # Sort by (run_idx, order of model_key in models_to_run, variation_index)
        model_order = {mk: i for i, mk in enumerate(models_to_run)}
        raw_results.sort(key=lambda r: (r[0], model_order.get(r[1], 0), r[2]))
        # Rebuild per-run lists (same shape as sequential path)
        run_preds_by_run = {}
        run_results_by_run = {}
        for run_idx, model_key, variation_index, pred, name, metrics, config in raw_results:
            run_preds_by_run.setdefault(run_idx, []).append(pred)
            run_results_by_run.setdefault(run_idx, []).append(metrics)
            if run_idx == 0 and config is not None:
                run_configs.append(config)
                first_run_names.append(name)
        for run_idx in range(n_runs):
            all_runs_preds.append(run_preds_by_run.get(run_idx, []))
            all_runs_results.append(run_results_by_run.get(run_idx, []))
        for run_idx in range(n_runs):
            run_results = all_runs_results[run_idx]
            if run_results and run_configs:
                run_df = pd.DataFrame(run_results)
                _, _, run_df_scored = compute_best_judgment(run_df)
                best_row = run_df_scored.iloc[0]
                best_idx = int(run_df_scored.index[0])
                cfg = run_configs[best_idx] if best_idx < len(run_configs) else {}
                best_per_run.append({
                    "run": run_idx + 1,
                    "best_model": best_row["model"],
                    "composite_score": round(float(best_row["composite_score"]), 4),
                    "rmse": float(best_row["rmse"]),
                    "mae": float(best_row["mae"]),
                    "r2": float(best_row["r2"]),
                    "variation_spec": cfg.get("variation_spec", {}),
                    "hyperparameters": cfg.get("hyperparameters", {}),
                })
                logger.info(f"Run {run_idx + 1} best: {best_row['model']} (composite={best_row['composite_score']:.4f})")
    else:
        # Sequential: one progress bar over all (run × model × variation) steps
        n_configs_per_run = sum(
            len(get_variations_for_model(m)) for m in models_to_run if m in available_models
        )
        total_steps = n_runs * n_configs_per_run
        with tqdm(total=total_steps, desc="Models", unit="config") as pbar:
            for run_idx in range(n_runs):
                logger.info(f"=== Program run {run_idx + 1}/{n_runs} ===")
                run_preds = []
                run_results = []
                for model_key in models_to_run:
                    if model_key not in available_models:
                        continue
                    t0_model = time.time()
                    variations = get_variations_for_model(model_key)
                    metadata = get_metadata_for_model(model_key)
                    default_hp = dict(metadata.get("default_hyperparameters", {})) if metadata else {}
                    for variation_index, variation_spec in enumerate(variations):
                        pbar.set_postfix_str(f"run {run_idx + 1}/{n_runs} {model_key}")
                        try:
                            var_lags = variation_spec.get("lags")
                            var_X_train = _subset_lag_features(X_train, var_lags) if var_lags else X_train
                            var_X_test = _subset_lag_features(X_test, var_lags) if var_lags else X_test
                            model_params = _base_params(default_setup, model_key, y_train, y_test, var_X_train, var_X_test, seasonal_periods)
                            model_params.update(variation_spec)
                            loss_key = variation_spec.get("loss")
                            logger.info(
                                f"Running {model_key.upper()}" + (f" ({loss_key.upper()})" if loss_key else "") + " ..."
                            )
                            t0 = time.time()
                            pred, name = execute_model(model_key, available_models, model_params)
                            metrics = evaluate_model(y_test, pred, name, y_train)
                            elapsed = time.time() - t0
                            run_preds.append(np.asarray(pred))
                            run_results.append(metrics)
                            logger.info(f"  var {variation_index + 1}/3 → RMSE={metrics['rmse']:.4f}  ({elapsed:.1f}s)")
                            if run_idx == 0:
                                first_run_names.append(name)
                                run_configs.append(_build_run_config(
                                    args, default_setup, model_key, variation_index, variation_spec,
                                    name, tuned=False, hyperparameters={**default_hp, **variation_spec}, metadata=metadata,
                                    n_train=len(y_train), n_test=len(y_test),
                                ))
                        except Exception as e:
                            logger.error(f"Error with {model_key.upper()} (variation {variation_index}): {e}")
                            logger.error(traceback.format_exc())
                        pbar.update(1)
                    logger.info(f"{model_key.upper()} completed in {time.time() - t0_model:.1f}s")
                all_runs_preds.append(run_preds)
                all_runs_results.append(run_results)

                # Best (model, variation) this run by composite score
                if run_results and run_configs:
                    run_df = pd.DataFrame(run_results)
                    _, _, run_df_scored = compute_best_judgment(run_df)
                    best_row = run_df_scored.iloc[0]
                    best_idx = int(run_df_scored.index[0])
                    cfg = run_configs[best_idx] if best_idx < len(run_configs) else {}
                    best_per_run.append({
                        "run": run_idx + 1,
                        "best_model": best_row["model"],
                        "composite_score": round(float(best_row["composite_score"]), 4),
                        "rmse": float(best_row["rmse"]),
                        "mae": float(best_row["mae"]),
                        "r2": float(best_row["r2"]),
                        "variation_spec": cfg.get("variation_spec", {}),
                        "hyperparameters": cfg.get("hyperparameters", {}),
                    })
                    logger.info(f"Run {run_idx + 1} best: {best_row['model']} (composite={best_row['composite_score']:.4f})")

    # Per-run composite scores for composite_std (same row order as all_runs_results[r])
    run_scored_dfs = []
    for r in range(n_runs):
        run_df = pd.DataFrame(all_runs_results[r])
        _, _, scored = compute_best_judgment(run_df)
        run_scored_dfs.append(scored)

    # Aggregate across program runs: mean metrics and mean predictions per model; rerank by mean metrics
    n_models = len(first_run_names)
    for i in range(n_models):
        name = first_run_names[i]
        result_dicts = [all_runs_results[r][i] for r in range(n_runs) if i < len(all_runs_results[r])]
        pred_arrays = [all_runs_preds[r][i] for r in range(n_runs) if i < len(all_runs_preds[r])]
        if not result_dicts or not pred_arrays:
            continue
        all_predictions.append(np.mean(pred_arrays, axis=0))
        model_names.append(name)
        result = _aggregate_run_results(result_dicts, name)
        # composite_std across runs (look up by model name in each run's scored df)
        composite_scores = []
        for r in range(n_runs):
            if r >= len(run_scored_dfs):
                break
            sr = run_scored_dfs[r]
            match = sr[sr['model'] == name]['composite_score']
            if len(match) > 0:
                composite_scores.append(float(match.iloc[0]))
        result['composite_std'] = float(np.nanstd(composite_scores)) if len(composite_scores) > 1 else 0.0
        results.append(result)

    return all_predictions, model_names, results, run_configs, best_per_run


def _subset_lag_features(X, lags: int):
    """Return feature matrix keeping only lag_1..lag_{lags} plus all non-lag columns.

    Features are built with max lags to keep y_train/y_test aligned across variations.
    Each variation that requests fewer lags gets a smaller but aligned feature matrix.
    """
    if X is None:
        return X
    lag_cols = sorted(
        [c for c in X.columns if c.startswith("lag_") and c.split("_")[1].isdigit()
         and int(c.split("_")[1]) <= lags],
        key=lambda c: int(c.split("_")[1]),
    )
    non_lag_cols = [c for c in X.columns
                    if not (c.startswith("lag_") and c.split("_")[1].isdigit())]
    return X[non_lag_cols + lag_cols]


def _base_params(default_setup, model_key, y_train, y_test, X_train, X_test, seasonal_periods):
    """Build base params for a run; n_steps and ma_window from default_setup."""
    n_steps = default_setup["n_steps_univariate"]
    if model_key in ("lstm_feat", "rnn_feat", "cnn1d"):
        n_steps = default_setup["n_steps_feature"]
    return {
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,
        "X_test": X_test,
        "n_steps": n_steps,
        "seasonal_periods": seasonal_periods,
        "ma_window": default_setup["ma_window"],
    }


def _run_one_model_variation(task):
    """Run a single (run_idx, model_key, variation) for parallel execution. Returns (run_idx, model_key, variation_index, pred, name, metrics, config_or_None)."""
    (run_idx, model_key, variation_index, variation_spec, default_setup,
     X_train, X_test, y_train, y_test, seasonal_periods, run_config_args) = task
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No frequency information was provided")
        warnings.filterwarnings("ignore", message=".*input_shape.*input_dim.*", category=UserWarning)
        var_X_train = _subset_lag_features(X_train, variation_spec["lags"]) if variation_spec.get("lags") else X_train
        var_X_test = _subset_lag_features(X_test, variation_spec["lags"]) if variation_spec.get("lags") else X_test
        model_params = _base_params(default_setup, model_key, y_train, y_test, var_X_train, var_X_test, seasonal_periods)
        model_params.update(variation_spec)
        available_models = get_available_models()
        pred, name = execute_model(model_key, available_models, model_params)
    metrics = evaluate_model(y_test, pred, name, y_train)
    config = None
    if run_idx == 0:
        metadata = get_metadata_for_model(model_key)
        default_hp = dict(metadata.get("default_hyperparameters", {})) if metadata else {}
        class _Args:
            pass
        args = _Args()
        args.file = run_config_args["file"]
        args.time_col = run_config_args["time_col"]
        args.data_col = run_config_args["data_col"]
        config = _build_run_config(
            args, default_setup, model_key, variation_index, variation_spec,
            name, tuned=False, hyperparameters={**default_hp, **variation_spec}, metadata=metadata,
            n_train=len(y_train), n_test=len(y_test),
        )
    return (run_idx, model_key, variation_index, np.asarray(pred), name, metrics, config)


def process_results(args, default_setup, all_predictions, model_names, results, run_configs, best_per_run, y_train, y_test, X_train, X_test, tuning_functions, tuned_best_params=None):
    """Process and display results. Saves model_configs.json and results_summary.json (single file, no redundancy)."""
    logger = logging.getLogger(__name__)

    if not all_predictions:
        logger.error("No models were successfully trained. Check the logs for errors.")
        return

    dataset_results_dir = _dataset_results_dir(args)
    os.makedirs(dataset_results_dir, exist_ok=True)

    # Create results DataFrame and add composite score (preserve row order for run_configs alignment)
    results_df = pd.DataFrame(results)
    results_df['_row_id'] = np.arange(len(results_df))
    best_row, judgment_text, scored_df = compute_best_judgment(results_df)
    results_df['composite_score'] = results_df['_row_id'].map(scored_df.set_index('_row_id')['composite_score'])
    results_df = results_df.drop(columns=['_row_id'])

    # Aggregate to one row per model: median metrics across variations; best variation = by test composite score (config saved is that variation's hyperparameters; when tuned, those are validation-selected)
    metric_cols = [c for c in ['composite_score', 'rmse', 'mae', 'r2', 'mase', 'mape'] if c in results_df.columns]
    agg_list = []
    for model_name, group in results_df.groupby('model', sort=False):
        medians = group[metric_cols].median()
        best_idx = int(group['composite_score'].idxmax())
        cfg = run_configs[best_idx] if best_idx < len(run_configs) else {}
        row = {'model': model_name, 'best_variation_index': best_idx}
        for c in metric_cols:
            row[c] = medians[c]
        if 'rmse_std' in group.columns:
            row['rmse_std'] = group['rmse_std'].median()
        if 'composite_std' in group.columns:
            row['composite_std'] = group['composite_std'].median()
        row['variation_spec'] = cfg.get('variation_spec', {})
        row['hyperparameters'] = cfg.get('hyperparameters', {})
        agg_list.append(row)
    results_df_agg = pd.DataFrame(agg_list).sort_values('composite_score', ascending=False).reset_index(drop=True)

    # Save model configs as JSON (same order as results; one per model, best variation first)
    if run_configs:
        ordered_configs = [run_configs[int(row["best_variation_index"])] for _, row in results_df_agg.iterrows() if int(row["best_variation_index"]) < len(run_configs)]
        configs_path = os.path.join(dataset_results_dir, "model_configs.json")
        with open(configs_path, "w") as f:
            json.dump(ordered_configs, f, indent=2)
        logger.info(f"Model configs saved to '{configs_path}'")

    # Save aggregated results (one row per model); column order matches report
    results_csv_path = os.path.join(dataset_results_dir, 'model_comparison_results.csv')
    csv_cols = [c for c in REPORT_COLUMNS if c in results_df_agg.columns]
    csv_df = results_df_agg[csv_cols].copy()
    csv_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to '{results_csv_path}'")

    # Top 3: use best variation's prediction per model
    top_models_df = results_df_agg.head(3)
    top_model_names = top_models_df['model'].tolist()
    top_model_indices = top_models_df['best_variation_index'].astype(int).tolist()
    valid_names = [model_names[i] for i in top_model_indices]
    top_model_predictions = [all_predictions[i] for i in top_model_indices]

    skip_charts = getattr(args, "no_charts", False)
    minimal = getattr(args, "minimal_output", False)

    if not skip_charts:
        # Trellis: top 9 models (skip in minimal-output)
        if not minimal:
            top_9_df = results_df_agg.head(9)
            top_9_indices = top_9_df['best_variation_index'].astype(int).tolist()
            top_9_predictions = [all_predictions[i] for i in top_9_indices]
            top_9_names = [model_names[i] for i in top_9_indices]
            create_trellis_plot(
                y_train, y_test, top_9_predictions, top_9_names, results_df_agg,
                os.path.join(dataset_results_dir, 'all_models_trellis.png'), max_models=9
            )

        if valid_names and top_model_predictions:
            create_top_models_plot(
                y_train, y_test, top_model_predictions, valid_names,
                os.path.join(dataset_results_dir, 'top_3_models_comparison.png')
            )

        create_performance_chart(
            results_df_agg,
            os.path.join(dataset_results_dir, 'model_performance.png')
        )
        if not minimal:
            create_radar_chart(
                results_df_agg,
                os.path.join(dataset_results_dir, 'model_radar.png'),
                max_models=5,
            )
            if valid_names and top_model_predictions:
                create_residuals_plot(
                    y_test, top_model_predictions, valid_names,
                    os.path.join(dataset_results_dir, 'top_3_residuals.png'),
                )
        plot_feature_importance(
            dataset_results_dir,
            os.path.join(dataset_results_dir, 'feature_importance.png'),
        )

    # Best model and judgment (with margin and family)
    model_name_to_family = _model_name_to_family_map(results_df_agg)
    best_row, judgment_text, _ = compute_best_judgment(results_df_agg, model_name_to_family=model_name_to_family)
    best_model_name = best_row['model']
    logger.info(f"Best model: {best_model_name} (composite={best_row['composite_score']:.4f}, RMSE={best_row['rmse']:.4f})")

    dataset_name = os.path.basename(args.file).split(".")[0]

    # Single results summary JSON: dataset, best_judgment, results (test metrics = final evaluation only)
    results_list = []
    for _, row in results_df_agg.iterrows():
        best_idx = int(row['best_variation_index'])
        cfg = run_configs[best_idx] if best_idx < len(run_configs) else {}
        entry = {
            'model': row['model'],
            'composite_score': round(float(row['composite_score']), 4),
            'rmse': float(row['rmse']),
            'mae': float(row['mae']),
            'r2': float(row['r2']),
            'variation_spec': cfg.get('variation_spec', {}),
            'hyperparameters': cfg.get('hyperparameters', {}),
            'hyperparameters_source': 'validation' if cfg.get('tuned') else 'default_variation',
        }
        if 'mase' in row and not (pd.isna(row['mase']) or np.isinf(row['mase'])):
            entry['mase'] = float(row['mase'])
        if 'mape' in row and not (pd.isna(row['mape']) or np.isinf(row['mape'])):
            entry['mape'] = float(row['mape'])
        if 'rmse_std' in row and pd.notna(row.get('rmse_std')):
            entry['rmse_std'] = round(float(row['rmse_std']), 4)
        if 'composite_std' in row and pd.notna(row.get('composite_std')):
            entry['composite_std'] = round(float(row['composite_std']), 4)
        results_list.append(entry)

    results_summary = {
        "dataset": {
            "name": dataset_name,
            "file": args.file,
            "time_column": args.time_col,
            "data_column": args.data_col,
            "total_samples": len(y_train) + len(y_test),
            "training_samples": len(y_train),
            "testing_samples": len(y_test),
        },
        "best_judgment": judgment_text,
        "results": results_list,
    }
    if best_per_run:
        results_summary["best_per_run"] = best_per_run
    if tuned_best_params:
        results_summary["tuned_best_params"] = tuned_best_params
        results_summary["parameter_selection"] = {
            "metric": "rmse",
            "note": "Best parameters chosen by validation only; test metrics are final evaluation only.",
        }
    summary_path = os.path.join(dataset_results_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Results summary saved to '{summary_path}'")

    generate_html_report(dataset_results_dir, results_summary, results_df_agg)

    # Console report (same column order and formatting as CSV)
    _print_results_report(results_df_agg, top_models_df, judgment_text)

def tune_selected_models(args, default_setup, models_to_tune, tuning_functions,
                        X_train, X_test, y_train, y_test, train_df, test_df):
    """Tune selected (model_key, loss) combos and return their predictions."""
    logger = logging.getLogger(__name__)

    tuned_predictions = []
    tuned_names = []
    tuned_results = []

    dataset_results_dir = _dataset_results_dir(args)
    os.makedirs(dataset_results_dir, exist_ok=True)
    seasonal_periods = determine_seasonal_periods(y_train)

    # models_to_tune is list of (model_key, loss) or (model_key, None)
    all_tuned_params = {}
    models_that_will_tune = [(mk, loss) for (mk, loss) in models_to_tune if mk in tuning_functions]
    tune_bar = tqdm(models_that_will_tune, desc="Tuning", unit="model")
    for model_key, loss_key in tune_bar:
        label = f"{model_key.upper()}" + (f" ({loss_key.upper()})" if loss_key else "")
        tune_bar.set_postfix(model=label)
        try:
            t0 = time.time()
            logger.info(f"Tuning {label} ...")
            step = default_setup["n_steps_univariate"] if model_key in ("rnn", "lstm") else default_setup["n_steps_feature"]
            model_params = {
                "y_train": y_train, "y_test": y_test,
                "X_train": X_train, "X_test": X_test,
                "n_steps": step,
                "seasonal_periods": seasonal_periods,
                "ma_window": default_setup["ma_window"],
                "results_dir": dataset_results_dir,
            }
            if loss_key is not None:
                model_params["loss"] = loss_key

            pred, name, best_params = execute_tuning(model_key, tuning_functions, model_params)
            tuned_predictions.append(pred)
            tuned_names.append(name)
            tuned_results.append(evaluate_model(y_test, pred, name, y_train))

            params_suffix = f"{model_key}_{loss_key}" if loss_key else model_key
            all_tuned_params[params_suffix] = best_params if best_params is not None else {}
            logger.info(f"  {label} tuning done in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"Error tuning {model_key.upper()} model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    return tuned_predictions, tuned_names, tuned_results, all_tuned_params

def determine_seasonal_periods(y_train):
    """Determine appropriate seasonal periods based on training series length.

    Ordered largest-to-smallest so each branch is reachable:
    weekly (104+) → quarterly (24+) → quarterly default.
    """
    n = len(y_train)
    if n > 52 * 2:   # 104+ points → assume weekly or higher frequency
        return 52
    if n > 12 * 2:   # 25-104 points → assume monthly
        return 12
    return 4          # shorter series → quarterly

# Helper functions for model execution
def execute_model(model_key, available_models, params):
    """Execute a model with the appropriate parameters and return predictions."""
    # Implementation details moved to models/registry.py
    return available_models[model_key](**params)

def execute_tuning(model_key, tuning_functions, params):
    """Execute tuning for a model with the appropriate parameters and return predictions."""
    # Implementation details moved to models/registry.py
    return tuning_functions[model_key](**params)

if __name__ == "__main__":
    main()