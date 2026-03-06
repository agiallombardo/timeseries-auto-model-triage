#!/usr/bin/env python3
import argparse
import os
import json
import logging
import traceback
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils import setup_logging
from src.data_handling import load_data, prepare_features, split_data
from src.evaluation import (
    evaluate_model, plot_results, create_trellis_plot,
    create_top_models_plot, create_performance_chart, compute_best_judgment
)
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
    'composite_score': '{:.3f}',
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


def _print_results_report(results_df_agg, top_models_df, judgment_text):
    """Print a consistent results report: table, best judgment, and top 3 models."""
    cols = [c for c in REPORT_COLUMNS if c in results_df_agg.columns]
    if not cols:
        return
    # Table with consistent formatting
    disp = _format_results_for_display(results_df_agg)
    print("\n" + "=" * 64)
    print("RESULTS (median across variations, sorted by composite score)")
    print("=" * 64)
    print(disp.to_string(index=False))
    print("-" * 64)
    print("Best:", judgment_text)
    print("=" * 64)
    # Top 3
    print("\nTop 3 models:")
    for i, (_, row) in enumerate(top_models_df.iterrows(), 1):
        parts = [f"{i}. {row['model']}", f"composite={row['composite_score']:.3f}", f"RMSE={row['rmse']:.4f}", f"MAE={row['mae']:.4f}", f"R²={row['r2']:.4f}"]
        if 'mase' in row and pd.notna(row.get('mase')) and not np.isinf(row.get('mase', 0)):
            parts.append(f"MASE={row['mase']:.3f}")
        if 'mape' in row and pd.notna(row.get('mape')) and not np.isinf(row.get('mape', 0)):
            parts.append(f"MAPE={row['mape']:.2f}%")
        print("  " + " | ".join(parts))


def _parse_display_name(display_name):
    """Parse 'XGBoost (L2)' -> ('xgb', 'l2'), 'ARIMA' -> ('arima', None)."""
    parts = display_name.split(" (")
    base = parts[0].strip()
    first_word = base.lower().split()[0]
    model_key = DISPLAY_NAME_TO_REGISTRY_KEY.get(first_word, first_word)
    if len(parts) > 1:
        loss_suffix = parts[1].rstrip(")").strip().lower()
        loss_map = {'l1': 'l1', 'l2': 'l2', 'huber': 'huber', 'quantile': 'quantile'}
        loss = loss_map.get(loss_suffix, loss_suffix)
    else:
        loss = None
    return model_key, loss


def _aggregate_run_results(run_results, model_name):
    """Aggregate n_runs evaluation dicts into one with mean metrics for reranking."""
    if not run_results:
        return {"model": model_name, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mse": np.nan, "mase": np.nan, "mape": np.nan}
    keys = ["mse", "rmse", "mae", "r2", "mase", "mape"]
    out = {"model": model_name}
    for k in keys:
        if k not in run_results[0]:
            continue
        vals = [r[k] for r in run_results if k in r]
        if not vals:
            continue
        out[k] = float(np.nanmean(vals))
    return out


def _dataset_results_dir(args):
    """Results path: output_dir / dataset_name / date (YYYY-MM-DD)."""
    dataset_name = os.path.basename(args.file).split(".")[0]
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(args.output_dir, dataset_name, date_str)


def _row_to_csv_ready(row, json_cols):
    """Convert a dict to CSV-ready dict: serialize listed keys to JSON strings."""
    out = {}
    for k, v in row.items():
        if k in json_cols and isinstance(v, (dict, list)):
            out[k] = json.dumps(v) if v is not None else ""
        else:
            out[k] = v
    return out


def _write_configs_csv(configs, path):
    """Write list of config dicts to CSV (nested dicts as JSON strings)."""
    if not configs:
        return
    json_cols = {"setup", "time_steps", "normalization", "shape_and_reshaping", "variation_spec", "hyperparameters"}
    rows = [_row_to_csv_ready(c, json_cols) for c in configs]
    keys = list(rows[0].keys()) if rows else []
    pd.DataFrame(rows)[keys].to_csv(path, index=False)


def _write_summary_csvs(dataset_results_dir, dataset_info, best_judgment, best_per_run, top_models, tuned_best_params):
    """Write results summary as CSV files instead of a single JSON."""
    # Dataset + best judgment
    row = dict(dataset_info)
    row["best_judgment"] = best_judgment
    pd.DataFrame([row]).to_csv(os.path.join(dataset_results_dir, "dataset_info.csv"), index=False)

    # Best per run
    if best_per_run:
        json_cols = {"variation_spec", "hyperparameters"}
        rows = [_row_to_csv_ready(r, json_cols) for r in best_per_run]
        pd.DataFrame(rows).to_csv(os.path.join(dataset_results_dir, "best_per_run.csv"), index=False)

    # Top models
    if top_models:
        json_cols = {"variation_spec", "hyperparameters"}
        rows = [_row_to_csv_ready(r, json_cols) for r in top_models]
        pd.DataFrame(rows).to_csv(os.path.join(dataset_results_dir, "top_models.csv"), index=False)

    # Tuned best params (model_key -> hyperparameters)
    if tuned_best_params:
        rows = [{"model_key": k, "hyperparameters": json.dumps(v) if isinstance(v, dict) else str(v)} for k, v in tuned_best_params.items()]
        pd.DataFrame(rows).to_csv(os.path.join(dataset_results_dir, "tuned_best_params.csv"), index=False)


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
    setup = {
        "data_file": args.file,
        "time_column": args.time_col,
        "data_column": args.data_col,
        "test_size": default_setup["test_size"],
        "lags": default_setup["lags"],
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
    normalization = build_normalization_entry(metadata) if metadata else {"type": "none", "scope": None, "description": None}
    shape_and_reshaping = (metadata.get("shape_and_reshaping") if metadata else None) or None
    return {
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


# Module-level logger
logger = logging.getLogger(__name__)

def main():

    """Main function to run the time series forecasting comparison."""
    parser = argparse.ArgumentParser(description='Time Series Forecasting Model Comparison')
    parser.add_argument('--file', required=True, help='Path to the data file')
    parser.add_argument('--time_col', required=True, help='Name of the time/date column')
    parser.add_argument('--data_col', required=True, help='Name of the data column to forecast')
    parser.add_argument('--date_format', help='Format of the date string (if needed)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--lags', type=int, default=5, help='Number of lag features for ML models')
    parser.add_argument('--n_steps', type=int, default=3, help='Number of time steps for sequence models (RNN/LSTM)')
    parser.add_argument('--ma_window', type=int, default=3, help='Window size for Moving Average')
    parser.add_argument('--models', nargs='+', default=['all'],
                        help='Models to run (all, arima, sarima, es, prophet, rf, svr, xgb, ma, lr, rnn, lstm, mlp, lstm_feat, rnn_feat, cnn1d)')
    parser.add_argument('--losses', nargs='+', default=None,
                        help='Loss variants to run (default: all). Choices: l1, l2, huber, quantile. Only applies to models that support losses.')
    parser.add_argument('--tune_top', type=int, default=3, 
                   help='Number of top models to tune (default: 3, set to 0 to disable tuning)')
    parser.add_argument('--tune_all', action='store_true', 
                   help='Tune all models instead of just the top performing ones')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir)

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

    logger.info(f"Loading data from {args.file}")
    logger.info(f"Time column: {args.time_col}")
    logger.info(f"Data column: {args.data_col}")

    default_setup = get_default_setup()
    logger.info(f"Using default setup: test_size={default_setup['test_size']}, lags={default_setup['lags']}")

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
    process_results(
        args, default_setup, all_predictions, model_names, results, run_configs, best_per_run,
        y_train, y_test, X_train, X_test, tuning_functions, tuned_best_params=all_tuned_params
    )

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
        for model_key in models_to_run:
            if model_key not in tuning_functions:
                continue
            variations = get_variations_for_model(model_key)
            for variation_index, variation_spec in enumerate(variations):
                try:
                    loss_key = variation_spec.get("loss")
                    logger.info(f"Tuning {model_key.upper()}" + (f" ({loss_key.upper()})" if loss_key else "") + " ...")
                    model_params = _base_params(default_setup, model_key, y_train, y_test, X_train, X_test, seasonal_periods)
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
    n_runs = default_setup.get("n_runs", 3)
    all_runs_preds = []
    all_runs_results = []
    first_run_names = []
    best_per_run = []

    for run_idx in range(n_runs):
        logger.info(f"=== Program run {run_idx + 1}/{n_runs} ===")
        run_preds = []
        run_results = []
        for model_key in models_to_run:
            if model_key not in available_models:
                continue
            variations = get_variations_for_model(model_key)
            metadata = get_metadata_for_model(model_key)
            default_hp = dict(metadata.get("default_hyperparameters", {})) if metadata else {}
            for variation_index, variation_spec in enumerate(variations):
                try:
                    model_params = _base_params(default_setup, model_key, y_train, y_test, X_train, X_test, seasonal_periods)
                    model_params.update(variation_spec)
                    loss_key = variation_spec.get("loss")
                    logger.info(
                        f"Running {model_key.upper()}" + (f" ({loss_key.upper()})" if loss_key else "") + " ..."
                    )
                    pred, name = execute_model(model_key, available_models, model_params)
                    run_preds.append(np.asarray(pred))
                    run_results.append(evaluate_model(y_test, pred, name, y_train))
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
                "composite_score": float(best_row["composite_score"]),
                "rmse": float(best_row["rmse"]),
                "mae": float(best_row["mae"]),
                "r2": float(best_row["r2"]),
                "variation_spec": cfg.get("variation_spec", {}),
                "hyperparameters": cfg.get("hyperparameters", {}),
            })
            logger.info(f"Run {run_idx + 1} best: {best_row['model']} (composite={best_row['composite_score']:.3f})")

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
        results.append(_aggregate_run_results(result_dicts, name))

    return all_predictions, model_names, results, run_configs, best_per_run


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

def process_results(args, default_setup, all_predictions, model_names, results, run_configs, best_per_run, y_train, y_test, X_train, X_test, tuning_functions, tuned_best_params=None):
    """Process and display results. Saves model_configs.csv and summary CSVs (dataset_info, best_per_run, top_models, tuned_best_params)."""
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

    # Aggregate to one row per model: median metrics across variations, params from best variation
    metric_cols = [c for c in ['composite_score', 'rmse', 'mae', 'r2', 'mase', 'mape'] if c in results_df.columns]
    agg_list = []
    for model_name, group in results_df.groupby('model', sort=False):
        medians = group[metric_cols].median()
        best_idx = int(group['composite_score'].idxmax())
        cfg = run_configs[best_idx] if best_idx < len(run_configs) else {}
        row = {'model': model_name, 'best_variation_index': best_idx}
        for c in metric_cols:
            row[c] = medians[c]
        row['variation_spec'] = cfg.get('variation_spec', {})
        row['hyperparameters'] = cfg.get('hyperparameters', {})
        agg_list.append(row)
    results_df_agg = pd.DataFrame(agg_list).sort_values('composite_score', ascending=False).reset_index(drop=True)

    # Save model configs as CSV (same order as results; one row per model, best variation first)
    if run_configs:
        ordered_configs = [run_configs[int(row["best_variation_index"])] for _, row in results_df_agg.iterrows() if int(row["best_variation_index"]) < len(run_configs)]
        configs_path = os.path.join(dataset_results_dir, "model_configs.csv")
        _write_configs_csv(ordered_configs, configs_path)
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

    # Trellis: top 9 models, each shown with its best variation's prediction
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

    # Best model = highest median composite (one row per model)
    best_row = results_df_agg.iloc[0]
    best_model_name = best_row['model']
    best_model_key = best_model_name.lower().split()[0]
    judgment_text = f"Recommended model: {best_model_name} (median composite score: {best_row['composite_score']:.3f}). Best balance across RMSE, MAE, R², MASE, and MAPE among the tested models."

    logger.info(f"Best model: {best_model_name} (composite={best_row['composite_score']:.3f}, RMSE={best_row['rmse']:.4f})")

    # Save top models information
    top_models_info = {}
    
    # Add dataset information
    dataset_name = os.path.basename(args.file).split(".")[0]
    top_models_info["dataset"] = {
        "name": dataset_name,
        'file': args.file,
        'time_column': args.time_col,
        'data_column': args.data_col,
        'total_samples': len(y_train) + len(y_test),
        'training_samples': len(y_train),
        'testing_samples': len(y_test)
    }
    
    # Add model information (median metrics; params from best variation)
    top_models_info['best_judgment'] = judgment_text
    top_models_info['models'] = []
    for idx, (_, row) in enumerate(top_models_df.iterrows()):
        best_idx = int(row['best_variation_index'])
        cfg = run_configs[best_idx] if best_idx < len(run_configs) else {}
        model_info = {
            'rank': idx + 1,
            'name': row['model'],
            'composite_score': float(row['composite_score']),
            'rmse': float(row['rmse']),
            'mae': float(row['mae']),
            'r2': float(row['r2']),
            'variation_spec': cfg.get('variation_spec', {}),
            'hyperparameters': cfg.get('hyperparameters', {}),
        }
        if 'mase' in row and not (pd.isna(row['mase']) or np.isinf(row['mase'])):
            model_info['mase'] = float(row['mase'])
        if 'mape' in row and not (pd.isna(row['mape']) or np.isinf(row['mape'])):
            model_info['mape'] = float(row['mape'])
        top_models_info['models'].append(model_info)

    # Summary as CSV files (dataset_info, best_per_run, top_models, tuned_best_params)
    _write_summary_csvs(
        dataset_results_dir,
        top_models_info["dataset"],
        top_models_info["best_judgment"],
        best_per_run,
        top_models_info["models"],
        tuned_best_params if tuned_best_params else {},
    )
    logger.info(f"Results summary saved to CSVs in '{dataset_results_dir}'")

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
    for model_key, loss_key in models_that_will_tune:
        try:
            logger.info(f"Tuning {model_key.upper()}" + (f" ({loss_key.upper()})" if loss_key else "") + " ...")
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
        except Exception as e:
            logger.error(f"Error tuning {model_key.upper()} model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    return tuned_predictions, tuned_names, tuned_results, all_tuned_params

def determine_seasonal_periods(y_train):
    """Determine appropriate seasonal periods based on data frequency."""
    if len(y_train) > 12 * 2:  # At least 2 years of monthly data
        return 12
    elif len(y_train) > 52 * 2:  # At least 2 years of weekly data
        return 52
    else:
        return 4  # Default to quarterly

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