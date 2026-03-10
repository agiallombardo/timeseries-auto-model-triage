import logging
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..models.exponential_smoothing import run_exponential_smoothing
from ._validation import (
    get_chronological_holdout_indices,
    score_validation_rmse,
    write_tuning_artifacts,
    DEFAULT_VAL_FRAC,
)

logger = logging.getLogger(__name__)


def grid_search_exponential_smoothing(train_data, test_data, seasonal_periods=12, results_dir=None, val_frac=None, **kwargs):
    """
    Grid search for Exponential Smoothing using train-only validation (out-of-sample RMSE).
    Refits best model on full training data before producing test predictions.
    Returns (best_params, best_predictions).
    """
    logger.info("Performing grid search for Exponential Smoothing model (validation RMSE)...")
    val_frac = val_frac or DEFAULT_VAL_FRAC
    train_arr = np.asarray(train_data).ravel()
    n = len(train_arr)
    train_idx, val_idx = get_chronological_holdout_indices(n, val_frac)
    train_part = train_arr[train_idx]
    val_part = train_arr[val_idx]
    n_val = len(val_part)

    trend_types = ["add", "mul", None]
    seasonal_types = ["add", "mul", None]

    best_rmse = float("inf")
    best_params = None
    results = []

    for trend in trend_types:
        for seasonal in seasonal_types:
            if seasonal is not None and len(train_part) < 2 * seasonal_periods:
                continue
            try:
                model = ExponentialSmoothing(
                    train_part,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods if seasonal else None,
                )
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=n_val)
                rmse = score_validation_rmse(val_part, forecast)
                params_dict = {"trend": trend, "seasonal": seasonal}
                results.append({
                    "trend": trend,
                    "seasonal": seasonal,
                    "validation_rmse": rmse,
                    "params": params_dict,
                })
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (trend, seasonal)
                logger.debug(f"ETS(trend={trend}, seasonal={seasonal}) - validation RMSE: {rmse:.4f}")
            except Exception:
                continue

    logger.info(f"Best Exponential Smoothing parameters: trend={best_params[0]}, seasonal={best_params[1]} with validation RMSE: {best_rmse:.4f}")
    best_predictions = run_exponential_smoothing(
        train_data, test_data,
        trend=best_params[0],
        seasonal=best_params[1],
        seasonal_periods=seasonal_periods if best_params[1] else None,
    )

    if results:
        path = os.path.join(results_dir, "es_grid_search_results.csv") if results_dir else "es_grid_search_results.csv"
        pd.DataFrame(results).sort_values("validation_rmse").to_csv(path, index=False)
        write_tuning_artifacts(
            results_dir,
            "exponential_smoothing",
            [{"params": {"trend": r["trend"], "seasonal": r["seasonal"]}, "validation_rmse": r["validation_rmse"], "aggregate_score": r["validation_rmse"]} for r in results],
            {"trend": best_params[0], "seasonal": best_params[1]},
            {"aggregate_score": best_rmse, "metric": "rmse", "fold_scores": [best_rmse]},
            filename_suffix="tuning_results",
        )
    return {"trend": best_params[0], "seasonal": best_params[1]}, best_predictions
