"""
Shared time-series validation policy for all tuners.

Contract:
- Splits are derived only from training data; no test data is used for parameter selection.
- No fold uses future data in training (chronological / expanding-window semantics).
- Default selection metric is RMSE; tuners refit on full training data before producing test predictions.
"""

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Default tuning policy (can be overridden via model_config.TUNING_SETUP)
DEFAULT_N_SPLITS = 3
DEFAULT_VAL_FRAC = 0.2
DEFAULT_SELECTION_METRIC = "rmse"


def get_time_series_splits(
    X: Union[np.ndarray, pd.DataFrame],
    n_splits: int = DEFAULT_N_SPLITS,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) from expanding-window time-series splits.
    Uses only training data; no test data. Each fold has train indices strictly before val indices.
    """
    n = len(X)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(np.arange(n)):
        yield train_idx, val_idx


def get_chronological_holdout(
    len_train: int,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Tuple[slice, slice]:
    """
    Return (train_slice, val_slice) for the last val_frac of training as validation.
    train_slice = 0:train_end, val_slice = train_end:len_train.
    No future data in training.
    """
    val_size = max(1, int(len_train * val_frac))
    val_size = min(val_size, len_train - 1)  # need at least 1 train sample
    train_end = len_train - val_size
    return slice(0, train_end), slice(train_end, len_train)


def get_chronological_holdout_indices(
    len_train: int,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) arrays for chronological holdout."""
    train_slice, val_slice = get_chronological_holdout(len_train, val_frac)
    return np.arange(len_train)[train_slice], np.arange(len_train)[val_slice]


def score_validation_rmse(
    y_val: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """Compute RMSE between validation targets and predictions."""
    y_v = np.asarray(y_val).ravel()
    y_p = np.asarray(y_pred).ravel()
    if len(y_v) == 0:
        return float("inf")
    return float(np.sqrt(mean_squared_error(y_v, y_p)))


def aggregate_fold_scores(
    scores: List[float],
    method: str = "mean",
) -> float:
    """Aggregate per-fold validation scores. method in ('mean', 'median')."""
    if not scores:
        return float("inf")
    arr = np.asarray(scores)
    if method == "median":
        return float(np.median(arr))
    return float(np.mean(arr))


def write_tuning_artifacts(
    results_dir: Optional[str],
    model_key: str,
    candidate_results: List[Dict[str, Any]],
    best_params: Dict[str, Any],
    validation_summary: Dict[str, Any],
    filename_suffix: Optional[str] = None,
) -> None:
    """
    Write a comparable tuning result table: candidate params, fold scores,
    aggregate validation score, chosen params, refit_done.
    """
    if not results_dir or not candidate_results:
        return
    suffix = filename_suffix or "tuning_results"
    path = os.path.join(results_dir, f"{model_key}_{suffix}.csv")
    best_agg = validation_summary.get("aggregate_score")
    rows = []
    for row in candidate_results:
        r = dict(row)
        agg = r.get("aggregate_score", r.get("validation_rmse", r.get("rmse")))
        r["chosen"] = agg is not None and best_agg is not None and abs(float(agg) - float(best_agg)) < 1e-9
        rows.append(r)
    df = pd.DataFrame(rows)
    if "fold_scores" in df.columns and df["fold_scores"].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        df = df.copy()
        df["fold_scores"] = df["fold_scores"].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
    df.to_csv(path, index=False)
    # Write small validation summary alongside
    summary_path = os.path.join(results_dir, f"{model_key}_validation_summary.json")
    summary = {
        "best_params": best_params,
        "validation_score": validation_summary.get("aggregate_score"),
        "selection_metric": validation_summary.get("metric", DEFAULT_SELECTION_METRIC),
        "refit_on_full_train": True,
    }
    if "fold_scores" in validation_summary:
        summary["fold_scores"] = validation_summary["fold_scores"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.debug("Wrote tuning artifacts to %s and %s", path, summary_path)


