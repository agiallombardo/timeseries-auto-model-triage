"""
Default setup, exactly 3 variations per model, and metadata for config saving.
No CLI: all behavior uses these built-in defaults.

Variation axes (settings tested to find best output):
- Preprocessing: scaler (standard, minmax, robust, none), feature_range for MinMax.
- Sequence: n_steps (look-back window), reshape (samples, timesteps, features).
- Model: loss (l2, l1, huber, quantile where supported), units, layers, dropout_rate,
  activation (relu, tanh), learning_rate.
- Training: optimizer, batch_size, epochs (in tuning).
"""

# Default setup: single value used when one is needed (e.g. feature build); 3 variations tested per model
# Option lists define the 3 possibilities tested to find the best (see VARIATIONS_PER_MODEL).
DEFAULT_SETUP = {
    "test_size": 0.2,
    "lags": 5,                          # middle of 3 tested; options [3, 5, 7] for ML feature lags
    "lags_options": [3, 5, 7],
    "rolling_window": 3,
    "n_steps_univariate": 5,            # middle of 3 tested; options [3, 5, 10] for RNN/LSTM
    "n_steps_univariate_options": [3, 5, 10],
    "n_steps_feature": 5,               # middle of 3 tested; options [3, 5, 7] for LSTM-feat, RNN-feat, CNN-1D
    "n_steps_feature_options": [3, 5, 7],
    "ma_window": 5,                     # middle of 3 tested; options [3, 5, 7] for MA
    "ma_window_options": [3, 5, 7],
    "n_runs": 3,                        # run each (model, variation) this many times; rerank by mean metrics
}

# Tuning policy: same validation contract for all models (train-only splits, RMSE selection)
TUNING_SETUP = {
    "n_splits": 3,                      # TimeSeriesSplit / expanding-window folds for tabular
    "val_frac": 0.2,                   # last fraction of train as validation for DL/statistical
    "selection_metric": "rmse",
}

# Exactly 3 variation specs per model (aligned to what each model supports)
VARIATIONS_PER_MODEL = {
    "arima": [
        {"order": [1, 0, 0]},
        {"order": [1, 1, 0]},
        {"order": [2, 1, 0]},
    ],
    "sarima": [
        {"order": [1, 1, 0], "seasonal_order": [0, 1, 1, 12]},
        {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]},
        {"order": [2, 1, 0], "seasonal_order": [0, 1, 1, 12]},
    ],
    "ma": [
        {"window": 3},
        {"window": 5},
        {"window": 7},
    ],
    "es": [
        {"trend": "add", "seasonal": "add"},
        {"trend": "add", "seasonal": None},
        {"trend": "mul", "seasonal": "mul"},
    ],
    "prophet": [
        {"changepoint_prior_scale": 0.01, "seasonality_prior_scale": 0.1},
        {"changepoint_prior_scale": 0.1, "seasonality_prior_scale": 1.0},
        {"changepoint_prior_scale": 0.5, "seasonality_prior_scale": 10.0},
    ],
    "rf": [
        {"loss": "l2"},
        {"loss": "l1"},
        {"loss": "l2"},  # only 2 losses; third variation repeats l2
    ],
    "svr": [
        {"C": 1.0, "kernel": "rbf"},
        {"C": 10.0, "kernel": "rbf"},
        {"C": 100.0, "kernel": "rbf"},
    ],
    "xgb": [
        {"loss": "l2"},
        {"loss": "l1"},
        {"loss": "huber"},
    ],
    "lr": [
        {"loss": "l2"},
        {"loss": "l1"},
        {"loss": "huber"},
    ],
    "rnn": [
        {"loss": "l2", "n_steps": 3, "scaler": "standard"},
        {"loss": "l2", "n_steps": 5, "scaler": "minmax", "feature_range": [0, 1]},
        {"loss": "l2", "n_steps": 10, "units": 64, "scaler": "standard"},
    ],
    "lstm": [
        {"loss": "l2", "n_steps": 3, "scaler": "standard"},
        {"loss": "l2", "n_steps": 5, "scaler": "minmax", "feature_range": [0, 1]},
        {"loss": "l2", "n_steps": 10, "units": 64, "dropout_rate": 0.2, "scaler": "standard"},
    ],
    "mlp": [
        {"loss": "l2"},
        {"loss": "l1"},
        {"loss": "huber"},
    ],
    "lstm_feat": [
        {"loss": "l2", "n_steps": 3},
        {"loss": "l2", "n_steps": 5},
        {"loss": "l2", "n_steps": 7},
    ],
    "rnn_feat": [
        {"loss": "l2", "n_steps": 3},
        {"loss": "l2", "n_steps": 5},
        {"loss": "l2", "n_steps": 7},
    ],
    "cnn1d": [
        {"loss": "l2", "n_steps": 3},
        {"loss": "l2", "n_steps": 5},
        {"loss": "l2", "n_steps": 7},
    ],
}

# Metadata per model: scaling, input type, shape, uses_n_steps, default hyperparameters
MODEL_METADATA = {
    "arima": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"order": [5, 1, 0]},
    },
    "sarima": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]},
    },
    "ma": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"window": 3},
    },
    "es": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
    },
    "prophet": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {},
    },
    "rf": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"loss": "l2"},
    },
    "svr": {
        "scaling": "X_and_y",
        "scaling_description": "StandardScaler on X and y; fit on train, inverse on predictions",
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {},
    },
    "xgb": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"loss": "l2"},
    },
    "lr": {
        "scaling": "none",
        "scaling_description": None,
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"loss": "l2"},
    },
    "rnn": {
        "scaling": "target",
        "scaling_description": "StandardScaler on target; fit on train reshape (-1,1); inverse on predictions",
        "input_type": "univariate_sequence",
        "uses_n_steps": True,
        "default_n_steps": 3,
        "shape_and_reshaping": {
            "input_shape": [3, 1],
            "sequence_builder": "prepare_sequence_data",
            "reshape": "(samples, n_steps, 1)",
        },
        "default_hyperparameters": {"n_steps": 3, "units": 50, "learning_rate": 0.001, "scaler": "standard"},
    },
    "lstm": {
        "scaling": "target",
        "scaling_description": "StandardScaler on target; fit on train reshape (-1,1); inverse on predictions",
        "input_type": "univariate_sequence",
        "uses_n_steps": True,
        "default_n_steps": 3,
        "shape_and_reshaping": {
            "input_shape": [3, 1],
            "sequence_builder": "prepare_sequence_data",
            "reshape": "(samples, n_steps, 1)",
        },
        "default_hyperparameters": {"n_steps": 3, "units": 50, "layers": 1, "dropout_rate": 0.0, "learning_rate": 0.001, "scaler": "standard"},
    },
    "mlp": {
        "scaling": "X_and_y",
        "scaling_description": "StandardScaler on X and y; fit on train",
        "input_type": "tabular",
        "uses_n_steps": False,
        "default_n_steps": None,
        "shape_and_reshaping": None,
        "default_hyperparameters": {"loss": "l2"},
    },
    "lstm_feat": {
        "scaling": "features",
        "scaling_description": "StandardScaler on sequence features; reshape (-1, n_features) fit/transform then reshape back to (samples, n_steps, n_features)",
        "input_type": "feature_sequence",
        "uses_n_steps": True,
        "default_n_steps": 5,
        "shape_and_reshaping": {
            "input_shape": [5, "n_features"],
            "sequence_builder": "prepare_feature_sequences",
            "reshape": "(samples, n_steps, n_features)",
        },
        "default_hyperparameters": {"n_steps": 5, "units": 64, "loss": "l2"},
    },
    "rnn_feat": {
        "scaling": "features",
        "scaling_description": "StandardScaler on sequence features; reshape (-1, n_features) fit/transform then reshape back to (samples, n_steps, n_features)",
        "input_type": "feature_sequence",
        "uses_n_steps": True,
        "default_n_steps": 5,
        "shape_and_reshaping": {
            "input_shape": [5, "n_features"],
            "sequence_builder": "prepare_feature_sequences",
            "reshape": "(samples, n_steps, n_features)",
        },
        "default_hyperparameters": {"n_steps": 5, "loss": "l2"},
    },
    "cnn1d": {
        "scaling": "features",
        "scaling_description": "StandardScaler on sequence features; reshape (-1, n_features) fit/transform then reshape back to (samples, n_steps, n_features)",
        "input_type": "feature_sequence",
        "uses_n_steps": True,
        "default_n_steps": 5,
        "shape_and_reshaping": {
            "input_shape": [5, "n_features"],
            "sequence_builder": "prepare_feature_sequences",
            "reshape": "(samples, n_steps, n_features)",
        },
        "default_hyperparameters": {"n_steps": 5, "loss": "l2"},
    },
}


def get_default_setup():
    return dict(DEFAULT_SETUP)


def get_variations_for_model(model_key):
    return list(VARIATIONS_PER_MODEL.get(model_key, [{}]))


def get_metadata_for_model(model_key):
    return dict(MODEL_METADATA.get(model_key, {}))


def build_normalization_entry(metadata):
    if not metadata or metadata.get("scaling") == "none":
        return {"type": "none", "scope": None, "description": None}
    return {
        "type": "StandardScaler",
        "scope": metadata["scaling"],
        "description": metadata.get("scaling_description"),
    }
