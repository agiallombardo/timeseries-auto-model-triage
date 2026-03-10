# Variation audit: what we test vs what we don't

This doc lists what is **varied** (3 variations per model), what is **fixed**, and remaining low-priority axes.

---

## Currently varied (3 variations per model)

| Model | Variation 1 | Variation 2 | Variation 3 |
|-------|--------------|-------------|-------------|
| **arima** | order (1,0,0) | (1,1,0) | (2,1,0) |
| **sarima** | order × seasonal_order combo 1 | combo 2 | combo 3 |
| **ma** | window 3 | 5 | 7 |
| **es** | trend+seasonal add/add | add/None | mul/mul |
| **prophet** | additive, low priors | multiplicative, mid priors | additive, high priors |
| **rf** | l2, lags=3 | l1, lags=5 | l2, n_est=200, lags=7 |
| **svr** | C=1, rbf, lags=3 | C=10, rbf, lags=5 | C=10, linear, lags=7 |
| **xgb** | l2, lags=3 | l1, lags=5 | huber, lags=7 |
| **lr** | l2, lags=3 | l1 (α=0.1), lags=5 | huber, lags=7 |
| **rnn** | l2, steps=3, standard, relu | l1, steps=5, minmax, relu | huber, steps=10, robust, **tanh** |
| **lstm** | l2, steps=3, standard, relu | l1, steps=5, minmax, relu | huber, steps=10, robust, **tanh** |
| **mlp** | l2, standard, lags=3, relu | l1, minmax, lags=5, relu | huber, robust, lags=7, **tanh, lr=0.01** |
| **lstm_feat** | l2, steps=3, std, lags=3, relu | l1, steps=5, minmax, lags=5, relu | huber, steps=7, robust, lags=7, **tanh, lr=0.01** |
| **rnn_feat** | l2, steps=3, std, lags=3, relu | l1, steps=5, minmax, lags=5, relu | huber, steps=7, robust, lags=7, **tanh, lr=0.01** |
| **cnn1d** | l2, steps=3, std, lags=3, lr=0.001 | l1, steps=5, minmax, lags=5, lr=0.001 | huber, steps=7, robust, lags=7, **lr=0.01** |

Bold = axes added in the last two implementation passes.

---

## How lags variation works

Features are built once with `DEFAULT_SETUP["lags"] = 7` (maximum). `_subset_lag_features` in `main.py` returns only `lag_1..lag_N` columns plus rolling/time columns for the variation's N, so all models share the same `y_train`/`y_test` for fair metric comparison. Statistical and univariate-sequence models ignore the lag feature matrix entirely.

---

## What the tuners search (when `--tune_top` or `--tune_all` is used)

| Model | Tuner grid |
|-------|-----------|
| ARIMA | p∈[1..5], d∈[0,1], q∈[0,1,2] |
| SARIMA | subset of p,d,q × P,D,Q |
| MA | window ∈ [2..10] |
| ETS | trend × seasonal combos |
| Prophet | changepoint_prior × seasonality_prior |
| RF | n_estimators, max_depth, loss |
| SVR | C, kernel, gamma, epsilon |
| XGB | n_estimators, max_depth, learning_rate, loss |
| LR | loss (alpha fixed per variation) |
| RNN | n_steps, units, learning_rate, batch_size |
| LSTM | n_steps, units, layers, learning_rate, batch_size |
| MLP | **activation**, units, dropout, learning_rate (uses variation's scaler) |
| LSTM-feat | **activation**, n_steps, units, dropout, learning_rate (uses variation's scaler) |
| RNN-feat | **activation**, n_steps, units, dropout, learning_rate (uses variation's scaler) |
| CNN-1D | n_steps, filters, kernel_size, learning_rate (uses variation's scaler) |

All tuners now use the variation's scaler (via `get_scaler`) consistently with the non-tuned runner path.

---

## Fixed (same for all models and runs)

- **Optimizer**: Adam throughout. SGD / RMSProp not tested.
- **Train/test split**: `test_size=0.2`.
- **Rolling window**: 3 (rolling_mean, rolling_std in all feature matrices).
- **Validation policy**: `TUNING_SETUP` — n_splits=3, val_frac=0.2, selection_metric=rmse.
- **Quantile loss**: Listed in `LOSS_SUPPORTED_MODELS` for many models but not in any variation (3 slots are fully used by l2/l1/huber).
- **Seasonal period**: derived from data length; not explicitly varied.

---

## Summary table: "Are we testing it?"

| Axis | Varied? | Where |
|------|---------|-------|
| Model family | Yes | Many models |
| Loss (l2/l1/huber) | Yes | All loss-supported models |
| Scaler (std/minmax/robust) | Yes | All DL + MLP/feat models |
| Activation (relu/tanh) | Yes | RNN, LSTM, MLP, LSTM-feat, RNN-feat |
| Learning rate | Yes | MLP, LSTM-feat, RNN-feat, CNN1d (via variation + tuner) |
| Lags / feature set | Yes | RF, SVR, XGB, LR, MLP, lstm_feat, rnn_feat, cnn1d |
| LR regularization strength (alpha) | Yes | LR variation 2 (alpha=0.1) |
| Prophet seasonality_mode | Yes | additive vs multiplicative |
| SVR kernel | Yes | rbf vs linear |
| RF n_estimators | Yes | Variation 3 (n_estimators=200) |
| Quantile loss | No | Not enough variation slots |
| Optimizer (SGD/RMSProp) | No | Adam hardcoded |
| Test size | No | Fixed 0.2 |
| Validation strictness | No | n_splits=3, val_frac=0.2 fixed |

---

## Remaining lower-priority axes

- **Quantile loss** — Add a 4th variation slot or replace one variation per model to include quantile.
- **Optimizer** — Expose optimizer as a variation parameter for DL models if gradient dynamics matter.
- **Test size sensitivity** — Run with 0.15/0.25 to check if ranking is stable.
- **Validation n_splits / val_frac** — Try n_splits=5 for tabular or val_frac=0.25 for DL to check tuning stability.
