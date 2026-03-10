import glob
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, model_name, y_train=None):
    """
    Evaluate model performance.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model being evaluated
    y_train : array-like, optional
        Training data (needed for MASE; used for MAPE scaling when zeros present)

    Returns:
    --------
    dict
        Dictionary with evaluation metrics (mse, rmse, mae, r2, mase, mape)
    """
    logger = logging.getLogger(__name__)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {'model': model_name, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    # MASE: scale by in-sample MAE of naïve (random walk)
    if y_train is not None and len(y_train) > 1:
        y_tr = np.asarray(y_train)
        scale = np.mean(np.abs(np.diff(y_tr)))
        if scale > 0:
            metrics['mase'] = mae / scale
        else:
            metrics['mase'] = np.nan
    else:
        metrics['mase'] = np.nan

    # MAPE: mean absolute percentage error, with zero-safe handling
    eps = 1e-10
    valid = np.abs(y_true) >= eps
    if np.any(valid):
        mape = np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0
        metrics['mape'] = mape
    else:
        metrics['mape'] = np.nan

    logger.info(
        f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
        + (f", MASE: {metrics['mase']:.4f}" if not np.isnan(metrics.get('mase', np.nan)) else "")
        + (f", MAPE: {metrics['mape']:.2f}%" if not np.isnan(metrics.get('mape', np.nan)) else "")
    )
    return metrics


def compute_best_judgment(results_df, model_name_to_family=None):
    """
    Evaluate all models holistically and produce a best judgment.

    Combines RMSE, MAE, R², MASE, and MAPE via min-max normalization so
    lower-error and higher-R² models score better. Returns the best model
    and a short explanation.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns: model, rmse, mae, r2, and optionally mase, mape
    model_name_to_family : dict, optional
        Map display name -> family string (e.g. "Statistical", "ML", "DL") for richer judgment text.

    Returns:
    --------
    tuple of (best_row, judgment_text, df_with_composite)
        best_row: row of best model; judgment_text: short explanation;
        df_with_composite: results_df with composite_score column, sorted by it
    """
    df = results_df.copy()

    # Ensure required columns
    for col in ['model', 'rmse', 'mae', 'r2']:
        if col not in df.columns:
            raise ValueError(f"results_df must contain column '{col}'")

    # Build per-metric normalized scores (0-1, 1 = best)
    # RMSE: lower is better -> invert
    r_min, r_max = df['rmse'].min(), df['rmse'].max()
    if r_max > r_min:
        df['_s_rmse'] = 1 - (df['rmse'] - r_min) / (r_max - r_min)
    else:
        df['_s_rmse'] = 1.0

    # MAE: lower is better -> invert
    m_min, m_max = df['mae'].min(), df['mae'].max()
    if m_max > m_min:
        df['_s_mae'] = 1 - (df['mae'] - m_min) / (m_max - m_min)
    else:
        df['_s_mae'] = 1.0

    # R²: higher is better
    r2_min, r2_max = df['r2'].min(), df['r2'].max()
    if r2_max > r2_min:
        df['_s_r2'] = (df['r2'] - r2_min) / (r2_max - r2_min)
    else:
        df['_s_r2'] = 1.0 if r2_max >= 0 else 0.0

    # MASE: lower is better (if available and finite)
    if 'mase' in df.columns:
        mase_vals = df['mase'].replace([np.inf, -np.inf], np.nan)
        valid = mase_vals.notna()
        if valid.any():
            mase_vals = mase_vals.clip(upper=10)  # cap extreme values
            m_min_m, m_max_m = mase_vals.min(), mase_vals.max()
            if m_max_m > m_min_m:
                df['_s_mase'] = np.where(valid, 1 - (mase_vals - m_min_m) / (m_max_m - m_min_m), 0.5)
            else:
                df['_s_mase'] = np.where(valid, 1.0, 0.5)
        else:
            df['_s_mase'] = 0.5  # neutral when unavailable
    else:
        df['_s_mase'] = 0.5

    # MAPE: lower is better (if available and finite)
    if 'mape' in df.columns:
        mape_vals = df['mape'].replace([np.inf, -np.inf], np.nan)
        valid = mape_vals.notna()
        if valid.any():
            mape_vals = mape_vals.clip(upper=200)  # cap at 200%
            p_min, p_max = mape_vals.min(), mape_vals.max()
            if p_max > p_min:
                df['_s_mape'] = np.where(valid, 1 - (mape_vals - p_min) / (p_max - p_min), 0.5)
            else:
                df['_s_mape'] = np.where(valid, 1.0, 0.5)
        else:
            df['_s_mape'] = 0.5
    else:
        df['_s_mape'] = 0.5

    # Composite: equal-weight average of normalized scores
    score_cols = ['_s_rmse', '_s_mae', '_s_r2', '_s_mase', '_s_mape']
    df['composite_score'] = df[score_cols].mean(axis=1)
    df = df.drop(columns=score_cols)

    # Sort by composite (best first)
    sort_df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    best_row = sort_df.iloc[0]

    # Build judgment text
    name = best_row['model']
    rmse = best_row['rmse']
    r2 = best_row['r2']
    composite = best_row['composite_score']

    parts = [f"Recommended model: {name} (composite score: {composite:.4f})."]
    parts.append(f"It achieves best overall balance across RMSE, MAE, R², MASE, and MAPE.")
    if r2 >= 0.7:
        parts.append("Strong explanatory power (R² ≥ 0.7).")
    elif r2 >= 0.5:
        parts.append("Moderate explanatory power (0.5 ≤ R² < 0.7).")
    elif r2 >= 0:
        parts.append("Low explanatory power (0 ≤ R² < 0.5); consider alternatives if interpretability matters.")
    else:
        parts.append("Negative R² indicates model underperforms a constant mean; review data and features.")

    if 'mase' in best_row and not (np.isnan(best_row['mase']) or np.isinf(best_row['mase'])):
        mase = best_row['mase']
        if mase < 1:
            parts.append(f"MASE < 1: outperforms a naïve baseline.")
        else:
            parts.append(f"MASE = {mase:.2f}: consider simpler baselines if parsimony is important.")

    if len(sort_df) >= 2:
        margin = float(sort_df.iloc[0]['composite_score'] - sort_df.iloc[1]['composite_score'])
        parts.append(f"Leads 2nd place by {margin:.3f} composite points.")
    if model_name_to_family and name in model_name_to_family:
        parts.append(f"Model family: {model_name_to_family[name]}.")

    judgment_text = " ".join(parts)
    return best_row, judgment_text, sort_df


def plot_results(y_train, y_test, predictions, model_names, save_path='forecast_comparison.png'):
    """
    Plot actual vs predicted values for each model.
    
    Parameters:
    -----------
    y_train : Series or array
        Training data
    y_test : Series or array
        Test data for comparison
    predictions : list of arrays
        List of predicted values from each model
    model_names : list of strings
        Names of the models corresponding to predictions
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Plot training data
    plt.plot(y_train.index, y_train, 'b-', label='Training Data')
    
    # Plot testing data
    plt.plot(y_test.index, y_test, 'g-', label='Actual Test Data')
    
    # Plot predictions from each model
    colors = ['r-', 'm-', 'c-', 'y-', 'k-', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        color = colors[i % len(colors)]
        plt.plot(y_test.index, pred, color, label=f'{name} Predictions')
    
    plt.title('Time Series Forecasting Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Forecast comparison plot saved as '{save_path}'")

def create_trellis_plot(y_train, y_test, predictions, model_names, results_df, save_path='trellis_plot.png', max_models=9):
    """
    Create a trellis plot showing the top models' predictions in a 3x3 grid.

    Parameters:
    -----------
    y_train : Series
        Training data
    y_test : Series
        Test data
    predictions : list of arrays
        List of predicted values from each model
    model_names : list of strings
        Names of the models corresponding to predictions
    results_df : DataFrame
        DataFrame with model performance metrics
    save_path : str
        Path to save the plot
    max_models : int
        Maximum number of models to show (default 9, in a 3x3 grid)
    """
    # Top 9 by composite score (fallback: RMSE). Match by model name so order matches predictions/model_names.
    sort_col = 'composite_score' if 'composite_score' in results_df.columns else 'rmse'
    ascending = False if sort_col == 'composite_score' else True
    sorted_df = results_df.sort_values(sort_col, ascending=ascending).head(max_models)
    top_names = sorted_df['model'].tolist()
    name_to_idx = {name: i for i, name in enumerate(model_names)}
    run_order_indices = [name_to_idx[n] for n in top_names if n in name_to_idx]
    sorted_predictions = [predictions[i] for i in run_order_indices]
    sorted_model_names = [model_names[i] for i in run_order_indices]
    n_models = len(sorted_model_names)

    # Fixed 3x3 grid for top 9
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot data for each of the top models
    for i, (pred, name, ax) in enumerate(zip(sorted_predictions, sorted_model_names, axes)):
        # Get metrics
        model_metrics = results_df[results_df['model'] == name].iloc[0]
        rmse = model_metrics['rmse']
        r2 = model_metrics['r2']
        composite = model_metrics.get('composite_score', np.nan)
        composite_str = f"Composite: {composite:.3f} | " if pd.notna(composite) else ""

        # Plot training data (partially transparent)
        ax.plot(y_train.index, y_train, 'b-', alpha=0.3, label='Training Data')

        # Plot test data and predictions
        ax.plot(y_test.index, y_test, 'g-', label='Actual Test Data')
        ax.plot(y_test.index, pred, 'r-', label='Predictions')

        # Add title with metrics
        ax.set_title(f"{name}\n{composite_str}RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Only add legend to the first plot to save space
        if i == 0:
            ax.legend(loc='upper left')
            
        # Add x-label only to bottom row
        if i >= (rows - 1) * cols:
            ax.set_xlabel('Date')
            
        # Add y-label only to leftmost column
        if i % cols == 0:
            ax.set_ylabel('Value')
            
        # Add grid
        ax.grid(True, alpha=0.3)

    # Hide unused subplots (if fewer than 9 models)
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    plt.suptitle('Top 9 Models (3×3) - Individual Performance', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Trellis plot saved as '{save_path}'")

def create_top_models_plot(y_train, y_test, predictions, model_names, save_path='top_models.png'):
    """
    Create a detailed comparison plot for the top performing models.
    Focuses the x-axis on the prediction (test) period so it is readable.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use only a trailing portion of training so the test period gets most of the x-axis
    train_frac = 0.15  # show last 15% of training for context
    n_train = len(y_train)
    train_tail_start = max(0, int((1 - train_frac) * n_train))
    train_index = y_train.index
    x_min = train_index[train_tail_start]
    x_max = y_test.index[-1]

    # Plot training (tail only) and full test + predictions
    ax.plot(y_train.index, y_train, 'k-', alpha=0.2, label='Training Data')
    ax.plot(y_test.index, y_test, 'k-', linewidth=2, label='Actual Test Data')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        ax.plot(y_test.index, pred, color=colors[i], marker=markers[i],
                markersize=6, markevery=max(1, len(y_test) // 20),
                linewidth=1.5, label=f"{name} Predictions")

    ax.set_xlim(x_min, x_max)
    ax.set_title('Top 3 Models Comparison', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Inset: zoom on middle of test period with readable date ticks
    mid = len(y_test) // 2
    half = len(y_test) // 4
    axins = zoomed_inset_axes(ax, 2.5, loc='upper right')
    axins.plot(y_test.index[mid - half:mid + half], y_test.iloc[mid - half:mid + half], 'k-', linewidth=2)
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        axins.plot(y_test.index[mid - half:mid + half], pred[mid - half:mid + half],
                   color=colors[i], marker=markers[i], markersize=4, markevery=5, linewidth=1.5)
    axins.set_xlim(y_test.index[mid - half], y_test.index[mid + half - 1])
    axins.tick_params(axis='x', rotation=45, labelsize=8)
    axins.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Top models comparison plot saved as '{save_path}'")

def create_performance_chart(results_df, save_path='model_performance.png'):
    """
    Create a horizontal bar chart comparing model performance.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame with model performance metrics
    save_path : str
        Path to save the plot
    """
    # Sort by composite score if available, else RMSE; include mase/mape for panels
    base_cols = ['model', 'rmse', 'mae', 'r2']
    if 'composite_score' in results_df.columns:
        base_cols = ['model', 'composite_score', 'rmse', 'mae', 'r2']
    optional = ['mase', 'mape']
    all_cols = [c for c in base_cols + optional if c in results_df.columns]
    metrics_df = results_df[all_cols].copy()
    if 'composite_score' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('composite_score', ascending=False)
    else:
        metrics_df = metrics_df.sort_values('rmse')
    metrics_df = metrics_df.reset_index(drop=True)

    # Set up colors based on performance
    top_3_mask = metrics_df.index < 3
    colors = np.where(top_3_mask, '#1f77b4', '#d3d3d3')  # Blue for top 3, gray for others

    # Build list of panels: composite, rmse, mae, r2, mase, mape (include all available)
    has_composite = 'composite_score' in metrics_df.columns
    has_mase = 'mase' in metrics_df.columns
    has_mape = 'mape' in metrics_df.columns
    n_plots = (4 if has_composite else 3) + (1 if has_mase else 0) + (1 if has_mape else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    # Composite score plot (higher is better) - first if available
    ax_idx = 0
    if has_composite:
        axes[ax_idx].barh(metrics_df['model'], metrics_df['composite_score'], color=colors)
        axes[ax_idx].set_title('Composite Score (higher is better)', fontsize=14)
        axes[ax_idx].set_xlabel('Score', fontsize=12)
        axes[ax_idx].grid(True, alpha=0.3, axis='x')
        axes[ax_idx].invert_yaxis()
        for i, v in enumerate(metrics_df['composite_score']):
            axes[ax_idx].text(v + 0.02, i, f"{v:.2f}", va='center')
        ax_idx += 1

    # RMSE plot (lower is better)
    axes[ax_idx].barh(metrics_df['model'], metrics_df['rmse'], color=colors)
    axes[ax_idx].set_title('RMSE (lower is better)', fontsize=14)
    axes[ax_idx].set_xlabel('Root Mean Squared Error', fontsize=12)
    axes[ax_idx].grid(True, alpha=0.3, axis='x')
    axes[ax_idx].invert_yaxis()
    for i, v in enumerate(metrics_df['rmse']):
        axes[ax_idx].text(v + 5, i, f"{v:.2f}", va='center')
    ax_idx += 1

    # MAE plot (lower is better)
    axes[ax_idx].barh(metrics_df['model'], metrics_df['mae'], color=colors)
    axes[ax_idx].set_title('MAE (lower is better)', fontsize=14)
    axes[ax_idx].set_xlabel('Mean Absolute Error', fontsize=12)
    axes[ax_idx].grid(True, alpha=0.3, axis='x')
    axes[ax_idx].invert_yaxis()
    for i, v in enumerate(metrics_df['mae']):
        axes[ax_idx].text(v + 5, i, f"{v:.2f}", va='center')
    ax_idx += 1

    # R² plot (higher is better)
    r2_colors = np.where(metrics_df['r2'] < 0, '#d62728', colors)  # Red for negative R²
    axes[ax_idx].barh(metrics_df['model'], metrics_df['r2'], color=r2_colors)
    axes[ax_idx].set_title('R² (higher is better)', fontsize=14)
    axes[ax_idx].set_xlabel('R-squared', fontsize=12)
    axes[ax_idx].grid(True, alpha=0.3, axis='x')
    axes[ax_idx].invert_yaxis()
    for i, v in enumerate(metrics_df['r2']):
        axes[ax_idx].text(v + 0.05, i, f"{v:.2f}", va='center')
    ax_idx += 1

    # MASE plot (lower is better; accent for MASE > 1)
    if has_mase:
        mase_vals = metrics_df['mase'].replace([np.inf, -np.inf], np.nan).fillna(10)
        mase_colors = np.where(mase_vals > 1, '#ff7f0e', colors)  # Orange when worse than naïve
        axes[ax_idx].barh(metrics_df['model'], mase_vals, color=mase_colors)
        axes[ax_idx].axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        axes[ax_idx].set_title('MASE (lower is better, <1 beats naïve)', fontsize=14)
        axes[ax_idx].set_xlabel('MASE', fontsize=12)
        axes[ax_idx].grid(True, alpha=0.3, axis='x')
        axes[ax_idx].invert_yaxis()
        for i, v in enumerate(metrics_df['mase']):
            if pd.notna(v) and not np.isinf(v):
                axes[ax_idx].text(float(v) + 0.05, i, f"{v:.2f}", va='center')
        ax_idx += 1

    # MAPE plot (lower is better; clip display at 200%)
    if has_mape:
        mape_vals = metrics_df['mape'].replace([np.inf, -np.inf], np.nan).clip(upper=200)
        axes[ax_idx].barh(metrics_df['model'], mape_vals, color=colors)
        axes[ax_idx].set_title('MAPE % (lower is better)', fontsize=14)
        axes[ax_idx].set_xlabel('MAPE (%)', fontsize=12)
        axes[ax_idx].grid(True, alpha=0.3, axis='x')
        axes[ax_idx].invert_yaxis()
        for i, v in enumerate(metrics_df['mape']):
            if pd.notna(v) and not np.isinf(v):
                disp_v = min(float(v), 200)
                axes[ax_idx].text(disp_v + 1, i, f"{v:.1f}%", va='center')
        ax_idx += 1

    # Add overall title
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Performance chart saved as '{save_path}'")


def _normalized_scores_for_radar(df):
    """Return DataFrame with columns model + composite_score + _s_rmse, _s_mae, _s_r2, _s_mase, _s_mape (0-1, 1=best)."""
    out = df[['model']].copy()
    if 'composite_score' in df.columns:
        out['composite_score'] = df['composite_score']
    # RMSE: lower better -> invert
    r_min, r_max = df['rmse'].min(), df['rmse'].max()
    out['_s_rmse'] = 1 - (df['rmse'] - r_min) / (r_max - r_min) if r_max > r_min else 1.0
    m_min, m_max = df['mae'].min(), df['mae'].max()
    out['_s_mae'] = 1 - (df['mae'] - m_min) / (m_max - m_min) if m_max > m_min else 1.0
    r2_min, r2_max = df['r2'].min(), df['r2'].max()
    out['_s_r2'] = (df['r2'] - r2_min) / (r2_max - r2_min) if r2_max > r2_min else (1.0 if df['r2'].max() >= 0 else 0.0)
    if 'mase' in df.columns:
        mase = df['mase'].replace([np.inf, -np.inf], np.nan).clip(upper=10)
        valid = mase.notna()
        m_min_m, m_max_m = mase.min(), mase.max()
        out['_s_mase'] = np.where(valid, 1 - (mase - m_min_m) / (m_max_m - m_min_m) if m_max_m > m_min_m else 1.0, 0.5)
    else:
        out['_s_mase'] = 0.5
    if 'mape' in df.columns:
        mape = df['mape'].replace([np.inf, -np.inf], np.nan).clip(upper=200)
        valid = mape.notna()
        p_min, p_max = mape.min(), mape.max()
        out['_s_mape'] = np.where(valid, 1 - (mape - p_min) / (p_max - p_min) if p_max > p_min else 1.0, 0.5)
    else:
        out['_s_mape'] = 0.5
    return out


def create_radar_chart(results_df, save_path='model_radar.png', max_models=5):
    """
    Create a radar (spider) chart of normalized metrics for the top N models.
    Axes: Composite, RMSE, MAE, R², MASE, MAPE (all 0-1, 1=best).
    """
    sort_col = 'composite_score' if 'composite_score' in results_df.columns else 'rmse'
    ascending = False if sort_col == 'composite_score' else True
    top = results_df.sort_values(sort_col, ascending=ascending).head(max_models)
    if len(top) == 0:
        return
    norm = _normalized_scores_for_radar(top)
    labels = ['Composite', 'RMSE', 'MAE', 'R²', 'MASE', 'MAPE']
    cols = ['composite_score', '_s_rmse', '_s_mae', '_s_r2', '_s_mase', '_s_mape']
    if 'composite_score' not in norm.columns:
        labels = labels[1:]
        cols = cols[1:]
    n_axes = len(cols)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = plt.cm.tab10(np.linspace(0, 1, max_models))
    for i, (_, row) in enumerate(norm.iterrows()):
        vals = [row[c] for c in cols]
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(angles, vals, 'o-', linewidth=1.5, label=row['model'], color=colors[i % max_models])
        ax.fill(angles, vals, alpha=0.15, color=colors[i % max_models])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.set_title('Top models – normalized metrics (1 = best)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Radar chart saved as '{save_path}'")


def create_residuals_plot(y_test, predictions, model_names, save_path='top_3_residuals.png', rolling_window=5):
    """
    Plot residuals (predicted - actual) for the top 3 models; optional rolling mean to show bias.
    """
    if not predictions or not model_names:
        return
    n = min(3, len(predictions), len(model_names))
    y_test = np.asarray(y_test)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        res = np.asarray(predictions[i]) - y_test
        axes[i].axhline(0, color='gray', linestyle='--', alpha=0.8)
        axes[i].plot(res, 'o-', markersize=3, alpha=0.7, label='Residual')
        if len(res) >= rolling_window:
            roll = pd.Series(res).rolling(rolling_window, center=True).mean()
            axes[i].plot(roll.values, '-', linewidth=2, label=f'Rolling mean (w={rolling_window})')
        axes[i].set_ylabel('Residual')
        axes[i].set_title(model_names[i])
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time index')
    plt.suptitle('Top 3 models – residuals (predicted − actual)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Residuals plot saved as '{save_path}'")


def plot_feature_importance(results_dir, save_path='feature_importance.png'):
    """
    Read rf_feature_importance.csv and/or xgb_feature_importance.csv from results_dir
    and plot horizontal bar charts (one subplot per model).
    """
    patterns = ['rf_feature_importance.csv', 'xgb_feature_importance.csv']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(results_dir, p)))
    if not files:
        return
    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, files):
        df = pd.read_csv(path)
        if 'feature' not in df.columns or 'importance' not in df.columns:
            continue
        df = df.sort_values('importance', ascending=True).tail(20)
        ax.barh(df['feature'], df['importance'], color='#1f77b4', alpha=0.8)
        ax.set_xlabel('Importance')
        ax.set_title(os.path.basename(path).replace('_feature_importance.csv', '').upper())
        ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved as '{save_path}'")