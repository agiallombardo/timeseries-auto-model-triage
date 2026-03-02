import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os

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


def compute_best_judgment(results_df):
    """
    Evaluate all models holistically and produce a best judgment.

    Combines RMSE, MAE, R², MASE, and MAPE via min-max normalization so
    lower-error and higher-R² models score better. Returns the best model
    and a short explanation.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns: model, rmse, mae, r2, and optionally mase, mape

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
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    best_row = df.iloc[0]

    # Build judgment text
    name = best_row['model']
    rmse = best_row['rmse']
    r2 = best_row['r2']
    composite = best_row['composite_score']

    parts = [f"Recommended model: {name} (composite score: {composite:.3f})."]
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

    judgment_text = " ".join(parts)
    return best_row, judgment_text, df


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

def create_trellis_plot(y_train, y_test, predictions, model_names, results_df, save_path='trellis_plot.png'):
    """
    Create a trellis plot showing each model's predictions separately.
    
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
    """
    # Sort models by performance (composite score if available, else RMSE)
    sort_col = 'composite_score' if 'composite_score' in results_df.columns else 'rmse'
    ascending = False if sort_col == 'composite_score' else True
    sorted_indices = results_df.sort_values(sort_col, ascending=ascending).index.values
    sorted_predictions = [predictions[i] for i in sorted_indices]
    sorted_model_names = [model_names[i] for i in sorted_indices]
    
    # Calculate grid dimensions
    n_models = len(sorted_model_names)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows), sharex=True, sharey=True)
    
    # Flatten axes for easier iteration if there are multiple rows and columns
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1 and cols > 1:
        axes = np.array([axes])  # Make 1D array into 2D
    elif cols == 1 and rows > 1:
        axes = axes.reshape(-1, 1)  # Ensure axes is 2D
    else:
        axes = np.array([[axes]])  # Single plot case
    
    # Plot data for each model
    for i, (pred, name, ax) in enumerate(zip(sorted_predictions, sorted_model_names, axes.flatten())):
        # Get metrics
        model_metrics = results_df[results_df['model'] == name].iloc[0]
        rmse = model_metrics['rmse']
        r2 = model_metrics['r2']
        
        # Plot training data (partially transparent)
        ax.plot(y_train.index, y_train, 'b-', alpha=0.3, label='Training Data')
        
        # Plot test data and predictions
        ax.plot(y_test.index, y_test, 'g-', label='Actual Test Data')
        ax.plot(y_test.index, pred, 'r-', label='Predictions')
        
        # Add title with metrics
        ax.set_title(f"{name}\nRMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Only add legend to the first plot to save space
        if i == 0:
            ax.legend(loc='upper left')
            
        # Add x-label only to bottom row
        if i >= n_models - cols:
            ax.set_xlabel('Date')
            
        # Add y-label only to leftmost column
        if i % cols == 0:
            ax.set_ylabel('Value')
            
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_models, len(axes.flatten())):
        axes.flatten()[i].set_visible(False)
    
    # Add overall title
    plt.suptitle('Model Comparison - Individual Model Performance', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Trellis plot saved as '{save_path}'")

def create_top_models_plot(y_train, y_test, predictions, model_names, save_path='top_models.png'):
    """
    Create a detailed comparison plot for the top performing models.
    
    Parameters:
    -----------
    y_train : Series
        Training data
    y_test : Series
        Test data
    predictions : list of arrays
        List of predicted values from top models
    model_names : list of strings
        Names of the top models
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.plot(y_train.index, y_train, 'k-', alpha=0.2, label='Training Data')
    
    # Plot test data
    plt.plot(y_test.index, y_test, 'k-', linewidth=2, label='Actual Test Data')
    
    # Plot predictions for top models with different colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    markers = ['o', 's', '^']  # Circle, square, triangle
    
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        plt.plot(y_test.index, pred, color=colors[i], marker=markers[i], 
                 markersize=6, markevery=max(1, len(y_test)//20),
                 linewidth=1.5, label=f"{name} Predictions")
    
    # Add title and labels
    plt.title('Top 3 Models Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add zoom inset for detailed view
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    # Create inset axes (zoom in to middle 50% of test data)
    mid_point = len(y_test) // 2
    quarter = len(y_test) // 4
    
    axins = zoomed_inset_axes(plt.gca(), 2.5, loc='upper right')
    
    # Plot test data in inset
    axins.plot(y_test.index[mid_point-quarter:mid_point+quarter], 
              y_test[mid_point-quarter:mid_point+quarter], 'k-', linewidth=2)
    
    # Plot predictions in inset
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        axins.plot(y_test.index[mid_point-quarter:mid_point+quarter], 
                  pred[mid_point-quarter:mid_point+quarter], 
                  color=colors[i], marker=markers[i], markersize=4,
                  markevery=5, linewidth=1.5)
    
    # Set inset axes limits
    axins.set_xlim(y_test.index[mid_point-quarter], y_test.index[mid_point+quarter-1])
    
    # Connect inset to main plot
    mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # Save the plot (bbox_inches='tight' handles layout; tight_layout is
    # incompatible with zoomed_inset_axes and would produce a warning).
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
    # Sort by composite score if available, else RMSE
    base_cols = ['model', 'rmse', 'mae', 'r2']
    if 'composite_score' in results_df.columns:
        base_cols = ['model', 'composite_score', 'rmse', 'mae', 'r2']
    metrics_df = results_df[[c for c in base_cols if c in results_df.columns]].copy()
    if 'composite_score' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('composite_score', ascending=False)
    else:
        metrics_df = metrics_df.sort_values('rmse')
    
    # Set up colors based on performance
    top_3_mask = metrics_df.index < 3
    colors = np.where(top_3_mask, '#1f77b4', '#d3d3d3')  # Blue for top 3, gray for others
    
    # Create figure (4 columns if composite_score available)
    n_plots = 4 if 'composite_score' in metrics_df.columns else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 8))

    # Composite score plot (higher is better) - first if available
    ax_idx = 0
    if 'composite_score' in metrics_df.columns:
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
    
    # Add overall title
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance chart saved as '{save_path}'")