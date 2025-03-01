import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, model_name):
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
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Get logger within the function
    logger = logging.getLogger(__name__)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

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
    # Sort models by performance (RMSE)
    sorted_indices = results_df['rmse'].argsort().values
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
    
    # Save the plot
    plt.tight_layout()
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
    # Sort by RMSE
    metrics_df = results_df[['model', 'rmse', 'mae', 'r2']].copy()
    metrics_df = metrics_df.sort_values('rmse')
    
    # Set up colors based on performance
    top_3_mask = metrics_df.index < 3
    colors = np.where(top_3_mask, '#1f77b4', '#d3d3d3')  # Blue for top 3, gray for others
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # RMSE plot (lower is better)
    axes[0].barh(metrics_df['model'], metrics_df['rmse'], color=colors)
    axes[0].set_title('RMSE (lower is better)', fontsize=14)
    axes[0].set_xlabel('Root Mean Squared Error', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()  # Best model at top
    
    # Add values next to bars
    for i, v in enumerate(metrics_df['rmse']):
        axes[0].text(v + 5, i, f"{v:.2f}", va='center')
    
    # MAE plot (lower is better)
    axes[1].barh(metrics_df['model'], metrics_df['mae'], color=colors)
    axes[1].set_title('MAE (lower is better)', fontsize=14)
    axes[1].set_xlabel('Mean Absolute Error', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()  # Best model at top
    
    # Add values next to bars
    for i, v in enumerate(metrics_df['mae']):
        axes[1].text(v + 5, i, f"{v:.2f}", va='center')
    
    # R² plot (higher is better)
    r2_colors = np.where(metrics_df['r2'] < 0, '#d62728', colors)  # Red for negative R²
    axes[2].barh(metrics_df['model'], metrics_df['r2'], color=r2_colors)
    axes[2].set_title('R² (higher is better)', fontsize=14)
    axes[2].set_xlabel('R-squared', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='x')
    axes[2].invert_yaxis()  # Best model at top
    
    # Add values next to bars
    for i, v in enumerate(metrics_df['r2']):
        axes[2].text(v + 0.05, i, f"{v:.2f}", va='center')
    
    # Add overall title
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance chart saved as '{save_path}'")