#!/usr/bin/env python3
import argparse
import os
import json
import logging
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils import setup_logging
from src.data_handling import load_data, prepare_features, split_data
from src.evaluation import (
    evaluate_model, plot_results, create_trellis_plot, 
    create_top_models_plot, create_performance_chart
)
from src.models.registry import get_available_models, get_tuning_functions

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
                      help='Models to run (all, arima, sarima, es, prophet, rf, svr, xgb, ma, rnn, lstm)')
    parser.add_argument('--tune_top', type=int, default=3, 
                   help='Number of top models to tune (default: 3, set to 0 to disable tuning)')
    parser.add_argument('--tune_all', action='store_true', 
                   help='Tune all models instead of just the top performing ones')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logger.info(f"Loading data from {args.file}")
    logger.info(f"Time column: {args.time_col}")
    logger.info(f"Data column: {args.data_col}")
    
    # Load and prepare data
    try:
        df = load_data(args.file, args.time_col, args.data_col, args.date_format)
        logger.info(f"Data loaded successfully with {len(df)} records")
        
        df_features = prepare_features(df, args.data_col, lags=args.lags)
        logger.info(f"Features prepared with {len(df_features)} records after handling lags")
        
        X_train, X_test, y_train, y_test, train_df, test_df = split_data(df_features, args.data_col, test_size=args.test_size)
        logger.info(f"Data split into {len(train_df)} training and {len(test_df)} testing records")
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        return
    
    # Get available models and tuning functions from registry
    available_models = get_available_models()
    tuning_functions = get_tuning_functions()
    
    all_predictions, model_names, results = run_models(
        args, available_models, tuning_functions,
        X_train, X_test, y_train, y_test, train_df, test_df,
        perform_tuning=False  # First run without tuning
    )
    
    # Create initial results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    
    # Determine which models to tune
    models_to_tune = []
    
    if args.tune_all:
        # Tune all models
        models_to_tune = list(tuning_functions.keys())
    elif args.tune_top > 0:
        # Get the top N performing models
        top_models = results_df.head(args.tune_top)
        for _, row in top_models.iterrows():
            model_key = row['model'].lower().split()[0]
            if model_key in tuning_functions and 'tuned' not in row['model'].lower():
                models_to_tune.append(model_key)
    
    # Tune selected models
    if models_to_tune:
        logger.info(f"Tuning the following models: {', '.join(models_to_tune)}")
        
        # Call a function to perform tuning on the selected models
        tuned_predictions, tuned_names, tuned_results = tune_selected_models(
            args, models_to_tune, tuning_functions,
            X_train, X_test, y_train, y_test, train_df, test_df
        )
        
        # Add tuned models to the results
        all_predictions.extend(tuned_predictions)
        model_names.extend(tuned_names)
        results.extend(tuned_results)
    
    # Process final results
    process_results(
        args, all_predictions, model_names, results,
        y_train, y_test, X_train, X_test, tuning_functions
    )

    # Process results
    process_results(
        args, all_predictions, model_names, results,
        y_train, y_test, X_train, X_test, tuning_functions
    )

def run_models(args, available_models, tuning_functions, X_train, X_test, y_train, y_test, train_df, test_df):
    """Run selected models with or without tuning."""
    logger = logging.getLogger(__name__)
    
    # Default to all models if --models flag is not provided
    models_to_run = list(available_models.keys()) if 'all' in args.models else [m.lower() for m in args.models]
    
    # Store model predictions and names
    all_predictions = []
    model_names = []
    results = []
    
    # Determine seasonal periods if needed
    seasonal_periods = determine_seasonal_periods(y_train)
    
    # If tune_all is True, perform hyperparameter tuning for all models
    if args.tune_all:
        logger.info("Performing hyperparameter tuning for all models")
        
        for model_key in models_to_run:
            if model_key not in tuning_functions:
                logger.warning(f"Tuning not implemented for model '{model_key}'. Skipping.")
                continue
                
            try:
                logger.info(f"Tuning {model_key.upper()} model...")
                
                # Execute tuning function based on model type
                model_params = {
                    'y_train': y_train, 
                    'y_test': y_test,
                    'X_train': X_train,
                    'X_test': X_test,
                    'n_steps': args.n_steps,
                    'seasonal_periods': seasonal_periods,
                    'ma_window': args.ma_window
                }
                
                pred, name, best_params = execute_tuning(model_key, tuning_functions, model_params)
                
                all_predictions.append(pred)
                model_names.append(name)
                results.append(evaluate_model(y_test, pred, name))
                
            except Exception as e:
                logger.error(f"Error tuning {model_key.upper()} model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        # Run selected models without tuning
        for model_key in models_to_run:
            if model_key not in available_models:
                logger.warning(f"Model '{model_key}' not recognized. Skipping.")
                continue
                
            try:
                logger.info(f"Running {model_key.upper()} model...")
                
                # Execute model function based on model type
                model_params = {
                    'y_train': y_train, 
                    'y_test': y_test,
                    'X_train': X_train,
                    'X_test': X_test,
                    'n_steps': args.n_steps,
                    'seasonal_periods': seasonal_periods,
                    'ma_window': args.ma_window
                }
                
                pred, name = execute_model(model_key, available_models, model_params)
                
                all_predictions.append(pred)
                model_names.append(name)
                results.append(evaluate_model(y_test, pred, name))
                
            except Exception as e:
                logger.error(f"Error with {model_key.upper()} model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    return all_predictions, model_names, results

def process_results(args, all_predictions, model_names, results, y_train, y_test, X_train, X_test, tuning_functions):
    """Process and display the results of model forecasting."""
    logger = logging.getLogger(__name__)
    
    if not all_predictions:
        logger.error("No models were successfully trained. Check the logs for errors.")
        return
    
    # Get dataset name from file path
    dataset_name = os.path.basename(args.file).split('.')[0]
    
    # Create results directory with dataset name
    dataset_results_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # Create a results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    
    # Save basic results to CSV
    results_csv_path = os.path.join(dataset_results_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to '{results_csv_path}'")
    
    # Get top 3 models
    top_models_df = results_df.head(3)
    top_model_names = top_models_df['model'].tolist()
    
    # Extract indices of top models from the full list
    top_model_indices = [model_names.index(name) for name in top_model_names]
    top_model_predictions = [all_predictions[i] for i in top_model_indices]
    
    # Create trellis plot for all models
    create_trellis_plot(
        y_train, y_test, all_predictions, model_names, results_df,
        os.path.join(dataset_results_dir, 'all_models_trellis.png')
    )
    
    # Create detailed comparison plot for top 3 models
    create_top_models_plot(
        y_train, y_test, top_model_predictions, top_model_names, 
        os.path.join(dataset_results_dir, 'top_3_models_comparison.png')
    )
    
    # Create performance bar chart
    create_performance_chart(
        results_df, 
        os.path.join(dataset_results_dir, 'model_performance.png')
    )
    
    # Determine the best model
    best_model_result = min(results, key=lambda x: x['rmse'])
    best_model_name = best_model_result['model']
    best_model_key = best_model_name.lower().split()[0]
    
    logger.info(f"\nBest model: {best_model_name} with RMSE: {best_model_result['rmse']:.4f}")
    
    # Save top models information
    top_models_info = {}
    
    # Add dataset information
    top_models_info['dataset'] = {
        'name': dataset_name,
        'file': args.file,
        'time_column': args.time_col,
        'data_column': args.data_col,
        'total_samples': len(y_train) + len(y_test),
        'training_samples': len(y_train),
        'testing_samples': len(y_test)
    }
    
    # Add model information
    top_models_info['models'] = []
    for idx, (_, row) in enumerate(top_models_df.iterrows()):
        model_info = {
            'rank': idx + 1,
            'name': row['model'],
            'rmse': float(row['rmse']),
            'mae': float(row['mae']),
            'r2': float(row['r2'])
        }
        
        # Try to add hyperparameters if available (for tuned models)
        if 'tuned' in row['model'].lower():
            model_key = row['model'].lower().split()[0]
            model_info['tuned'] = True
            # We'd need to capture hyperparameters during tuning
            # This is just a placeholder
            model_info['hyperparameters'] = "Hyperparameters would be stored here"
        
        top_models_info['models'].append(model_info)
    
    # Save top models information
    top_models_file = os.path.join(dataset_results_dir, 'top_models_info.json')
    with open(top_models_file, 'w') as f:
        json.dump(top_models_info, f, indent=4)
    
    logger.info(f"Top models information saved to '{top_models_file}'")
    
    # Display final results
    logger.info("\nFinal Model Comparison Results:")
    print("\nModel Performance Summary (sorted by RMSE):")
    print(results_df[['model', 'rmse', 'mae', 'r2']].to_string(index=False))
    
    # Print top 3 models summary
    print("\nTop 3 Models:")
    for i, (_, row) in enumerate(top_models_df.iterrows()):
        print(f"{i+1}. {row['model']} - RMSE: {row['rmse']:.4f}, MAE: {row['mae']:.4f}, R²: {row['r2']:.4f}")

def tune_selected_models(args, models_to_tune, tuning_functions, 
                        X_train, X_test, y_train, y_test, train_df, test_df):
    """Tune selected models and return their predictions."""
    logger = logging.getLogger(__name__)
    
    tuned_predictions = []
    tuned_names = []
    tuned_results = []
    
    # Get dataset name from file path
    dataset_name = os.path.basename(args.file).split('.')[0]
    
    # Create results directory with dataset name
    dataset_results_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # Determine seasonal periods if needed
    seasonal_periods = determine_seasonal_periods(y_train)
    
    for model_key in models_to_tune:
        if model_key not in tuning_functions:
            logger.warning(f"Tuning not implemented for model '{model_key}'. Skipping.")
            continue
            
        try:
            logger.info(f"Tuning {model_key.upper()} model...")
            
            # Execute tuning function based on model type
            model_params = {
                'y_train': y_train, 
                'y_test': y_test,
                'X_train': X_train,
                'X_test': X_test,
                'n_steps': args.n_steps,
                'seasonal_periods': seasonal_periods,
                'ma_window': args.ma_window
            }
            
            pred, name, best_params = execute_tuning(model_key, tuning_functions, model_params)
            
            tuned_predictions.append(pred)
            tuned_names.append(name)
            tuned_results.append(evaluate_model(y_test, pred, name))
            
            # Save best parameters to file - in dataset-specific directory
            params_file = os.path.join(dataset_results_dir, f"{model_key}_best_params.json")
            with open(params_file, 'w') as f:
                json.dump(best_params, f, indent=4)
                
            logger.info(f"Best parameters for {model_key.upper()} saved to {params_file}")
            
        except Exception as e:
            logger.error(f"Error tuning {model_key.upper()} model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    return tuned_predictions, tuned_names, tuned_results

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

def tune_best_model(args, best_model_key, best_model_name, best_model_result, 
                   tuning_functions, X_train, X_test, y_train, y_test,
                   all_predictions, model_names, results, results_df, results_csv_path):
    """Tune the best performing model."""
    logger = logging.getLogger(__name__)
    
    # Check if tuning is implemented for this model
    if best_model_key in tuning_functions:
        logger.info(f"Performing hyperparameter tuning for the best model: {best_model_name}")
        
        try:
            # Execute tuning function
            seasonal_periods = determine_seasonal_periods(y_train)
            
            model_params = {
                'y_train': y_train, 
                'y_test': y_test,
                'X_train': X_train,
                'X_test': X_test,
                'n_steps': args.n_steps,
                'seasonal_periods': seasonal_periods,
                'ma_window': args.ma_window
            }
            
            tuned_pred, tuned_model_name, best_params = execute_tuning(
                best_model_key, tuning_functions, model_params
            )
            
            # Evaluate tuned model
            tuned_result = evaluate_model(y_test, tuned_pred, tuned_model_name)
            
            # Add tuned model to results
            all_predictions.append(tuned_pred)
            model_names.append(tuned_model_name)
            results.append(tuned_result)
            
            # Update results DataFrame
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('rmse')
            results_df.to_csv(results_csv_path, index=False)
            
            # Update plot with tuned model
            plot_path = os.path.join(args.output_dir, 'forecast_comparison_with_tuned.png')
            plot_results(y_train, y_test, all_predictions, model_names, plot_path)
            
            # Compare original vs tuned performance
            improvement = ((best_model_result['rmse'] - tuned_result['rmse']) / best_model_result['rmse']) * 100
            
            logger.info(f"\nTuned model: {tuned_model_name} with RMSE: {tuned_result['rmse']:.4f}")
            logger.info(f"Performance improvement: {improvement:.2f}%")
            
        except Exception as e:
            logger.error(f"Error tuning {best_model_key.upper()} model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning(f"Hyperparameter tuning not implemented for model: {best_model_key}")

if __name__ == "__main__":
    main()