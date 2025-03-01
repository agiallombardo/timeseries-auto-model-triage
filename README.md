## Time Series Auto Model Triage
This framework automatically tests and evaluates multiple forecasting models on time series data, including statistical, machine learning, and deep learning approaches. It provides comprehensive visualizations and model selection capabilities.

The top 3 performing models will be selected for the given time series data and the corresponding hyperparameter tuning results.

## Author
Anthony Giallombardo & Claude 3.7

## Features
Supports 10 different forecasting models
Automatic hyperparameter tuning for top-performing models
Comprehensive visualization of results
Detailed performance metrics and comparisons
Dataset-specific result organization

** Sample Data
The package includes sample datasets in the data/samples directory:

Retail Sales Data - Daily retail sales with seasonal patterns, trends, and holiday effects
Energy Consumption Data - Hourly energy usage with multiple seasonal patterns and temperature effects

You can generate these sample datasets using:
`python data/samples/generate_retail_sales.py`
`python data/samples/generate_energy_consumption.py`

** Basic Usage
Run the forecaster with default settings on the retail sales sample:
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales`

This will:
Run all available forecasting models
Automatically tune the top 3 performing models
Generate visualizations and performance metrics
Save results in results/retail_sales_daily/

## Example Scenarios
** Forecasting Retail Sales
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales`
For this dataset, machine learning models like Random Forest and XGBoost typically outperform statistical models because they can capture complex patterns, including seasonal effects and holiday periods.

** Forecasting Energy Consumption
`python main.py --file data/samples/energy_consumption_hourly.csv --time_col timestamp --data_col energy_consumption`
This high-frequency data with multiple seasonality patterns (daily, weekly, yearly) benefits from models like LSTM and Prophet that can capture these complex temporal dependencies.

** Forecasting with Selected Models
To test only specific models:
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --models rf xgb prophet`

Tune all models (can be time-consuming):
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --tune_all`

Tune only the top 5 models (default is 3):
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --tune_top 5`

## Understanding Results
Each dataset's results are saved in a dedicated directory (results/{dataset_name}/):
all_models_trellis.png - Grid of plots showing each model's performance
top_3_models_comparison.png - Detailed comparison of the best models
model_performance.png - Bar charts comparing metrics across models
model_comparison_results.csv - Tabular performance metrics
top_models_info.json - Details about the top models and dataset
{model}_best_params.json - Optimized parameters for tuned models

## Supported Models

** Statistical Models
ARIMA (AutoRegressive Integrated Moving Average)
SARIMA (Seasonal ARIMA)
Moving Average
Exponential Smoothing

** Machine Learning Models
Random Forest
SVR (Support Vector Regression)
XGBoost

** Deep Learning Models
RNN (Recurrent Neural Network)
LSTM (Long Short-Term Memory)

** Other Models
Facebook Prophet

## Advanced Configuration
Adjust lag features for machine learning models:
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --lags 10

Set the test set size:
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --test_size 0.3

Configure sequence length for neural networks:
`python main.py --file data/samples/retail_sales_daily.csv --time_col date --data_col sales --n_steps 5
