#!/usr/bin/env python3
"""
Generate sample retail sales data with seasonality, trend, and holidays.
This creates realistic time series data that mimics retail sales patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def generate_retail_sales(start_date='2018-01-01', periods=730, freq='D'):
    """
    Generate synthetic retail sales data with the following components:
    - Upward trend
    - Weekly seasonality (weekend peaks)
    - Monthly seasonality (end of month peaks)
    - Yearly seasonality (holiday season peaks in Nov-Dec)
    - Random noise
    - Special events/holidays with impact
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    periods : int
        Number of periods (days) to generate
    freq : str
        Frequency of time series ('D' for daily)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with date index and sales column
    """
    # Create date range
    date_rng = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Initialize DataFrame
    df = pd.DataFrame(date_rng, columns=['date'])
    
    # Extract date features
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year
    
    # Base sales (starting level)
    base_sales = 1000
    
    # Linear trend component
    trend_growth = 0.5  # daily growth rate
    df['trend'] = df.index * trend_growth
    
    # Weekly seasonality (weekend effect)
    weekly_pattern = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.1, 4: 1.2, 5: 1.5, 6: 1.3}  # Mon to Sun
    df['weekly_seasonal'] = df['dayofweek'].map(weekly_pattern)
    
    # Monthly seasonality (end of month has higher sales)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int) * 0.3 + 1
    
    # Yearly seasonality (holiday season effect)
    yearly_pattern = {1: 0.8, 2: 0.8, 3: 0.9, 4: 0.9, 5: 1.0, 6: 1.0, 
                     7: 1.0, 8: 1.1, 9: 1.1, 10: 1.2, 11: 1.5, 12: 1.7}
    df['yearly_seasonal'] = df['month'].map(yearly_pattern)
    
    # Special events/holidays
    holidays = {
        # Format: 'YYYY-MM-DD': impact_multiplier
        '2018-01-01': 0.7,  # New Year's Day
        '2018-02-14': 1.3,  # Valentine's Day
        '2018-05-28': 1.4,  # Memorial Day
        '2018-07-04': 0.7,  # Independence Day
        '2018-11-23': 2.5,  # Black Friday
        '2018-12-24': 2.0,  # Christmas Eve
        '2018-12-25': 0.5,  # Christmas Day
        '2019-01-01': 0.7,  # New Year's Day
        '2019-02-14': 1.3,  # Valentine's Day
        '2019-05-27': 1.4,  # Memorial Day
        '2019-07-04': 0.7,  # Independence Day
        '2019-11-29': 2.5,  # Black Friday
        '2019-12-24': 2.0,  # Christmas Eve
        '2019-12-25': 0.5,  # Christmas Day
    }
    
    # Convert to datetime for matching
    holidays_dates = {pd.Timestamp(date): impact for date, impact in holidays.items()}
    
    # Add holiday impact
    df['holiday_impact'] = 1.0
    for date, impact in holidays_dates.items():
        if date in df['date'].values:
            df.loc[df['date'] == date, 'holiday_impact'] = impact
    
    # Generate sales with all components combined
    df['sales'] = (base_sales + df['trend']) * df['weekly_seasonal'] * \
                 df['is_month_end'] * df['yearly_seasonal'] * df['holiday_impact']
    
    # Add random noise
    np.random.seed(42)  # for reproducibility
    noise = np.random.normal(1, 0.05, len(df))  # 5% noise
    df['sales'] = df['sales'] * noise
    
    # Round sales to whole numbers
    df['sales'] = df['sales'].round().astype(int)
    
    # Keep only date and sales columns for the final dataset
    result = df[['date', 'sales']].copy()
    
    return result

def main():
    """Generate and save sample retail sales data."""
    # Create samples directory if it doesn't exist
    os.makedirs('data/samples', exist_ok=True)
    
    # Generate retail sales data
    sales_data = generate_retail_sales(start_date='2018-01-01', periods=730)
    
    # Save to CSV
    output_path = 'data/samples/retail_sales_daily.csv'
    sales_data.to_csv(output_path, index=False)
    print(f"Retail sales data saved to {output_path}")
    
    # Generate a preview plot
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data['date'], sales_data['sales'])
    plt.title('Daily Retail Sales (2018-2019)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plot_path = 'data/samples/retail_sales_preview.png'
    plt.savefig(plot_path)
    print(f"Preview plot saved to {plot_path}")
    
    # Display sample data
    print("\nSample data (first 10 rows):")
    print(sales_data.head(10))
    
    print("\nData statistics:")
    print(sales_data['sales'].describe())

if __name__ == "__main__":
    main()